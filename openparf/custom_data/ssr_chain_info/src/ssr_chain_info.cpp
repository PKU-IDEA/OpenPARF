/**
 * File              : ssr_chain_info.cpp
 * Author            : Jing Mai <jingmai@pku.edu.cn>
 * Date              : 12.02.2021
 * Last Modified Date: 12.02.2021
 * Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
 */
#include "ssr_chain_info.h"

#include <fstream>
#include <limits>
#include <queue>
#include <string>
#include <vector>

OPENPARF_BEGIN_NAMESPACE

using database::Design;
using database::Inst;
using database::Layout;
using database::Model;
using database::Net;
using database::Pin;
using database::PlaceDB;

namespace ssr_chain_info {

ChainInfoVec MakeChainInfoVecFromPlaceDB(const PlaceDB &placedb) {
  ChainInfoVec       rv;
  const auto         db              = placedb.db();
  const Design      &design          = db->design();
  const Layout      &layout          = db->layout();
  auto               top_module_inst = design.topModuleInst();
  auto const        &netlist         = top_module_inst->netlist();
  const auto        &place_params    = placedb.place_params();
  const std::string &module_name     = place_params.ssr_chain_module_name_;
  int32_t            module_id       = design.modelId(module_name);

  openparfAssert(module_id != std::numeric_limits<decltype(module_id)>::max());

  int32_t           num_insts = netlist.numInsts();
  int32_t           count     = 0;
  std::vector<bool> visited_mark(num_insts, false);
  auto              IsVisited = [&visited_mark](const Inst &inst) -> bool { return visited_mark[inst.id()] == true; };
  auto              IsChain   = [&module_id](const Inst &inst) -> bool { return inst.attr().modelId() == module_id; };

  openparfPrint(kDebug, "ssr_chain_module_name: %s(%i)\n", module_name.c_str(), module_id);

  for (int32_t new_inst_id = 0; new_inst_id < num_insts; new_inst_id++) {
    int32_t     old_inst_id = placedb.oldInstId(new_inst_id);
    const Inst &inst        = netlist.inst(old_inst_id);
    if (!IsChain(inst) || IsVisited(inst)) continue;
    rv.push_back(detail::ExtractChainFromOneInst(placedb, module_id, inst, visited_mark));
  }

  if (false) {
    // debug info
    std::ofstream of("ssr_chain_info.txt");
    for (const auto &chain_info : rv) {
      of << "===" << std::endl;
      for (const auto &new_id : chain_info.inst_new_ids()) {
        int32_t     old_id = placedb.oldInstId(new_id);
        const auto &inst   = netlist.inst(old_id);
        of << inst.attr().name() << "(" << new_id << ") ";
      }
      of << std::endl;
    }
    of.close();
  }
  return rv;
}

FlatNestedVectorInt MakeNestedNewIdx(const ChainInfoVec &chain_info_vec) {
  FlatNestedVectorInt nested_inst_ids;
  {
    int32_t count = 0;
    for (auto const &chain_info : chain_info_vec) {
      const auto &new_ids = chain_info.inst_new_ids();
      nested_inst_ids.pushBackIndexBegin(count);
      for (auto const &new_id : new_ids) {
        nested_inst_ids.pushBack(new_id);
      }
      count += new_ids.size();
    }
  }
  return nested_inst_ids;
}

namespace detail {

ChainInfo ExtractChainFromOneInst(const PlaceDB     &placedb,
                                  int32_t            module_id,
                                  const Inst        &source_inst,
                                  std::vector<bool> &visited_mark) {
  const Design        &design          = placedb.db()->design();
  const Layout        &layout          = placedb.db()->layout();
  auto                 top_module_inst = design.topModuleInst();
  auto const          &netlist         = top_module_inst->netlist();
  const Model         &model           = design.model(module_id);
  ChainInfo            chain;
  auto                &ordinal_inst_ids = chain.inst_ids();
  auto                &new_inst_ids      = chain.inst_new_ids();
  std::vector<int32_t> reversed_inst_ids;
  std::queue<int32_t>  que;
  auto IsVisited       = [&visited_mark](const Inst &inst) -> bool { return visited_mark[inst.id()] == true; };
  auto IsChain         = [&module_id](const Inst &inst) -> bool { return inst.attr().modelId() == module_id; };
  auto SetVisited      = [&visited_mark](const Inst &inst) { visited_mark[inst.id()] = true; };
  auto IsCascadedInPin = [&model](const Pin &pin) {
    const auto &model_pin = model.modelPin(pin.modelPinId());
    return model_pin.signalDirect() == SignalDirection::kInput && model_pin.signalType() == SignalType::kCascade;
  };
  auto IsCascadedOutPin = [&model](const Pin &pin) {
    const auto &model_pin = model.modelPin(pin.modelPinId());
    return model_pin.signalDirect() == SignalDirection::kOutput && model_pin.signalType() == SignalType::kCascade;
  };

  SetVisited(source_inst);
  openparfAssert(IsChain(source_inst));

  /* only search from cascaded output pin to cascaded input pin */ {
    ordinal_inst_ids.clear();
    que.push(source_inst.id());
    ordinal_inst_ids.push_back(source_inst.id());
    while (!que.empty()) {
      auto        inst_id = que.front();
      const Inst &inst    = netlist.inst(inst_id);
      que.pop();
      for (auto pin_id : inst.pinIds()) {
        const Pin &pin = netlist.pin(pin_id);
        if (!IsCascadedOutPin(pin)) continue;
        const Net &net = netlist.net(pin.netId());
        for (auto adjacent_pin_id : net.pinIds()) {
          const Pin  &adjacent_pin     = netlist.pin(adjacent_pin_id);
          auto        adjacent_inst_id = adjacent_pin.instId();
          const Inst &adjacent_inst    = netlist.inst(adjacent_inst_id);
          if (!IsChain(adjacent_inst) || IsVisited(adjacent_inst) || !IsCascadedInPin(adjacent_pin)) {
            continue;
          }
          SetVisited(adjacent_inst);
          que.push(adjacent_inst_id);
          ordinal_inst_ids.push_back(adjacent_inst_id);
        }
      }
    }
  }

  /* only search form cascaded input pin to cascaded output pin */ {
    reversed_inst_ids.clear();
    que.push(source_inst.id());
    while (!que.empty()) {
      auto        inst_id = que.front();
      const Inst &inst    = netlist.inst(inst_id);
      que.pop();
      for (auto pin_id : inst.pinIds()) {
        const Pin &pin = netlist.pin(pin_id);
        if (!IsCascadedInPin(pin)) continue;
        const Net &net = netlist.net(pin.netId());
        for (auto adjacent_pin_id : net.pinIds()) {
          const Pin  &adjacent_pin     = netlist.pin(adjacent_pin_id);
          auto        adjacent_inst_id = adjacent_pin.instId();
          const Inst &adjacent_inst    = netlist.inst(adjacent_inst_id);
          if (!IsChain(adjacent_inst) || IsVisited(adjacent_inst) || !IsCascadedOutPin(adjacent_pin)) {
            continue;
          }
          SetVisited(adjacent_inst);
          que.push(adjacent_inst_id);
          reversed_inst_ids.push_back(adjacent_inst_id);
        }
      }
    }
  }

  // combine the instances in two directions.
  std::reverse(reversed_inst_ids.begin(), reversed_inst_ids.end());
  reversed_inst_ids.insert(reversed_inst_ids.end(), ordinal_inst_ids.begin(), ordinal_inst_ids.end());
  ordinal_inst_ids = reversed_inst_ids;

  // transfer the old ids to the new ids
  new_inst_ids.clear();
  for (auto const &old_id : ordinal_inst_ids) {
    auto new_inst_id = placedb.newInstId(old_id);
    new_inst_ids.push_back(new_inst_id);
  }

  return chain;
}
}   // namespace detail

}   // namespace ssr_chain_info

OPENPARF_END_NAMESPACE
