#include "static_timing_analysis.h"

// c++ library headers
#include <algorithm>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

OPENPARF_BEGIN_NAMESPACE

namespace static_timing_analysis {

/**
 * @brief Static Timing Analysis based on the assumption that the clock skew is minor.
 *  We also return the ignored nets, i.e., clock nets & the VDD/VSS net via the parameter `ignored_net_masks`
 * @tparam T
 * @param placedb
 * @param pin_delays
 * @param[out] pin_arrivals
 * @param[out] pin_requires
 * @param net_mask_ignore_large
 * @param[out] ignored_net_masks
 * @param timing_period
 */
template<class T>
void StaticTimingAnalysis(const database::PlaceDB& placedb,
                          T*                       pin_delays,
                          T*                       pin_arrivals,
                          T*                       pin_requires,
                          uint8_t*                 net_mask_ignore_large,
                          bool*                    ignored_net_masks,
                          T                        timing_period) {
  struct TimingArc {
    int32_t target_inst_id;
    int32_t driver_pin_id;
    int32_t sink_pin_id;
    T       delay;
  };
  struct ReversedTimingArc {
    int32_t source_inst_id;
    int32_t driver_pin_id;
    int32_t sink_pin_id;
    T       delay;
  };

  auto                                        db              = placedb.db();
  auto&                                       design          = db->design();
  auto&                                       layout          = db->layout();
  auto                                        top_module_inst = design.topModuleInst();
  auto&                                       netlist         = top_module_inst->netlist();
  int32_t                                     num_pins        = netlist.numPins();
  int32_t                                     num_insts       = netlist.numInsts();
  int32_t                                     vdd_vss_net_id  = design.VddVssNetId();
  std::vector<bool>                           is_clock_terminals;
  std::vector<int32_t>                        clock_terminal_ids;
  std::vector<std::vector<TimingArc>>         edges(num_insts);
  std::vector<std::vector<ReversedTimingArc>> reversed_edges(num_insts);
  std::vector<T>                              inst_arrivals(num_insts, 0);
  std::vector<T>                              inst_requires(num_insts, std::numeric_limits<T>::max());

  auto                                        isClockNet = [&placedb](int32_t net_id) {
    return placedb.netIdToClockId(net_id) != InvalidIndex<database::PlaceDB::IndexType>::value;
  };
  auto isClockNetOrVddVssNetOrLargeNet = [&placedb, &vdd_vss_net_id, &net_mask_ignore_large](int32_t net_id) {
    return (placedb.netIdToClockId(net_id) != InvalidIndex<database::PlaceDB::IndexType>::value ||
            net_id == vdd_vss_net_id || net_mask_ignore_large[net_id] == 0);
  };

  // DEBUG(Jing Mai): dump inst name
  if (false) {
    std::ofstream of("inst_name.txt");
    DEFER({ of.close(); });
    for (int inst_id = 0; inst_id < num_insts; inst_id++) {
      const auto& inst = netlist.inst(inst_id);
      of << inst_id << ": " << inst.attr().name() << std::endl;
    }
  }

  // DEBUG(Jing Mai): dump pin
  if (false) {
    std::ofstream of("pin.txt");
    DEFER({ of.close(); });
    for (int pin_id = 0; pin_id < num_pins; pin_id++) {
      const auto& pin  = netlist.pin(pin_id);
      const auto& inst = netlist.inst(pin.instId());
      const auto& net  = netlist.net(pin.netId());
      of << pin_id << ": " << inst.attr().name() + "/" + toString(pin.attr().signalDirect()) + " " << net.attr().name()
         << std::endl;
    }
  }

  /* identify clock terminals */ {
    // any cell connecting clock signals is identified as clock terminals.
    is_clock_terminals.resize(num_insts, false);
    clock_terminal_ids.clear();
    for (auto const& net_id : netlist.netIds()) {
      if (isClockNet(net_id)) {
        const auto& net     = netlist.net(net_id);
        const auto& pin_ids = net.pinIds();
        for (const auto& pin_id : pin_ids) {
          const auto& pin             = netlist.pin(pin_id);
          int32_t     inst_id         = pin.instId();
          is_clock_terminals[inst_id] = true;
          clock_terminal_ids.push_back(inst_id);
        }
      }
    }
    // remove duplicated clock terminals ids
    std::sort(clock_terminal_ids.begin(), clock_terminal_ids.end());
    clock_terminal_ids.erase(std::unique(clock_terminal_ids.begin(), clock_terminal_ids.end()),
                             clock_terminal_ids.end());
  }


  /* build timing graph upon data path */ {
    for (auto const& net_id : netlist.netIds()) {
      if (isClockNetOrVddVssNetOrLargeNet(net_id)) {
        ignored_net_masks[net_id] = true;
        continue;
      }
      const auto& net           = netlist.net(net_id);
      const auto& pin_ids       = net.pinIds();
      int32_t     driver_pin_id = std::numeric_limits<int32_t>::max();
      int32_t     source_inst_id;

      for (const auto& pin_id : pin_ids) {
        const auto& pin = netlist.pin(pin_id);
        if (pin.attr().signalDirect() == SignalDirection::kOutput) {
          openparfAssert(driver_pin_id == std::numeric_limits<int32_t>::max());
          driver_pin_id  = pin_id;
          source_inst_id = pin.instId();
        }
      }
      openparfAssert(driver_pin_id != std::numeric_limits<int32_t>::max());

      for (const auto& sink_pin_id : pin_ids) {
        if (sink_pin_id != driver_pin_id) {
          const auto& pin            = netlist.pin(sink_pin_id);
          int32_t     target_inst_id = pin.instId();
          T           delay          = pin_delays[sink_pin_id];
          edges[source_inst_id].push_back({static_cast<int32_t>(target_inst_id),
                                           static_cast<int32_t>(driver_pin_id),
                                           static_cast<int32_t>(sink_pin_id),
                                           delay});
          reversed_edges[target_inst_id].push_back({static_cast<int32_t>(source_inst_id),
                                                    static_cast<int32_t>(driver_pin_id),
                                                    static_cast<int32_t>(sink_pin_id),
                                                    delay});
        }
      }
    }
  }

  // timing source = clock terminals + data Input
  auto IsTimingSource = [&](int32_t inst_id) {
    return is_clock_terminals[inst_id] || reversed_edges[inst_id].size() == 0;
  };
  // timing terminal = clock terminals + data Output
  auto IsTimingTerminal = [&](int32_t inst_id) { return is_clock_terminals[inst_id] || edges[inst_id].size() == 0; };

  /* pin arrivals */ {
    std::vector<int32_t> degrees(num_insts, 0);
    std::queue<int32_t>  que;

    for (int32_t inst_id = 0; inst_id < num_insts; inst_id++) {
      degrees[inst_id] = reversed_edges[inst_id].size();
    }

    for (int32_t inst_id = 0; inst_id < num_insts; inst_id++) {
      if (IsTimingSource(inst_id)) {
        degrees[inst_id] = 0;
        que.push(inst_id);
        inst_arrivals[inst_id] = 0;
      }
    }

    while (!que.empty()) {
      int32_t source_inst_id = que.front();
      que.pop();
      for (const TimingArc& arc : edges[source_inst_id]) {
        int32_t target_inst_id      = arc.target_inst_id;
        int32_t driver_pin_id       = arc.driver_pin_id;
        int32_t sink_pin_id         = arc.sink_pin_id;
        T       delay               = arc.delay;

        pin_arrivals[driver_pin_id] = inst_arrivals[source_inst_id];
        pin_arrivals[sink_pin_id]   = inst_arrivals[source_inst_id] + delay;

        if (!IsTimingSource(target_inst_id)) {
          inst_arrivals[target_inst_id] = std::max(inst_arrivals[target_inst_id], pin_arrivals[sink_pin_id]);
          degrees[target_inst_id]--;
          if (degrees[target_inst_id] == 0) {
            que.push(target_inst_id);
          }
        }
      }
    }
  }

  /* pin requires */ {
    std::vector<int32_t> degrees(num_insts, 0);
    std::queue<int32_t>  que;
    for (int32_t inst_id = 0; inst_id < num_insts; inst_id++) {
      degrees[inst_id] = edges[inst_id].size();
    }

    for (int32_t inst_id = 0; inst_id < num_insts; inst_id++) {
      for (const ReversedTimingArc& arc : reversed_edges[inst_id]) {
        int32_t driver_pin_id       = arc.driver_pin_id;
        pin_requires[driver_pin_id] = std::numeric_limits<T>::max();
      }
    }

    for (int32_t inst_id = 0; inst_id < num_insts; inst_id++) {
      if (IsTimingTerminal(inst_id)) {
        degrees[inst_id] = 0;
        que.push(inst_id);
        inst_requires[inst_id] = timing_period;
      }
    }

    while (!que.empty()) {
      int32_t target_inst_id = que.front();
      que.pop();
      for (const ReversedTimingArc& arc : reversed_edges[target_inst_id]) {
        int32_t source_inst_id      = arc.source_inst_id;
        int32_t driver_pin_id       = arc.driver_pin_id;
        int32_t sink_pin_id         = arc.sink_pin_id;
        T       delay               = arc.delay;

        pin_requires[sink_pin_id]   = inst_requires[target_inst_id];
        pin_requires[driver_pin_id] = std::min(pin_requires[driver_pin_id], inst_requires[target_inst_id] - delay);

        if (!IsTimingTerminal(source_inst_id)) {
          inst_requires[source_inst_id] = std::min(inst_requires[source_inst_id], pin_requires[driver_pin_id]);
          degrees[source_inst_id]--;
          if (degrees[source_inst_id] == 0) {
            que.push(source_inst_id);
          }
        }
      }
    }
  }

  // DEBUG(Jing Mai): dump the worst slack
  if (false) {
    T* pin_slacks   = reinterpret_cast<T*>(calloc(num_pins, sizeof(T)));
    T  worest_slack = std::numeric_limits<T>::max();
    DEFER({ free(pin_slacks); });
    for (int i = 0; i < num_pins; i++) {
      pin_slacks[i] = pin_requires[i] - pin_arrivals[i];
      worest_slack  = std::min(worest_slack, pin_slacks[i]);
    }
    std::cerr << "worest_slack: " << worest_slack << std::endl;
    for (int pin_id = 0; pin_id < num_pins; pin_id++) {
      if (pin_slacks[pin_id] != worest_slack) {
        continue;
      }
      const auto& pin     = netlist.pin(pin_id);
      int32_t     inst_id = pin.instId();
      const auto& inst    = netlist.inst(inst_id);
      std::cerr << "===========" << std::endl;
      std::cerr << "inst_id: " << inst_id << std::endl;
      std::cerr << "inst_arrivals: " << inst_arrivals[inst_id] << std::endl;
      std::cerr << "inst_requires: " << inst_requires[inst_id] << std::endl;
      std::cerr << "is_clock_terminal: " << is_clock_terminals[inst_id] << std::endl;
      if (IsTimingSource(inst_id)) {
        std::cerr << "Timing Source: " << inst_id << "(" + inst.attr().name() << ")" << pin_arrivals[pin_id] << " "
                  << pin_requires[pin_id] << std::endl;
      }
      if (IsTimingTerminal(inst_id)) {
        std::cerr << "Timing Terminal: " << inst_id << "(" + inst.attr().name() << ")" << pin_arrivals[pin_id] << " "
                  << pin_requires[pin_id] << std::endl;
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> StaticTimingAnalysisForward(database::PlaceDB const& placedb,
                                                                           at::Tensor               pin_delays,
                                                                           at::Tensor net_mask_ignore_large,
                                                                           double     timing_period) {
  auto       db                = placedb.db();
  auto&      design            = db->design();
  auto&      layout            = db->layout();
  auto       top_module_inst   = design.topModuleInst();
  auto&      netlist           = top_module_inst->netlist();
  int32_t    num_pins          = netlist.numPins();
  int32_t    num_nets          = netlist.numNets();
  at::Tensor pin_arrivals      = at::zeros({num_pins}, pin_delays.options());
  at::Tensor pin_requires      = at::zeros({num_pins}, pin_delays.options());
  at::Tensor ignored_net_masks = at::zeros({num_nets}, pin_delays.options()).to(torch::kBool);
  OPENPARF_DISPATCH_FLOATING_TYPES(pin_delays, "StaticTimingAnalysis", [&] {
    StaticTimingAnalysis<scalar_t>(placedb,
                                   OPENPARF_TENSOR_DATA_PTR(pin_delays, scalar_t),
                                   OPENPARF_TENSOR_DATA_PTR(pin_arrivals, scalar_t),
                                   OPENPARF_TENSOR_DATA_PTR(pin_requires, scalar_t),
                                   OPENPARF_TENSOR_DATA_PTR(net_mask_ignore_large, std::uint8_t),
                                   OPENPARF_TENSOR_DATA_PTR(ignored_net_masks, bool),
                                   timing_period);
  });
  return {pin_arrivals, pin_requires, ignored_net_masks};
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void StaticTimingAnalysis<T>(const database::PlaceDB& placedb,                                              \
                                        T*                       pin_delays,                                           \
                                        T*                       pin_arrivals,                                         \
                                        T*                       pin_requires,                                         \
                                        uint8_t*                 net_mask_ingore_large,                                \
                                        bool*                    ignore_net_masks,                                     \
                                        T                        timing_period);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace static_timing_analysis

OPENPARF_END_NAMESPACE
