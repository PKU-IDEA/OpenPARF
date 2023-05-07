/**
 * @file   netlist_wrapper.h
 * @author Yibai Meng
 * @date   Sep 2020
 * @brief  Wrapper class that emulates the netlist class needed for UTPlaceFX's ClockNetworkPlanner to run.
 */
#ifndef CNP_NETLIST_WRAPPER
#define CNP_NETLIST_WRAPPER

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include "util/namespace.h"
#include "util/util.h"

#include "database/clock_region.h"
#include "database/layout.h"
#include "database/placedb.h"
#include "database/site.h"

#include "ops/clock_network_planner/src/utplacefx/Types.h"

OPENPARF_BEGIN_NAMESPACE

namespace utplacefx {
    // This class interfaces database::PlaceDB to utplacefx::Netlist, so that we can
    // use utplacefx's code with minimal changes
    template<typename T>
    class WrapperNetlist {
    public:
        using ClockIdType = uint32_t;
        WrapperNetlist(database::PlaceDB const &p) : _p(p),
            _inst2ClockId(_p.instToClocks()),
            _net2ClockId(_p.getClockNetIndex()),
            _netId2ClockSourceInstId(_p.getNetClockSourceMapping())
    {
            _pos          = nullptr;
            _numClockNets = _p.numClockNets();
            _clockId2Net.resize(_numClockNets);
            // Generate clock id to net id mapping
            for (IndexType i = 0; i < _net2ClockId.size(); i++) {
                if (_net2ClockId[i] != InvalidIndex<ClockIdType>::value) _clockId2Net[_net2ClockId[i]] = i;
            }
            // Get the net id of the added floating net. It should not be counted in most circumstances
            // TODO: merge this fix to elsewhere
            floating_net_id = _p.nameToNet("OPENPARF_VDDVSS");
        }
        IndexType numNets() {
            return _p.numNets();
        }
        IndexType numClockNets() {
            return _numClockNets;
        }
        IndexType getClockSourceNodeIdFromNetId(ClockIdType ckNetid) {
            auto inst_id = _netId2ClockSourceInstId[ckNetid];
            return inst_id;
        }
        IndexType numNodes() {
            return _p.numInsts();
        }
        IndexType nodeNumPins(IndexType nodeId) {
            auto const &instPins = _p.instPins();
            auto const &pins     = instPins.at(nodeId);
            //TODO: optimize this
            IndexType cnt = 0;
            std::unordered_set<IndexType> in;
            for(auto pin : pins) {
                auto netId = _p.pin2Net(pin);
                if(in.find(netId) != in.end()) continue;
                in.insert(netId);
                if(netId == floating_net_id) continue;
                cnt++;
            }
            return cnt;
        }
        IndexType pinToNodeId(IndexType pinId) {
            return _p.pin2Inst(pinId);
        }

        IndexType netPin(IndexType netId, IndexType pinIdxInNet) {
            auto &n = _p.netPins();
            return n.at(netId, pinIdxInNet);
        }

        bool isNodeClockSource(IndexType nodeId) {
            return _p.isInstClockSource(nodeId);
        }

        std::vector<ClockIdType> clockIdxOfNode(IndexType nodeId) {
            return _inst2ClockId[nodeId];
        }
        XY<RealType> getXYFromNodeId(IndexType nodeId) {
            auto         x = _pos[2 * nodeId];
            auto         y = _pos[2 * nodeId + 1];
            XY<RealType> xy((RealType) x, (RealType) y);
            return std::move(xy);
        }
        std::string instName(IndexType nodeId) {
            return _p.instName(nodeId);
        }
        std::string netName(IndexType netId) {
            return _p.netName(netId);
        }

        RealType nodeRsrcDem(IndexType nodeId) {
            // FIXME(Jing Mai): A temporary hack. Each DSP and Node
            return 1.0;
        }
        RsrcType nodeRsrcType(IndexType nodeId) {
            // FIXME(Jing Mai): very hacky
            if (_p.isInstLUT(nodeId)) return RsrcType::LUTL;
            else if (_p.isInstFF(nodeId))
                return RsrcType::FF;
            auto areaType = _p.instAreaType(nodeId);
            openparfAssert(areaType.size() > 0);
            std::string name = _p.areaTypeName(areaType[0]);
            if (name.find("DSP") == 0) return RsrcType::DSP;
            if (name.find("RAM") == 0) return RsrcType::RAM;
            if (name.find("IO") == 0) return RsrcType::IO;
            openparfAssert(false);
            return RsrcType::INVALID;
        }
        bool isNodeFF(IndexType nodeId) {
            return _p.isInstFF(nodeId);
        }
        std::vector<OPENPARF_NAMESPACE::ResourceCategory> getNodeResourceCategory(IndexType nodeId) {
            return _p.instResourceCategory(nodeId);
        }
        RealType nodeArea(IndexType nodeId) {
            auto ats = _p.instAreaType(nodeId);
            openparfAssert(ats.size() == 1);   //TODO: more than one areatype?
             return _p.instSize(nodeId, ats[0]).area();
//            return _inflated_areas[nodeId];
        }
        IndexType netSize(IndexType netId) {
            auto const &netPins = _p.netPins().at(netId);
            return netPins.size();
        }
        IndexType clockNet(ClockIdType clockId) {
            openparfAssertMsg(clockId >= 0 and clockId < _numClockNets, "Invalid clock id %i", clockId);
            return _clockId2Net[clockId];
        }
        XY<IndexType> getFixedNodeSite(IndexType nodeId) {
            auto pt = _p.instLocs()[nodeId];
            // instLocs stores the center of instances. That means we need to round down to get the site x and y.
            // TODO: IO site size always one?
            return XY<IndexType>(IndexType(pt.x()), IndexType(pt.y()));
        }
        // TODO: a better way to pass the position of variables?
        void setPos(T const *pos) {
            _pos = pos;
        }

        void setInflatedAreas(T const *inflated_areas){
            _inflated_areas = inflated_areas;
        }

    private:
        IndexType                             _numClockNets;
        database::PlaceDB const &             _p;
        T const *                             _pos;
        std::vector<ClockIdType> const &              _net2ClockId;
        std::vector<IndexType>                _clockId2Net;
        std::vector<std::vector<ClockIdType>> const & _inst2ClockId;   // -1
        std::vector<IndexType> const &                _netId2ClockSourceInstId;
        IndexType floating_net_id;
        T const *                                     _inflated_areas;
    };

    template class WrapperNetlist<float>;
    template class WrapperNetlist<double>;

}   // namespace utplacefx

OPENPARF_END_NAMESPACE
#endif