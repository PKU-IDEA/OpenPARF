#ifndef __CLOCKNETWORKPLANNER_H__
#define __CLOCKNETWORKPLANNER_H__

#include <cmath>
#include <boost/container/flat_map.hpp>
#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>
#include <stdint.h>

#include "database/database.h"
#include "util/util.h"
#include "util/namespace.h"

#include "ops/clock_network_planner/src/netlist_wrapper.h"
#include "ops/clock_network_planner/src/database_wrapper.h"

#include "ops/clock_network_planner/src/utplacefx/Types.h"
#include "ops/clock_network_planner/src/utplacefx/Parameters.h"

#include "ops/clock_network_planner/src/utplacefx/SpiralAccessor.h"
#include "ops/clock_network_planner/src/utplacefx/Vector2D.h"

//#include "global/global.h"
//#include "db/Database.h"
//#include "db/Netlist.h"
//#include "util/XY.h"
//#include "util/Vector2D.h"
//#include "util/SpiralAccessor.h"

OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {
// Forward declaration
//class ClockNetworkPlannerVis;

/// Class for clock network planning
template <typename T>
class ClockNetworkPlanner {

public:
    struct ClockAvailCR;

private:
    // Forward declaration
    struct ClockRegionInfo;

    // Friend class
    friend class ClockRegionInfo;
    // friend class ClockNetworkPlannerVis;

private:
    using NodeIdGrid                   = Vector2D<std::vector<IndexType>>;
    using ClockMask                    = Vector2D<Byte>;
    using LemonGraph                   = lemon::ListDigraph;
    using LemonMinCostMaxFlowAlgorithm = lemon::NetworkSimplex<LemonGraph, FlowIntType>;

    /// Class to query and record clock region related information during node-to-clock region
    /// assignment
    struct ClockRegionInfo {
        explicit ClockRegionInfo() : crXY(INDEX_TYPE_MAX, INDEX_TYPE_MAX) {}

        ////////////////////////
        //                    //
        //  Member Functions  //
        //                    //
        ////////////////////////

        IndexType crX() const  { return crXY.x(); }
        IndexType crY() const  { return crXY.y(); }

        RealType rsrcDemOfType(RsrcType rt) const
        {
            switch (rt)
            {
                case RsrcType::LUTL   : return lutlDem;
                case RsrcType::LUTM   : return lutmDem;
                case RsrcType::FF     : return flDem;
                case RsrcType::SLICEL : return sllDem;
                case RsrcType::DSP    : return dspDem;
                case RsrcType::RAM    : return ramDem;
                default               : openparfAssert(false);
            }
            return 0.0;
        }

        RealType& rsrcDemOfType(RsrcType rt)
        {
            switch (rt)
            {
                case RsrcType::LUTL   : return lutlDem;
                case RsrcType::LUTM   : return lutmDem;
                case RsrcType::FF     : return flDem;
                case RsrcType::SLICEL : return sllDem;
                case RsrcType::DSP    : return dspDem;
                case RsrcType::RAM    : return ramDem;
                default               : openparfAssert(false);
            }
        }

        RealType rsrcCapOfType(RsrcType rt) const
        {
            switch (rt)
            {
                case RsrcType::LUTL   : return sllCap + slmCap - lutmDem;
                case RsrcType::LUTM   : return slmCap;
                case RsrcType::FF     : return sllCap + slmCap;
                case RsrcType::SLICEL : return sllCap + slmCap;
                case RsrcType::DSP    : return dspCap;
                case RsrcType::RAM    : return ramCap;
                default               : openparfAssert(false);
            }
            return 0.0;
        }

        ////////////////////
        //                //
        //  Data Members  //
        //                //
        ////////////////////

        // ID and Location of this clock region
        XY<IndexType>           crXY;

        // Capacity of each resource (site) type
        RealType                sllCap  = 0.0;
        RealType                slmCap  = 0.0;
        RealType                dspCap  = 0.0;
        RealType                ramCap  = 0.0;

        // Resource demand of each node type
        RealType                lutlDem = 0.0;
        RealType                lutmDem = 0.0;
        RealType                flDem   = 0.0;
        RealType                sllDem  = 0.0;
        RealType                dspDem  = 0.0;
        RealType                ramDem  = 0.0;
    };
    /*
    /// Class to represent a shape during node-to-clock region assignment
    struct ShapeInfo
    {
        // Information of each atomic node in this shape
        struct AtomInfo
        {
            explicit AtomInfo(IndexType id, IndexType idx, RealType ofs, RealType dem)
                : nodeId(id), yIdx(idx), yOffset(ofs), rsrcDem(dem) {}

            IndexType nodeId  = INDEX_TYPE_MAX; // The node ID
            IndexType yIdx    = INDEX_TYPE_MAX; // The site index of this node w.r.t. the bottom site occupied
            RealType  yOffset = 0.0;            // The yOffset w.r.t. the bottom node y coordinate
            RealType  rsrcDem = 0.0;            // The resource demand of this node
            IndexType tgtCrId = INDEX_TYPE_MAX; // The target clock region of this node
        };

        explicit ShapeInfo() = default;

        RealType                 x() const        { return xy.x(); }
        RealType                 yLo() const      { return xy.y(); }
        RealType                 yHi() const      { return xy.y() + atomInfoArray.back().yOffset; }
        IndexType                numSites() const { return atomInfoArray.back().yIdx + 1; }

        std::vector<AtomInfo>    atomInfoArray;                // The information of each atomic node
        std::vector<IndexType>   ckSig;                        // The sorted list of clock indices for this node set
        XY<RealType>             xy;                           // The bottom-left corner of this shape
        RsrcType                 rsrcType = RsrcType::INVALID; // The resource type of this shape
        RealType                 wt = 1.0;                     // The assignment weight of this shape
    };
    */
    /// Class to help the clustering process for node set construction
    struct NodeCluster
    {
        explicit NodeCluster(IndexType nodeId, IndexType ci, WrapperNetlist<T>* _nlPtr)
            :
              _nlPtr(_nlPtr),
              nodeIdArray(1, nodeId),
              crId(ci),
              ckSig(_nlPtr->clockIdxOfNode(nodeId))
        {
            auto node_xy = _nlPtr->getXYFromNodeId(nodeId);
            xy = node_xy;
            bbox.setXL(xy.x());
            bbox.setYL(xy.y());
            bbox.setXH(xy.x());
            bbox.setYH(xy.y());
            wt = _nlPtr->nodeNumPins(nodeId);
            auto r = _nlPtr->nodeRsrcType(nodeId);
            switch (r)
            {
                case RsrcType::DSP:
                case RsrcType::RAM:
                case RsrcType::IO:
                    rsrcType = r;
                    rsrcDem = _nlPtr->nodeRsrcDem(nodeId);
                    break;
                default:
                    // For non-DSP non-RAM nodes, we treat them all as SLICEL
                    // and use their areas as the resource demands
                    rsrcType = RsrcType::SLICEL;
                    rsrcDem = _nlPtr->nodeArea(nodeId);
            }
        }

        std::vector<IndexType>  nodeIdArray;
        std::vector<IndexType>  ckSig;
        Box<RealType>           bbox;
        XY<RealType>            xy;
        RsrcType                rsrcType  = RsrcType::INVALID;
        RealType                rsrcDem   = 0;
        RealType                wt        = 0.0;
        IndexType               parentIdx = INDEX_TYPE_MAX;  // The cluster ID that this cluster is merged to
        IndexType               crId      = INDEX_TYPE_MAX;  // The clock region ID of the node's location
        WrapperNetlist<T>* _nlPtr;
    };

    /// Class to represent the atomic node set during node-to-clock region assignment
    /// We assign a set of node instead of a single node is for runtime reason
    /// It should be noted that node sets should not be across clock region boundaries
    struct NodeSetInfo
    {
        /// Class for (clock region ID, resource demand, node IDs) tuple
        struct CrDem
        {
            explicit CrDem(IndexType i, RealType r) : crId(i), rsrcDemRatio(r) {}

            IndexType               crId         = INDEX_TYPE_MAX;
            RealType                rsrcDemRatio = 0;
            std::vector<IndexType>  nodeIdArray;
        };

        explicit NodeSetInfo() = default;
        /*
        explicit NodeSetInfo(IndexType nodeId)
            : nodeIdArray(1, nodeId),
              xy(_nlPtr->getXYFromNodeId(nodeId)),
              rsrcType(_nlPtr->nodeRsrcType(nodeId)),
              rsrcDem(_nlPtr->nodeRsrcDem(nodeId)),
              wt(_nlPtr->nodeNumPins(nodeId))
        {}*/
        explicit NodeSetInfo(const NodeCluster &cls)
            : nodeIdArray(cls.nodeIdArray),
              ckSig(cls.ckSig),
              xy(cls.xy),
              rsrcType(cls.rsrcType),
              rsrcDem(cls.rsrcDem),
              wt(cls.wt)
        {}

        RealType                x() const { return xy.x(); }
        RealType                y() const { return xy.y(); }

        std::vector<IndexType>  nodeIdArray;                       // All the nodes in this node set
        std::vector<IndexType>  ckSig;                             // The sorted list of clock indices for this node set
        XY<RealType>            xy;                                // The average locations of nodes in this node set
        RsrcType                rsrcType      = RsrcType::INVALID; // The resource type of this node set
        RealType                rsrcDem       = 0.0;               // The total resource demand of this node set
        RealType                wt            = 1.0;               // The assignment weight of this node set
        std::vector<CrDem>      tgtCrDemArray;                     // The set of target (clock region ID, resource demand) pair
        std::vector<IndexType>  crIdArray;                         // The array of clock region IDs sorted by distance from small to large
    };

    /// Class to record the masked (forbidden) clocks in each clock region
    struct ClockMaskSet
    {
        explicit ClockMaskSet() = default;
        ClockMaskSet(IndexType nck, IndexType xs, IndexType ys)
            : cmArray(nck, ClockMask(xs, ys, 0)), cmCostArray(nck, 0.0) {}

        bool operator<(const ClockMaskSet &rhs) const                                       { return cost < rhs.cost; }
        void mask(IndexType ckIdx, IndexType crId)                                          { cmArray.at(ckIdx).at(crId) = 1; }
        void mask(IndexType ckIdx, const XY<IndexType> &crXY)                               { cmArray.at(ckIdx).at(crXY) = 1; }
        bool isMasked(IndexType ckIdx, IndexType crId) const                                { return cmArray.at(ckIdx).at(crId); }
        bool isMasked(IndexType ckIdx, IndexType crX, IndexType crY) const                  { return cmArray.at(ckIdx).at(crX, crY); }
        bool isMasked(IndexType ckIdx, const XY<IndexType> &crXY) const                     { return cmArray.at(ckIdx).at(crXY); }
        bool isMasked(const std::vector<IndexType> &ckSig, const XY<IndexType> &crXY) const { return isMasked(ckSig, crXY.x(), crXY.y()); }
        bool isMasked(const std::vector<IndexType> &ckSig, IndexType crId) const
        {
            for (IndexType ckIdx : ckSig)
            {
                if (isMasked(ckIdx, crId))
                {
                    return true;
                }
            }
            return false;
        }
        bool isMasked(const std::vector<IndexType> &ckSig, IndexType crX, IndexType crY) const
        {
            for (IndexType ckIdx : ckSig)
            {
                if (isMasked(ckIdx, crX, crY))
                {
                    return true;
                }
            }
            return false;
        }

        std::vector<ClockMask>  cmArray;     // The clock mask of each clock
        std::vector<RealType>   cmCostArray; // The cost of each clock mask
        RealType                cost = 0.0;  // The sum of all costs in cmCostArray
    };

    /// Class to contains a pool of ClockMaskSets that have been explored
    /// This is a B-tree like implementation
    struct ClockMaskSetPool
    {
        // Each tree node represent a single clock mask
        struct TreeNode
        {
            explicit TreeNode() = default;
            TreeNode(const ClockMask &m) : cm(m) {}

            ClockMask               cm;            // The clock mask pattern
            std::vector<IndexType>  childIdxArray; // Childern tree node indices
        };

        // Add only root in 'treeNodeArray' and the root is a dummy node
        // that does not represent any clock mask pattern
        explicit ClockMaskSetPool() : treeNodeArray(1) {}

        // Add a ClockMaskSet in the pool, return true if success, return false if it exists in the pool already
        bool addClockMaskSet(const ClockMaskSet &cms)
        {
            IndexType curIdx = 0; // Current tree node index, initial it's root
            for (IndexType ckIdx = 0; ckIdx < cms.cmArray.size(); ++ckIdx)
            {
                TreeNode &cur = treeNodeArray.at(curIdx);
                const auto &cm = cms.cmArray.at(ckIdx);

                // Find the matching mask in cur's child
                bool exist = false;
                for (IndexType childIdx : cur.childIdxArray)
                {
                    const TreeNode &child = treeNodeArray.at(childIdx);
                    if (cm == child.cm)
                    {
                        // Found the matching clock mask
                        // Search the subtree rooted at 'child' for the next clock
                        curIdx = childIdx;
                        exist = true;
                        break;
                    }
                }

                if (! exist)
                {
                    // Cannot find the clock mask in 'cur''s childs for clock 'ckIdx'
                    // Need to insert a subtree rooted at 'cur'
                    IndexType parentIdx = curIdx;
                    for (IndexType idx = ckIdx; idx < cms.cmArray.size(); ++idx)
                    {
                        IndexType newIdx = treeNodeArray.size();
                        treeNodeArray.emplace_back(cms.cmArray.at(ckIdx));
                        treeNodeArray.at(parentIdx).childIdxArray.push_back(newIdx);
                        parentIdx = newIdx;
                    }
                    return true;
                }
            }

            // The pattern already exists in the pool
            return false;
        }

        // The 'root' is a dummy node that does not represent any clock mask
        // Here the tree depth is N + 1, where N is number of clocks
        // Each level (from level 1) in the tree represent a clock
        // Each tree node represent a certain clock mask pattern for the clock at that level
        // The 'treeNodeArray' contains all the tree node, and its first element is 'root'
        TreeNode               root;
        std::vector<TreeNode>  treeNodeArray;
    };

    /// Calss to record a shape assignment solution
    /*
    struct ShapeAssignSolution
    {
        explicit ShapeAssignSolution() = default;
        explicit ShapeAssignSolution(SiteColumnArray::SCIter sc, SiteColumn::YIter y, RealType d) : scIt(sc), yIt(y), dist(d) {}

        void set(SiteColumnArray::SCIter sc, SiteColumn::YIter y, RealType d)
        {
            scIt = sc;
            yIt = y;
            dist = d;
        }

        SiteColumnArray::SCIter    scIt;                 // The iterator points to the site column
        SiteColumn::YIter          yIt;                  // The iterator points to the bottom site Y coordinate
        RealType                   dist = REAL_TYPE_MAX; // The assignment movement
    };
    */
    /// Class for clock assignment results
    struct ClockAssignment
    {
        explicit ClockAssignment() = default;
        ClockAssignment(IndexType nck, IndexType xs, IndexType ys)
            : gridArray(nck, Vector2D<Byte>(xs, ys, 0))
        {}

        void reset()
        {
            for (auto &g : gridArray)
            {
                std::fill(g.begin(), g.end(), 0);
            }
        }

        void add(IndexType ckIdx, IndexType crId)                      { gridArray.at(ckIdx).at(crId) = 1; }
        void add(IndexType ckIdx, IndexType crX, IndexType crY)        { gridArray.at(ckIdx).at(crX, crY) = 1; }
        bool has(IndexType ckIdx, IndexType crId) const                { return gridArray.at(ckIdx).at(crId); }
        bool has(IndexType ckIdx, IndexType crX, IndexType crY) const  { return gridArray.at(ckIdx).at(crX, crY); }

        std::vector<Vector2D<Byte>>  gridArray;
    };

    /// Class for node to clock region assignment results
    struct NodeAssignResult
    {
        explicit NodeAssignResult(IndexType nck, IndexType xs, IndexType ys)
            : clockAssign(nck, xs, ys) {}

        bool                        legal = false;   // Found a legal solution or not
        RealType                    cost  = 0;       // The cost (e.g. required cell movement) of this solution
        ClockAssignment             clockAssign;     // The solution of the clock assignment
    };

    /// Class for branch objects in DRoute
    /// Branches are horizontal routes
    struct DRouteBranch
    {
        DRouteBranch(IndexType yy, IndexType xl, IndexType xh)
            : y(yy), xLo(xl), xHi(xh)
        {}

        IndexType len() const { return xHi - xLo + 1; }
        bool operator==(const DRouteBranch &rhs) const { return y == rhs.y && xLo == rhs.xLo && xHi == rhs.xHi; }

        IndexType y;
        IndexType xLo;
        IndexType xHi;
    };

    /// Class for distribution layer route objects for a clock tree
    /// Each route is a trunk tree with a vertical trunk
    struct DRoute
    {
        void reset()
        {
            topoCost = 0;
            penalty = 0;
            tkX = INDEX_TYPE_MAX;
            tkYLo = INDEX_TYPE_MAX;
            tkYHi = 0;
            branches.clear();
        }

        bool operator==(const DRoute &rhs) const
        {
            return tkX == rhs.tkX && tkYLo == rhs.tkYLo && tkYHi == rhs.tkYHi && branches == rhs.branches;
        }

        bool occupyVD(const XY<IndexType> &xy) const
        {
            return xy.x() == tkX && xy.y() >= tkYLo && xy.y() <= tkYHi;
        }

        bool occupyHD(const XY<IndexType> &xy) const
        {
            for (const auto br : branches)
            {
                if (xy.y() == br.y && xy.x() >= br.xLo && xy.x() <= br.xHi)
                {
                    return true;
                }
            }
            return false;
        }

        RealType cost() const { return topoCost + penalty; }

        RealType                    topoCost     = 0;              // Cost of this route topology
        RealType                    penalty      = 0;              // Penalty cost of this route for LR
        RealType                    deltaPenalty = 0;              // Delta penalty cost, temporary variable for penalty updating
        IndexType                   tkX          = INDEX_TYPE_MAX; // X index of the trunk
        IndexType                   tkYLo        = INDEX_TYPE_MAX; // YLo index of the trunk
        IndexType                   tkYHi        = 0;              // YHi index of the trunk
        std::vector<DRouteBranch>   branches;                      // Branches in this trunk tree, they are sorted by their y from low to high
    };

    /// Class to track distribuition layer routing
    struct DRouteTracker
    {
        /// Get the DRoute of the given index
        const DRoute & dRoute(IndexType idx) const                        { return candArray.at(idx); }
        DRoute &       dRoute(IndexType idx)                              { return candArray.at(idx); }
        const DRoute & selDRoute(IndexType ckIdx) const                   { return candArray.at(selIdxArray.at(ckIdx)); }
        DRoute &       selDRoute(IndexType ckIdx)                         { return candArray.at(selIdxArray.at(ckIdx)); }
        IndexType      selDRouteIdx(IndexType ckIdx) const                { return selIdxArray.at(ckIdx); }
        void           setSelDRouteIdx(IndexType ckIdx, IndexType drIdx)  { selIdxArray.at(ckIdx) = drIdx; }

        /// Reset delta penalty of all DRoute
        void resetDRouteDeltaPenalty()
        {
            for (DRoute &dr : candArray)
            {
                dr.deltaPenalty = 0;
            }
        }

        /// Reset delta penalty of all DRoute
        void applyAndResetDRouteDeltaPenalty(RealType scale)
        {
            for (DRoute &dr : candArray)
            {
                // Penalty must be non-negative
                dr.deltaPenalty *= scale;
                dr.penalty = std::max(dr.penalty + dr.deltaPenalty, (RealType) 0.0);
                dr.deltaPenalty = 0;
            }
        }

        /// Get the DRoute with the lowest cost for a given clock index
        IndexType bestDRouteIdx(IndexType ckIdx) const
        {
            IndexType retIdx = INDEX_TYPE_MAX;
            RealType minCost = REAL_TYPE_MAX;
            for (IndexType idx : ckIdxToDRouteIdxArray.at(ckIdx))
            {
                RealType cost = dRoute(idx).cost();
                if (cost < minCost)
                {
                    minCost = cost;
                    retIdx = idx;
                }
            }
            return retIdx;
        }

        /// Add a new DRoute candidate to clock 'ckIdx', return false if the candidate is already in the candidate pool
        bool addDRoute(IndexType ckIdx, const DRoute &dr)
        {
            // Check duplication
            for (IndexType curIdx : ckIdxToDRouteIdxArray.at(ckIdx))
            {
                if (candArray.at(curIdx) == dr)
                {
                    return false;
                }
            }

            // This is not a duplicate DRoute, add it into candArray
            IndexType drIdx = candArray.size();
            candArray.emplace_back(dr);
            ckIdxToDRouteIdxArray.at(ckIdx).push_back(drIdx);
            for (IndexType y = dr.tkYLo; y <= dr.tkYHi; ++y)
            {
                crToVertDRouteIdxArray.at(dr.tkX, y).push_back(drIdx);
            }
            for (const auto &br : dr.branches)
            {
                for (IndexType x = br.xLo; x <= br.xHi; ++x)
                {
                    crToHoriDRouteIdxArray.at(x, br.y).push_back(drIdx);
                }
            }
            return true;
        }

        std::vector<DRoute>                    candArray;             // All DRoute candidates
        std::vector<IndexType>                 selIdxArray;           // The indices of DRoute candidates currently being selected
        std::vector<std::vector<IndexType>>    ckIdxToDRouteIdxArray;
        Vector2D<std::vector<IndexType>>       crToHoriDRouteIdxArray;
        Vector2D<std::vector<IndexType>>       crToVertDRouteIdxArray;
    };

    /// Clock demand grid for DRoute
    struct DRouteGrid
    {
        struct Dem
        {
            IndexType vert = 0; // Vertical D-layer demand
            IndexType hori = 0; // Horizontal D-layer demand
        };

        DRouteGrid(IndexType w, IndexType h) : demGrid(w, h) {}

        IntType                 vdDemCapDiff(const XY<IndexType> &xy) const  { return (IntType)demGrid.at(xy).vert - (IntType)Parameters::archClockRegionClockCapacity; }
        IntType                 hdDemCapDiff(const XY<IndexType> &xy) const  { return (IntType)demGrid.at(xy).hori - (IntType)Parameters::archClockRegionClockCapacity; }
        IndexType               vdOverflow(const XY<IndexType> &xy) const    { return std::max(0, (IntType)demGrid.at(xy).vert - (IntType)Parameters::archClockRegionClockCapacity); }
        IndexType               vdOverflow(IndexType x, IndexType y) const   { return std::max(0, (IntType)demGrid.at(x, y).vert - (IntType)Parameters::archClockRegionClockCapacity); }
        IndexType               hdOverflow(const XY<IndexType> &xy) const    { return std::max(0, (IntType)demGrid.at(xy).hori - (IntType)Parameters::archClockRegionClockCapacity); }
        IndexType               hdOverflow(IndexType x, IndexType y) const   { return std::max(0, (IntType)demGrid.at(x, y).hori - (IntType)Parameters::archClockRegionClockCapacity); }

        IndexType vdOverflow() const
        {
            IntType res = 0;
            for(IndexType x = 0; x < demGrid.xSize(); x++) {
                for(IndexType y = 0; y < demGrid.ySize(); y++) {
                    XY<IndexType> xy (x, y);
                    res += vdOverflow(xy);
                }
            }
            /*
            for (auto xy : BoxIterator<IndexType>(0, 0, demGrid.xSize() - 1, demGrid.ySize() - 1))
            {
                res += vdOverflow(xy);
            }*/
            return res;
        }

        IndexType hdOverflow() const
        {
            IntType res = 0;
            for(IndexType x = 0; x < demGrid.xSize(); x++) {
                for(IndexType y = 0; y < demGrid.ySize(); y++) {
                    XY<IndexType> xy (x, y);
                    res += hdOverflow(xy);
                }
            }
            /*
            for (auto xy : BoxIterator<IndexType>(0, 0, demGrid.xSize() - 1, demGrid.ySize() - 1))
            {
                res += hdOverflow(xy);
            }*/
            return res;
        }

        /// Update the demand grid when add the given DRoute
        void addDRouteDemand(const DRoute &dr)
        {
            // Update VD
            for (IndexType y = dr.tkYLo; y <= dr.tkYHi; ++y)
            {
                demGrid.at(dr.tkX, y).vert += 1;
            }
            // Update hori
            for (const DRouteBranch &br : dr.branches)
            {
                for (IndexType x = br.xLo; x <= br.xHi; ++x)
                {
                    demGrid.at(x, br.y).hori += 1;
                }
            }
        }

        /// Update the demand grid when remove the given DRoute
        void removeDRouteDemand(const DRoute &dr)
        {
            // Update VD
            for (IndexType y = dr.tkYLo; y <= dr.tkYHi; ++y)
            {
                demGrid.at(dr.tkX, y).vert -= 1;
            }
            // Update HD
            for (const DRouteBranch &br : dr.branches)
            {
                for (IndexType x = br.xLo; x <= br.xHi; ++x)
                {
                    demGrid.at(x, br.y).hori -= 1;
                }
            }
        }

        Vector2D<Dem>  demGrid;
    };

    /// Enum type for routing orientation
    enum class Orient
    {
        H,  // Horizontal
        V   // Vertical
    };

    /// Clock source information used for R-layer routing
    struct ClockSourceInfo
    {
        XY<IndexType>  crXY; // The belonging clock region XY
    };

    /// Class for edges in routing layer routing
    struct RRouteEdge
    {
        RRouteEdge(const XY<IndexType> &xyi, Orient o)
            : xy(xyi), orient(o) {}

        IndexType x() const { return xy.x(); }
        IndexType y() const { return xy.y(); }

        XY<IndexType>  xy;
        Orient         orient;
    };

    /// Class for routing layer route objects for a clock
    /// Each route is a 2-pin net
    struct RRoute
    {
        std::vector<RRouteEdge> edgeArray; // Edges from source to sink
    };

    /// Class for RRoute grid
    struct RRouteGrid
    {
        struct Dem
        {
            IndexType    dem(Orient orient) const { return orient == Orient::H ? hori : vert; }
            IndexType &  dem(Orient orient)       { return orient == Orient::H ? hori : vert; }

            IndexType hori = 0;
            IndexType vert = 0;
        };

        struct Cost
        {
            RealType    cost(Orient orient) const { return orient == Orient::H ? hori : vert; }
            RealType &  cost(Orient orient)       { return orient == Orient::H ? hori : vert; }

            RealType hori = 1.0;
            RealType vert = 1.0;
        };

        struct PathCost
        {
            RealType    cost(Orient orient) const { return orient == Orient::H ? hori : vert; }
            RealType &  cost(Orient orient)       { return orient == Orient::H ? hori : vert; }

            RealType hori = REAL_TYPE_MAX;
            RealType vert = REAL_TYPE_MAX;
        };

        RRouteGrid(IndexType w, IndexType h)
            : demGrid(w, h), costGrid(w, h), pathCostGrid(w, h)
        {}

        void         resetPathCosts()                                       { std::fill(pathCostGrid.begin(), pathCostGrid.end(), PathCost()); }
        IndexType    dem(const XY<IndexType> &xy, Orient orient) const      { return orient == Orient::H ? demGrid.at(xy).hori : demGrid.at(xy).vert; }
        IndexType &  dem(const XY<IndexType> &xy, Orient orient)            { return orient == Orient::H ? demGrid.at(xy).hori : demGrid.at(xy).vert; }
        RealType     cost(const XY<IndexType> &xy, Orient orient) const     { return orient == Orient::H ? costGrid.at(xy).hori : costGrid.at(xy).vert; }
        RealType &   cost(const XY<IndexType> &xy, Orient orient)           { return orient == Orient::H ? costGrid.at(xy).hori : costGrid.at(xy).vert; }
        RealType     pathCost(const XY<IndexType> &xy, Orient orient) const { return orient == Orient::H ? pathCostGrid.at(xy).hori : pathCostGrid.at(xy).vert; }
        RealType &   pathCost(const XY<IndexType> &xy, Orient orient)       { return orient == Orient::H ? pathCostGrid.at(xy).hori : pathCostGrid.at(xy).vert; }

        void addRRouteDemand(const RRoute &rr)
        {
            for (const auto &e : rr.edgeArray)
            {
                ++dem(e.xy, e.orient);
            }
        }

        void removeRRouteDemand(const RRoute &rr)
        {
            for (const auto &e : rr.edgeArray)
            {
                --dem(e.xy, e.orient);
            }
        }

        Vector2D<Dem>       demGrid;       // Record routing demand at each clock region
        Vector2D<Cost>      costGrid;      // Record routing cost at each clock region
        Vector2D<PathCost>  pathCostGrid;  // Record current best path cost in each grid
    };

    /// Class for frontiers in R-layer routing
    struct RRouteFrontier
    {
        RRouteFrontier(const XY<IndexType> &xyi, Orient o, RealType dc)
            : xy(xyi), orient(o), detCost(dc) {}

        RRouteFrontier(IndexType xi, IndexType yi, Orient o, RealType dc)
            : xy(xi, yi), orient(o), detCost(dc) {}

        IndexType              x() const  { return xy.x(); }
        IndexType              y() const  { return xy.y(); }

        XY<IndexType>          xy;
        Orient                 orient;
        RealType               detCost = 0.0; // Determined cost
        RealType               estCost = 0.0; // Estimated cost to finish
    };

    /// RRouteFrontier comparator for min heap
    struct RRouteFrontierComp
    {
        bool operator()(const RRouteFrontier &l, const RRouteFrontier &r) const
        {
            return (l.detCost + l.estCost) > (r.detCost + r.estCost);
        }
    };

    /// Enum type for clock network planning solution
    enum class SolutionType
    {
        NODE_ASSIGN_FAIL,
        DROUTE_FAIL,
        RROUTE_FAIL,
        BBOX_FAIL,
        LEGAL
    };

    /// A clock network planning solution
    /// This solution may or may not be legal
    struct Solution {
        bool operator<(const Solution &rhs) const { return cost < rhs.cost; }

        SolutionType type = SolutionType::NODE_ASSIGN_FAIL;   // The type (status) of this solution
        RealType     cost = REAL_TYPE_MAX;   // Cost so far for this solution, e.g., the minimum
                                             // movement required
        ClockMaskSet        cms;             // The clock mask set used for this solution
        std::vector<DRoute> dRouteArray;     // DRoute solution of each clock
        std::vector<RRoute> rRouteArray;     // RRoute solution of each clock
        ClockAvailCR        clkAvailCR;      ///< Bounding box based routing estimation
        Vector2D<IndexType> crCkCountMap;    ///< Clock region demand map
    };

public:
    /// Class for clock availability information in clock regions
    /// An object of this class tells which clock is available in each clock region
    class ClockAvailCR
    {
    public:
        const std::vector<Vector2D<Byte>> &  ckIdxToAvailGrid() const                                     { return _ckIdxToAvailGrid; }
        std::vector<Vector2D<Byte>> &        ckIdxToAvailGrid()                                           { return _ckIdxToAvailGrid; }
        bool                                 isAvail(IndexType ckIdx, IndexType crX, IndexType crY) const { return _ckIdxToAvailGrid.at(ckIdx).at(crX, crY); }
        bool                                 isAvail(IndexType ckIdx, IndexType crId) const               { return _ckIdxToAvailGrid.at(ckIdx).at(crId); }

    private:
        // The array size is the number of clock nets
        // Each Vector2D at _ckIdxToAvailGrid.at(ckIdx) stores the available clock regions for clock 'ckIdx'
        std::vector<Vector2D<Byte>>  _ckIdxToAvailGrid;
    };

    /// Class for clock availability information in half-column regions
    /// An object of this class tells which clock is available in each half-column regions
    class ClockAvailHC
    {
    public:
        const std::vector<std::vector<Byte>> &  ckIdxToAvailArray() const                                    { return _ckIdxToAvailArray; }
        std::vector<std::vector<Byte>> &        ckIdxToAvailArray()                                          { return _ckIdxToAvailArray; }
        bool                                    isAvail(IndexType ckIdx, IndexType hcId) const               { return _ckIdxToAvailArray.at(ckIdx).at(hcId); }

    private:
        // The array size is the number of clock nets
        // Each vector at _ckIdxToAvailArray.at(ckIdx) stores the available half-column regions for clock 'ckIdx'
        std::vector<std::vector<Byte>>  _ckIdxToAvailArray;
    };

    /// Class to store the arc information in the min-cost flow
    struct ArcInfo
    {
        explicit ArcInfo(IndexType l, IndexType r) : lIdx(l), rIdx(r) {}

        IndexType lIdx = INDEX_TYPE_MAX;
        IndexType rIdx = INDEX_TYPE_MAX;
    };

public:
    explicit ClockNetworkPlanner(WrapperDatabase *db)
        : _db(db),
          _spiralAccessor(std::max(Parameters::cnpNetCoarseningMaxClusterDimX, Parameters::cnpNetCoarseningMaxClusterDimY) + 1)
    {
        initClockRegionInfos();
    }
    void setNetlist(WrapperNetlist<T> *nl)
    {
        _nlPtr = nl;
        //initShapeInfos();
        initClockSourceInfos();
    }

    bool                            run();
    void                            updateNodeToClockRegionAssignment();
    void                            planHalfColumnRegionClockAvailability();
    const std::vector<IndexType> &  nodeIdArrayOfCr(IndexType crId) const { return _crToNodeIdArray.at(crId); }
    const ClockAvailCR &            clockAvailCR() const                  { return _clockAvailCR; }
    const ClockAvailHC &            clockAvailHC() const                  { return _clockAvailHC; }
    void                            exportClockTreeToFile(const std::string &fileName) const;
    void                            transferSolutionToTorchTensor(int32_t* nodeToCr, uint8_t* clkAvailCR);

    bool nodeToCrIsClockLegal(IndexType nodeId, IndexType crId) const
    {
        auto ckIdxs = _nlPtr->clockIdxOfNode(nodeId); // TODO: Only one clock supported!
        //for (IndexType ckIdx : node.clockSignature())
        //{
        for(auto const idx : ckIdxs) {
            if (! _clockAvailCR.isAvail(idx, crId))
            {
                return false;
            }
        }
        //}
        return true;
    }

private:
    bool                            run(const ClockMaskSet &initCMS);
    void                            initClockSourceInfos();
    void                            initClockRegionInfos();
    void                            resetClockRegionResourceDemands();
    //void                            initShapeInfos();
    //void                            updateShapeLocations();
    void                            createNodeSetInfos();
    void                            netCoarsening(IndexType netId, std::vector<NodeCluster> &clsArray) const;
    void                            physicalClustering(std::vector<NodeCluster> &clsArray) const;
    bool                            mergeNodeClustersIfLegal(IndexType idxA, IndexType idxB, std::vector<NodeCluster> &clsArray) const;
    void                            findSolution(const ClockMaskSet &cms, Solution &sol);
    NodeAssignResult                runNodeAssignment(const ClockMaskSet &cms);
    void                            shapeAssignmentKernel(const ClockMaskSet &cms, NodeAssignResult &res);
    /*
    ShapeAssignSolution             getShapeAssignment(const ClockMaskSet &cms, const ShapeInfo &shapeInfo) const;
    SiteColumn::YIter               getShapeToSiteColumnAssignment(const ClockMaskSet &cms, const ShapeInfo &shapeInfo, const SiteColumn &sc) const;
    bool                            shapeToSiteIsLegal(const ClockMaskSet &cms, const ShapeInfo &shapeInfo, const SiteColumn &sc, const SiteColumn::YIter yIt) const;
    void                            applyShapeAssignment(const ShapeAssignSolution &sol, ShapeInfo &shapeInfo, NodeAssignResult &res);
    */
    void                            nodeSetAssignmentRGTKernel(const ClockMaskSet &cms, NodeAssignResult &res);
    void                            nodeSetAssignmentMCFKernel(const ClockMaskSet &cms, NodeAssignResult &res);
    RealType     getXYToClockRegionDist(const XY<RealType> &xy, const ClockRegionInfo &crInfo,
                                        RsrcType rt) const;
    XY<RealType> getDistXYToCrSiteOfType(const XY<RealType> &xy, const XY<IndexType> &crXY,
                                         SiteType st) const;
    bool   runDLayerRouting(const ClockAssignment &clockAssign, std::vector<DRoute> &dRouteArray);
    bool   runBBoxRouting(const ClockAssignment &clockAssign, ClockAvailCR &clkAvailCR);
    void   buildDRoutesAndInitDRouteTracker(const ClockAssignment &clockAssign);
    DRoute buildDRoute(const ClockAssignment &clockAssign, IndexType ckIdx, IndexType tkX) const;
    bool   selectDRoutesLR(std::vector<DRoute> &dRouteArray);
    bool   runRLayerRouting(const std::vector<DRoute> &dRouteArray,
                            std::vector<RRoute> &      rRouteArray) const;
    void   orderRLayerNetRouting(const std::vector<DRoute> &dRouteArray,
                                 std::vector<IndexType> &   order) const;
    void   routeRLayerNet(IndexType ckIdx, const DRoute &dr, RRouteGrid &rrg, RRoute &rr) const;
    void   setRRouteEstCost(const DRoute &dr, RRouteFrontier &rf) const;
    void   updateRRouteCostGrid(const RRoute &rr, RRouteGrid &rrg) const;
    void   collectRipupClockNets(const RRouteGrid &rrg, const std::vector<RRoute> &rRouteArray,
                                 std::vector<IndexType> &ripupCkIdxArray) const;
    void   createNewClockMaskSets(const Solution &sol, RealType maxCost, ClockMaskSetPool &cmsPool,
                                  std::vector<ClockMaskSet> &newCMSArray) const;
    void   createNewClockMaskSetsToResolveVDOverflow(const Solution &sol, const XY<IndexType> &crXY,
                                                     RealType maxCost, ClockMaskSetPool &cmsPool,
                                                     std::vector<ClockMaskSet> &newCMSArray) const;
    void   createNewClockMaskSetsToResolveHDOverflow(const Solution &sol, const XY<IndexType> &crXY,
                                                     RealType maxCost, ClockMaskSetPool &cmsPool,
                                                     std::vector<ClockMaskSet> &newCMSArray) const;
    void createNewClockMaskSetsToResolveBBoxOverflow(const Solution &sol, const XY<IndexType> &crXY,
                                                     RealType maxCost, ClockMaskSetPool &cmsPool,
                                                     std::vector<ClockMaskSet> &newCMSArray) const;
    RealType getClockMaskCost(IndexType ckIdx, const ClockMaskSet &cms) const;
    void     commitSolution(const Solution &sol);
    bool     commitNodeAssignment(const ClockMaskSet &cms);
    void     realizeNodeAssignment(NodeSetInfo &nsInfo);

    RealType maxClusterRsrcDemOfType(RsrcType rt) const;
    RealType nodeToHcProbability(IndexType nodeId, IndexType hcid) const;
    RealType gaussPhi(RealType x1, RealType x2, RealType mu, RealType sigma) const;

private:
    WrapperDatabase *  _db;
    WrapperNetlist<T> *_nlPtr;

    SpiralAccessor            _spiralAccessor;
    Vector2D<ClockRegionInfo> _crInfoGrid;
    // boost::container::flat_map<RsrcType, std::vector<ShapeInfo>> _rsrcTypeToShapeInfoArray;
    boost::container::flat_map<RsrcType, std::vector<NodeSetInfo>> _rsrcTypeToNodeSetInfoArray;
    boost::container::flat_map<RsrcType, std::vector<std::vector<IndexType>>>
            _rsrcTypeToCkIdxToShapeIdxArray;
    boost::container::flat_map<RsrcType, std::vector<std::vector<IndexType>>>
            _rsrcTypeToCkIdxToNodeSetIdxArray;

    // For D-layer routing
    DRouteTracker _drtk;

    // For R-layer routing
    std::vector<ClockSourceInfo> _ckSrcInfoArray;

    // The commited legal solution information
    ClockMaskSet                     _clockMaskSet;      // The clock mask set
    ClockAvailCR                     _clockAvailCR;      // The clock availability information
    ClockAvailHC                     _clockAvailHC;      // The clock availability information
    std::vector<DRoute>              _dRouteArray;       // The D-layer routing solution
    std::vector<RRoute>              _rRouteArray;       // The R-layer routing solution
    Vector2D<std::vector<IndexType>> _crToNodeIdArray;   // The IDs of nodes in each clock region
    Vector2D<IndexType>              _crCkCountMap;      ///< Clock region demand map
};

/// Get the distance between a given XY to a clock region
template<typename T>
inline RealType ClockNetworkPlanner<T>::getXYToClockRegionDist(const XY<RealType> &xy, const ClockRegionInfo &crInfo, RsrcType rt) const
{
    XY<RealType> distXY;
    switch (rt)
    {
        case RsrcType::DSP:
            distXY = getDistXYToCrSiteOfType(xy, crInfo.crXY, SiteType::DSP);

        case RsrcType::RAM:
            distXY = getDistXYToCrSiteOfType(xy, crInfo.crXY, SiteType::RAM);

        default: {
            auto b = _db->cr(crInfo.crXY).bbox();
            RealType dx = 0, dy = 0;
            if(xy.x() > b.xh()) dx = xy.x() - b.xh();
            else if(xy.x() < b.xl()) dx = b.xl() - xy.x();
            if(xy.y() > b.yh()) dy = xy.y() - b.yh();
            else if(xy.y() < b.yl()) dx = b.yl() - xy.y();
            distXY.setX(dx);
            distXY.setY(dy);
            //distXY = _db->cr(crInfo.crXY).bbox().manhDistXY(xy);

        }
    }

    if (distXY.x() == REAL_TYPE_MAX)
    {
        return REAL_TYPE_MAX;
    }
    RealType xCost = Parameters::scaledXLen(distXY.x());
    RealType yCost = Parameters::scaledXLen(distXY.y());
    return xCost + yCost;
}

/// Get the maximum resource demand for a given resource type
template <typename T>
inline RealType ClockNetworkPlanner<T>::maxClusterRsrcDemOfType(RsrcType rt) const
{
    switch (rt)
    {
        case RsrcType::SLICEL : return Parameters::cnpNetCoarseningMaxClusterDemandSLICE;
        case RsrcType::DSP    : return Parameters::cnpNetCoarseningMaxClusterDemandDSP;
        case RsrcType::RAM    : return Parameters::cnpNetCoarseningMaxClusterDemandRAM;
        default               : openparfAssert(false);
    }
    return 0.0;
}

/// Get the probability that a given node is in the given half-column region
template <typename T>
inline RealType ClockNetworkPlanner<T>::nodeToHcProbability(IndexType nodeId, IndexType hcId) const
{
    auto b =_db->getHalfColumnRegionBbox(hcId);
    auto xy = _nlPtr->getXYFromNodeId(nodeId);
    // If this node is in the half-column region, return 1.0
    //if(xy.x() >= b.xl() and xy.x() <= b.xh() and xy.y() >= b.yl() and xy.y() <= b.yh())
    if (b.contain(xy))
    {
        return 1.0;
    }

    // Check clock region constraint
    if (! nodeToCrIsClockLegal(nodeId, _db->siteCrId(xy)))
    {
        return 0.0;
    }

    RealType probX = gaussPhi(b.xl() - 0.5, b.xh() + 0.5, xy.x(), Parameters::cnpHalfColumnPlanningGaussianSigma);
    RealType probY = gaussPhi(b.yl() - 0.5, b.yh() + 0.5, xy.y(), Parameters::cnpHalfColumnPlanningGaussianSigma);
    return probX * probY;
}

/// Return the phi(x1, x2) of a Gaussian distribution N ~ (mu, sigma^2)
/// phi is the gaussCDF(x2) - gaussCDF(x1)
template <typename T>
inline RealType ClockNetworkPlanner<T>::gaussPhi(RealType x1, RealType x2, RealType mu, RealType sigma) const
{
    constexpr RealType SQRT2 = 1.4142135623730951;
    RealType deno = SQRT2 * sigma;
    return 0.5 * (std::erf((x2 - mu) / deno) - std::erf((x1 - mu) / deno));
}

}

OPENPARF_END_NAMESPACE

#endif