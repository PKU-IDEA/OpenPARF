#include <stack>
#include <utility>
#include <boost/heap/fibonacci_heap.hpp>

#include "ops/clock_network_planner/src/utplacefx/ClockNetworkPlanner.h"
#include "ops/clock_network_planner/src/utplacefx/GeneralAssignmentProblem.h"
#include "ops/clock_network_planner/src/utplacefx/RegretMinimizationSolver.h"
#include "ops/clock_network_planner/src/utplacefx/Types.h"

//#include "cnp/ClockNetworkPlanner.h"
//#include "util/BoxIterator.h"
//#include "util/Interval.h"
//#include "gas/GeneralAssignmentProblem.h"
//#include "gas/RegretMinimizationSolver.h"
//#include "vis/ClockNetworkPlannerVis.h"

OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {

/// Initialize clock source informations
template<typename T>
void ClockNetworkPlanner<T>::initClockSourceInfos()
{
    _ckSrcInfoArray.clear();
    _ckSrcInfoArray.resize(_nlPtr->numClockNets());

    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {
        auto &info = _ckSrcInfoArray.at(ckIdx);

        // Yibai Meng: this code here is trying to find the clock region where the clock source of clk i is at
        // To be done in init of wrapper nl
        // Get the clock source crXY
        //const Net &ckNet = _nlPtr->clockNet(ckIdx);
        //const Pin &srcPin = _nlPtr->pin(ckNet.clockSourcePinId());
        //const Node &srcNode = _nlPtr->nodeOfPinId(srcPin.id());
        auto clkSrcNetId = _nlPtr->clockNet(ckIdx);
        auto srcNodeId = _nlPtr->getClockSourceNodeIdFromNetId(clkSrcNetId);
        auto srcNode = _nlPtr->getFixedNodeSite(srcNodeId);
        info.crXY = XY<IndexType>(_db->siteCrId(srcNode) / _db->numCrY(), _db->siteCrId(srcNode) % _db->numCrY()); // MOD1
    }
}

/// Initialize clock region informations
template<typename T>
void ClockNetworkPlanner<T>::initClockRegionInfos() {
    _crInfoGrid.clear();
    _crInfoGrid.resize(_db->numCrX(), _db->numCrY());
    for (IndexType crI = 0; crI < _db->numCrX(); crI++)
        for (IndexType crJ = 0; crJ < _db->numCrY(); crJ++) {
            XY<IndexType> crXY(crI, crJ);
            auto &        crInfo = _crInfoGrid.at(crXY);
            const auto &  cr     = _db->cr(crXY);

            crInfo.crXY = crXY;
            // MOD
            crInfo.sllCap = cr.numSites(_db->siteTypeToId(SiteType::SLICEL));
            // TODO: SLICEM are currently not considered in OpenPARF
            crInfo.slmCap = 0;   // cr.numSites(_db->siteTypeToId(SiteType::SLICEM));
            crInfo.dspCap = cr.numSites(_db->siteTypeToId(SiteType::DSP));
            crInfo.ramCap = cr.numSites(_db->siteTypeToId(SiteType::RAM));
            // openparfprint(kdebug, "# of slicels in cr(%d, %d): %f\n", cri, crj, crinfo.sllcap);
            // openparfPrint(kDebug, "# of DSPs in CR(%d, %d): %f\n", crI, crJ, crInfo.dspCap);
            // openparfPrint(kDebug, "# of RAMs in CR(%d, %d): %f\n", crI, crJ, crInfo.ramCap);
        }
}

/// Reset clock region resource demands
template<typename T>
void ClockNetworkPlanner<T>::resetClockRegionResourceDemands()
{
    for (auto &crInfo : _crInfoGrid)
    {
        crInfo.lutlDem = 0.0;
        crInfo.lutmDem = 0.0;
        crInfo.flDem = 0.0;
        crInfo.sllDem = 0.0;
        crInfo.dspDem = 0.0;
        crInfo.ramDem = 0.0;
    }
}

inline RealType siteHeightOfType(SiteType st)
{
    switch (st)
    {
        case SiteType::SLICEL: case SiteType::SLICEM:  return 1.0;
        case SiteType::DSP:                            return 2.5;
        case SiteType::RAM:                            return 5.0;
        default:                                       openparfAssert(false);
    }
    return 0.0;
}

/*
/// Initialize shape informations
template<typename T>
void ClockNetworkPlanner<T>::initShapeInfos()
{
    // Create a ShapeInfo object for each shape
    _rsrcTypeToShapeInfoArray.clear();
    for (ShapeType st : ALL_SHAPE_TYPES)
    {
        const auto &shapeIdArray = _nlPtr->shapeIdArrayOfType(st);
        if (shapeIdArray.empty())
        {
            continue;
        }

        RsrcType rt = shapeTypeToRsrcType(st);
        auto &shapeInfoArray = _rsrcTypeToShapeInfoArray[rt];
        RealType unitH = siteHeightOfType(rsrcTypeToSiteType(rt));

        for (IndexType shapeId : shapeIdArray)
        {
            const auto &shape = _nlPtr->shape(shapeId);
            shapeInfoArray.emplace_back();
            auto &info = shapeInfoArray.back();
            info.wt = 0.0;
            for (const auto &e : shape.elementArray())
            {
                const Node &node = _nlPtr->node(e.nodeId);
                info.atomInfoArray.emplace_back(e.nodeId, e.y(), e.y() * unitH, node.rsrcDem());
                info.ckSig.insert(info.ckSig.end(), node.clockSignature().begin(), node.clockSignature().end());
                info.rsrcType = rt;
                info.wt += node.numPins();
            }

            // Finalize clock signature
            std::sort(info.ckSig.begin(), info.ckSig.end());
            info.ckSig.erase(std::unique(info.ckSig.begin(), info.ckSig.end()), info.ckSig.end());
        }
    }

    // Fill the _rsrcTypeToCkIdxToShapeIdxArray
    _rsrcTypeToCkIdxToShapeIdxArray.clear();
    for (const auto &p : _rsrcTypeToShapeInfoArray)
    {
        const auto &shapeInfoArray = p.second;
        auto &ckIdxToShapeIdxArray = _rsrcTypeToCkIdxToShapeIdxArray[p.first];
        ckIdxToShapeIdxArray.resize(_nlPtr->numClockNets());
        for (IndexType shapeIdx = 0; shapeIdx < shapeInfoArray.size(); ++shapeIdx)
        {
            const auto &shapeInfo = shapeInfoArray.at(shapeIdx);
            for (IndexType ckIdx : shapeInfo.ckSig)
            {
                ckIdxToShapeIdxArray.at(ckIdx).push_back(shapeIdx);
            }
        }
    }
}
*/

/*
/// Update shape locations in ShpaeInfo objects
template<typename T>
void ClockNetworkPlanner<T>::updateShapeLocations()
{
    for (auto &p : _rsrcTypeToShapeInfoArray)
    {
        auto &shapeInfoArray = p.second;
        for (auto &shapeInfo : shapeInfoArray)
        {
            RealType x = 0.0, y = 0.0;
            for (const auto &atomInfo : shapeInfo.atomInfoArray)
            {
                const Node &node = _nlPtr->node(atomInfo.nodeId);
                x += node.x();
                y += node.y() - atomInfo.yOffset;
            }
            shapeInfo.xy.setXY(std::max(x, 0.0), std::max(y, 0.0));
            shapeInfo.xy /= shapeInfo.atomInfoArray.size();
        }
    }
}
*/
/// Create all NodeSetInfo objects based on current placement
/// Each node set here is a set of cells that are strongly connected and have the following property
///   a) The bounding box of all the nodes in a node set should be bounded (small enough)
///   b) The bounding box of all the nodes in a node set should not be accross clock region boundaries
///   b) For any pair of nodes (i, j) in the node set, one of the clock signature must be empty or exactly equal to the other one
///   c) All nodes in a node set must be 1) all DSPs, or 2) all RAMs, or 3) all non-DSP non-RAM cells (e.g., LUTL, LUTM, FF, CLBL, CLBM, CARRY, ...)
///   d) For DSPs and RAMs, node set demands are their resource demands, and clock region capacities are their site counts
///   e) For non-DSP non-RAM cases, node set demands are their total area, and clock region capacities are SLICEL + SLICELM site counts
///   f) The demand of each node site should be bounded, and this bound should be small enough w.r.t. a clock region capacity
///
/// We create node sets using a simple and greedy clustering algorithm as follows:
///   1) We assume the _nlPtr->netArray() are sorted by their pin counts from low to high
///   2) We iterate through each net and try to merge the belonging clusters of all nodes in this net together.
///      The merging will be rejected if any of the property above is violated
///   3) We stop the merging process if all nets are checked or the remaining nets are too huge (have too many pins)
///   4) Cluster nodes that are physcially close
///   5) We create a node set for each cluster
template<typename T>
void ClockNetworkPlanner<T>::createNodeSetInfos()
{

    auto countNumberOfClusters = [](std::vector<ClockNetworkPlanner::NodeCluster>& clsArray) {
    IndexType cnt = 0;
    for(IndexType i = 0 ; i < clsArray.size(); i++) {
        const auto &cls = clsArray.at(i);
        if (cls.parentIdx == INDEX_TYPE_MAX) cnt++;
    }
    return cnt;
    };

    // Create a cluster each node
    std::vector<NodeCluster> clsArray;
    clsArray.reserve(_nlPtr->numNodes());
    for(IndexType i = 0; i < _nlPtr->numNodes(); i++) {
        auto xy = _nlPtr->getXYFromNodeId(i);
        auto cr_id = _db->siteCrId(xy);
        clsArray.emplace_back(i, cr_id, _nlPtr);
    }
    /*
    for (const auto &node : _nlPtr->nodeArray())
    {
        clsArray.emplace_back(node, _db->siteCrId(node.xy()));
    }
    */
    // Iteratively absort nets from small to large
    // The _nlPtr->netArray() should be sorted by pin count from low to high
    // TODO: should we order the nets first?
    for(IndexType netSize = 1; netSize <= Parameters::cnpNetCoarseningMaxNetDegree; netSize++) {
        for (IndexType i = 0; i < _nlPtr->numNets(); i++) {
            if(_nlPtr->netSize(i) == netSize) {
                netCoarsening(i, clsArray);
            }
        }
    }

    // Further reduce number of clusters by physical clustering
    physicalClustering(clsArray);

    // Create a node set for each cluster
    IndexType numClusters = 0;
    _rsrcTypeToNodeSetInfoArray.clear();
    for (const auto &cls : clsArray)
    {
        if (cls.parentIdx != INDEX_TYPE_MAX || cls.rsrcType == RsrcType::IO)
        {
            // This is a sub-cluster or an IO node, skip it
            continue;
        }
        ++numClusters;
        _rsrcTypeToNodeSetInfoArray[cls.rsrcType].emplace_back(cls);

        // Sort all clock regions for this node set by distance
        auto &nsInfo = _rsrcTypeToNodeSetInfoArray[cls.rsrcType].back();
        nsInfo.crIdArray.resize(_crInfoGrid.size());
        std::iota(nsInfo.crIdArray.begin(), nsInfo.crIdArray.end(), 0);
        std::sort(nsInfo.crIdArray.begin(), nsInfo.crIdArray.end(),
            [&](IndexType l, IndexType r) {

                return _db->cr(l).bbox().manhDist(nsInfo.xy) < _db->cr(r).bbox().manhDist(nsInfo.xy);
        });
    }
    openparfPrint(MessageType::kDebug, "#clusters = %u\n", numClusters);

    // Fill the _rsrcTypeToCkIdxToNodeSetIdxArray
    _rsrcTypeToCkIdxToNodeSetIdxArray.clear();
    for (const auto &p : _rsrcTypeToNodeSetInfoArray)
    {
        const auto &nsInfoArray = p.second;
        auto &ckIdxToNodeSetIdxArray = _rsrcTypeToCkIdxToNodeSetIdxArray[p.first];
        ckIdxToNodeSetIdxArray.resize(_nlPtr->numClockNets());
        for (IndexType nsIdx = 0; nsIdx < nsInfoArray.size(); ++nsIdx)
        {
            const auto &nsInfo = nsInfoArray.at(nsIdx);
            for (IndexType ckIdx : nsInfo.ckSig)
            {
                ckIdxToNodeSetIdxArray.at(ckIdx).push_back(nsIdx);
            }
        }
    }
}

/// Perfrom net coarsening
/// The net coarsening is based on the "Modified Hyperedge Coarsening" algorithm
template<typename T>
void ClockNetworkPlanner<T>::netCoarsening(IndexType netId, std::vector<NodeCluster> &clsArray) const
{
    // Collect all different non-IO node clusters in this net
    std::vector<IndexType> clsIdxArray;
    auto numPins = _nlPtr->netSize(netId);
    clsIdxArray.reserve(numPins);
    for (IndexType pinIdxInNet = 0; pinIdxInNet < numPins; pinIdxInNet++)
    {
        IndexType pinId = _nlPtr->netPin(netId, pinIdxInNet);
        // Get the cluster that currently contains this node
        //IndexType clsIdx = _nlPtr->pin(pinId).nodeId();
        IndexType clsIdx = _nlPtr->pinToNodeId(pinId);
        const auto &cls = clsArray.at(clsIdx);
        if (cls.parentIdx != INDEX_TYPE_MAX)
        {
            clsIdx = cls.parentIdx;
        }
        if (cls.rsrcType != RsrcType::IO && std::find(clsIdxArray.begin(), clsIdxArray.end(), clsIdx) == clsIdxArray.end())
        {
            clsIdxArray.push_back(clsIdx);
        }
    }

    // Perfrom "Modified Hyperedge Coarsening"
    while (clsIdxArray.size() > 1)
    {
        // Merge all nodes to the first cluster in clsIdxArray
        for (IndexType i = 1; i < clsIdxArray.size(); ++i)
        {
            if (mergeNodeClustersIfLegal(clsIdxArray.at(0), clsIdxArray.at(i), clsArray))
            {
                // The merging is successful, invalidate the clsIdxArray.at(i)
                clsIdxArray.at(i) = INDEX_TYPE_MAX;
            }
        }
        // Invalidate the clsIdxArray.at(0) and remove all invalid cluster indices in clsIdxArray
        clsIdxArray.at(0) = INDEX_TYPE_MAX;
        clsIdxArray.erase(std::remove(clsIdxArray.begin(), clsIdxArray.end(), INDEX_TYPE_MAX), clsIdxArray.end());
    }
}

/// Perfrom physical clustering
/// Merge clusters that are physically close
template<typename T>
void ClockNetworkPlanner<T>::physicalClustering(std::vector<NodeCluster> &clsArray) const
{
    // Collect all clusters and put them into a grid
    std::vector<IndexType> clsIdxArray;
    Vector2D<std::vector<IndexType>> grid(_db->numSiteX(), _db->numSiteY());
    for (IndexType clsIdx = 0; clsIdx < _nlPtr->numNodes(); ++clsIdx)
    {
        const auto &cls = clsArray.at(clsIdx);
        if (cls.rsrcType != RsrcType::IO && cls.parentIdx == INDEX_TYPE_MAX)
        {
            clsIdxArray.push_back(clsIdx);
            grid.at(cls.xy.x(), cls.xy.y()).push_back(clsIdx);
        }
    }

    // For each cluster, spirally access its physical neighbors and try to merge with them
    // We start from clusters with smallest clock count
    std::sort(clsIdxArray.begin(), clsIdxArray.end(), [&](IndexType l, IndexType r){ return clsArray.at(l).ckSig.size() < clsArray.at(r).ckSig.size(); });
    for (IndexType clsIdx : clsIdxArray)
    {
        const auto &cls = clsArray.at(clsIdx);
        if (cls.parentIdx != INDEX_TYPE_MAX)
        {
            // This cluster has been merged and becomes a sub-cluster
            continue;
        }

        IndexType maxD = std::ceil(Parameters::cnpNetCoarseningMaxClusterDimX - std::min(cls.xy.x() - cls.bbox.xl(), cls.bbox.xh() - cls.xy.x())) +
                         std::ceil(Parameters::cnpNetCoarseningMaxClusterDimY - std::min(cls.xy.y() - cls.bbox.yl(), cls.bbox.yh() - cls.xy.y()));
        auto beg = _spiralAccessor.begin(0);
        auto end = _spiralAccessor.begin(maxD);
        XY<IntType> clsXY(IntType(cls.xy.x()), IntType(cls.xy.y()));
        Box<IntType> bndBox(_db->cr(cls.crId).bbox());
        for (auto it = beg; it != end; ++it)
        {
            XY<IntType> xy(clsXY + *it);
            if (! bndBox.contain(xy))
            {
                // We only merge clusters in the same clock region
                continue;
            }
            for (IndexType &i : grid.at(xy))
            {
                if (i != INDEX_TYPE_MAX && (i == clsIdx || mergeNodeClustersIfLegal(clsIdx, i, clsArray)))
                {
                    i = INDEX_TYPE_MAX;
                }
            }
        }
    }
}

/// Given two different node clusters, merge them if legal
/// Return true if the merging is legal. Otherwise, return false
template<typename T>
bool ClockNetworkPlanner<T>::mergeNodeClustersIfLegal(IndexType idxA, IndexType idxB, std::vector<NodeCluster> &clsArray) const
{
    NodeCluster &a = clsArray.at(idxA);
    NodeCluster &b = clsArray.at(idxB);

    // Check resource type compatibility
    // Check if the two clusters are in different clock regions
    if (a.rsrcType != b.rsrcType || a.crId != b.crId)
    {
        return false;
    }

    // Check if the merged resource demand is too large
    RealType rsrcDem = a.rsrcDem + b.rsrcDem;
    if (rsrcDem > maxClusterRsrcDemOfType(a.rsrcType))
    {
        return false;
    }

    // Check if the merged bounding box is too large
    auto bbox(a.bbox);
    bbox.join(b.bbox);
    if (bbox.width() > Parameters::cnpNetCoarseningMaxClusterDimX || bbox.height() > Parameters::cnpNetCoarseningMaxClusterDimY)
    {
        return false;
    }

    // Check clock signature compatibility
    // At least one of the following must be hold, otherwise the merging is illegal
    //   1) One of the two clock signatures is empty
    //   2) The two clock signatures are exactly the same
    if (! a.ckSig.empty() && ! b.ckSig.empty() && a.ckSig != b.ckSig)
    {
        return false;
    }

    // If the execution hits here, the merging is legal
    // Merge b to a

    // Put all the nodes in b into a
    // Set the parents of nodes in b to a
    a.nodeIdArray.insert(a.nodeIdArray.end(), b.nodeIdArray.begin(), b.nodeIdArray.end());
    for (IndexType idx : b.nodeIdArray)
    {
        clsArray.at(idx).parentIdx = idxA;
    }

    // Update the clock signature if needed
    if (a.ckSig.empty())
    {
        a.ckSig = b.ckSig;
    }

    // Update the bounding box, location, resource demand, and weight
    a.bbox = bbox;
    //
    auto new_x = (a.xy.x() * a.wt + b.xy.x() * b.wt) / (a.wt + b.wt);
    auto new_y = (a.xy.y() * a.wt + b.xy.y() * b.wt) / (a.wt + b.wt);
    a.xy.setX(new_x);
    a.xy.setY(new_y);
    a.rsrcDem = rsrcDem;
    a.wt += b.wt;
    return true;
}

/// The top function to run the clock network planning start with a empty clock mask
/// Return if a legal solution is found
template<typename T>
bool ClockNetworkPlanner<T>::run()
{
    // Create an empty clock mask
    ClockMaskSet initCMS(_nlPtr->numClockNets(), _db->numCrX(), _db->numCrY());
    return run(initCMS);
}

/// The top function to run the clock network planning start with the given clock mask set
/// Return if a legal solution is found
template<typename T>
bool ClockNetworkPlanner<T>::run(const ClockMaskSet &initCMS)
{
    // Initialization
    //updateShapeLocations();
    createNodeSetInfos();
    openparfPrint(MessageType::kDebug, "Node sets created.\n");
    // Clock mask set stack, we explore the solution tree in a DFS manner
    // Push the initial mask set into the stack as the starting point
    std::stack<ClockMaskSet> cmsStack;
    cmsStack.emplace(initCMS);

    // ClockMaskSetPool stores all clock mask sets have been or will be considered
    // This is used for pruning duplicate clock masks
    ClockMaskSetPool cmsPool;
    cmsPool.addClockMaskSet(initCMS);

    // Record the best legal solution found and the number of legal solutions found
    Solution bestSol;
    IndexType numLegalSol = 0;
    openparfPrint(MessageType::kDebug, "Start exploring the ClockMaskSet tree in the DFS order.\n");

    // Explore the ClockMaskSet tree in the DFS order
    while (!cmsStack.empty()) {
        // Get the stack top
        ClockMaskSet topCMS = cmsStack.top();
        //        {
        //            openparfPrint(kDebug, "************* topCMS ****************");
        //           for(int32_t ck_id=0; ck_id < topCMS.cmArray.size(); ck_id++){
        //               openparfPrint(kDebug, "ck_id: %d\n", ck_id);
        //               auto &cm = topCMS.cmArray[ck_id];
        //               for(int i = 0; i < cm.xSize(); i++){
        //                   std::stringstream ss;
        //                   for(int j = 0; j < cm.ySize(); j++){
        //                      ss << int(cm.at(i,j)) << " ";
        //                   }
        //                   openparfPrint(kDebug, "%s\n", ss.str().c_str());
        //               }
        //           }
        //        }
        cmsStack.pop();

        // Generate the solution corresponding to the top clock mask
        Solution topSol;
        findSolution(topCMS, topSol);

        constexpr const char *const SolutionTypeName[] = {"NODE_ASSIGN_FAIL", "DROUTE_FAIL",
                                                          "RROUTE_FAIL", "BBOX_FAIL", "LEGAL"};
        openparfPrint(MessageType::kDebug, "Sol node assignment cost %.2lf. Type %s\n", topSol.cost,
                      SolutionTypeName[(uint32_t) topSol.type]);

        if (topSol.type == SolutionType::LEGAL) {
            if (bestSol.type != SolutionType::LEGAL || topSol.cost < bestSol.cost) {
                bestSol = topSol;
            }

            openparfPrint(MessageType::kDebug,
                          "Legal Sol Cost = %.2lf, CMS Cost = %.2lf, Best Sol Cost = %.2lf\n",
                          topSol.cost, topCMS.cost, bestSol.cost);
            if (++numLegalSol == Parameters::cnpMaxNumLegalSolution) {
                // Found enough number of legal solutions
                // Stop exploring and commit the best solution found so far
                commitSolution(bestSol);
                return true;
            }
        } else if (topSol.type == SolutionType::BBOX_FAIL) {
            // The top solution has D-layer overflow
            // Continue to explore this solution only if
            //   1) no legal solution has been found yet, OR
            //   2) the cost of the new solution is still less than that in the best solution
            if (bestSol.type == SolutionType::LEGAL && topSol.cost >= bestSol.cost) {
                // This top solution will only generate sub-optimal solution
                continue;
            }
            // Generate new clock mask sets
            // New mask sets are sorted by their exploration priority from high to low
            // Since clock mask set cost is a (not tight) lower bound solution cost,
            // we can safely discard clock masks with costs larger than bestSol.cost
            std::vector<ClockMaskSet> newCMSArray;
            createNewClockMaskSets(topSol, bestSol.cost, cmsPool, newCMSArray);

            // Push new masks into the stack
            // Note that we need to push the masks in the reverse order
            for (auto rit = newCMSArray.rbegin(); rit != newCMSArray.rend(); ++rit) {
                cmsStack.emplace(*rit);
            }
            // openparfPrint(kDebug, "%i new clock masks generated after DROUTE_FAIL\n",
            // newCMSArray.size());
        } else {
            openparfAssert(false);
            // if topSol.type == SolutionType::CELL_ASSIGN_FAIL || SolutionType::RROUTE_FAIL,
            // we simply discard it
        }
    }

    if (bestSol.type == SolutionType::LEGAL)
    {
        commitSolution(bestSol);
        return true;
    }
    return false;
}

/// Update node to clock region assignment without changing clock network solution
template<typename T>
void ClockNetworkPlanner<T>::updateNodeToClockRegionAssignment()
{
    //updateShapeLocations();
    //createNodeSetInfos();

    // We try the legal clock mask set found previously
    // If this function fails, no changed will be commited
    //commitNodeAssignment(_clockMaskSet);

    // We start with the legal clock mask set found previously
    // If this function fails, no changed will be commited
    run();
}

/// Find a solution for a given clock mask set
/// \param  cms     the given clock mask set
/// \param  output  the output solution
template<typename T>
void ClockNetworkPlanner<T>::findSolution(const ClockMaskSet &cms, Solution &sol)
{
    // Find the corresponding movement-minimized node assignment solution
    // and its corresponding clock assignment solution
    // Use the cost of the node assignment as the cost of current solution
    sol.cms = cms;
    NodeAssignResult nodeAssignRes = runNodeAssignment(sol.cms);
    sol.cost                       = nodeAssignRes.cost;

    if (!nodeAssignRes.legal) {
        // No legal node assignment can be found using the given ClockMaskSet
        sol.type = SolutionType::NODE_ASSIGN_FAIL;
        openparfPrint(kWarn,
                      "No legal node assignment can be found using the given ClockMaskSet\n");
        return;
    }

    // Perform D-Layer routing using the clock assignment got from the node assignment result
    //    if (!runDLayerRouting(nodeAssignRes.clockAssign, sol.dRouteArray)) {
    //        // No legal D-layer routing solution is found
    //        sol.type = SolutionType::DROUTE_FAIL;
    //        return;
    //    }

    // Perform R-Layer routing using the D-layer routing solution
    //    if (!runRLayerRouting(sol.dRouteArray, sol.rRouteArray)) {
    //        // No legal R-layer routing solution is found
    //        sol.type = SolutionType::RROUTE_FAIL;
    //        return;
    //    }

    runBBoxRouting(nodeAssignRes.clockAssign, sol.clkAvailCR);

    int32_t num_cr_x    = _db->numCrX();
    int32_t num_cr_y    = _db->numCrY();
    auto &  cr_ck_count = sol.crCkCountMap;
    cr_ck_count.clear();
    cr_ck_count.resize(num_cr_x, num_cr_y, 0);
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx) {
        auto &availGrid = sol.clkAvailCR.ckIdxToAvailGrid().at(ckIdx);
        for (int i = 0; i < num_cr_x; i++) {
            for (int j = 0; j < num_cr_y; j++) {
                if (availGrid.at(i, j) == 1) { cr_ck_count.at(i, j)++; }
            }
        }
    }
    openparfPrint(kDebug, "**** CR-CK Count:\n");
    for (int i = 0; i < num_cr_x; i++) {
        std::stringstream ss;
        for (int j = 0; j < num_cr_y; j++) { ss << cr_ck_count.at(i, j) << " "; }
        openparfPrint(kDebug, "%s\n", ss.str().c_str());
    }

    for (int i = 0; i < num_cr_x; i++) {
        for (int j = 0; j < num_cr_y; j++) {
            if (cr_ck_count.at(i, j) > Parameters::archClockRegionClockCapacity) {
                sol.type = SolutionType::BBOX_FAIL;
                return;
            }
        }
    }


    // A legal solution is found
    sol.type = SolutionType::LEGAL;
}

/// Run node-to-clock region assignment with the constraint of the given clock mask set
/// \param  cms  the given ClockMaskSet that tells with clock is masked in with clock region
template<typename T>
typename ClockNetworkPlanner<T>::NodeAssignResult ClockNetworkPlanner<T>::runNodeAssignment(const ClockNetworkPlanner<T>::ClockMaskSet &cms)
{
    // Reset all clock region resource demands
    resetClockRegionResourceDemands();

    NodeAssignResult res(_nlPtr->numClockNets(), _db->numCrX(), _db->numCrY());
    res.legal = true;

    // Assign shapes
    /*
    shapeAssignmentKernel(cms, res);
    if (! res.legal)
    {
        // The shape assignment fails, no need to proceed
        return res;
    }
    */
    // Assign node sets
    //nodeSetAssignmentRGTKernel(cms, res);
    nodeSetAssignmentMCFKernel(cms, res);
    return res;
}

/// The kernel function to perform Shape to clock region assignment
/// This function is based on the regret-minimization heuristic for general min-cost assignment problem
/// \param  cms  the given ClockMaskSet that tells with clock is masked in with clock region
/// \param  res  results of the node assignment
/*
template<typename T>
void ClockNetworkPlanner<T>::shapeAssignmentKernel(const ClockMaskSet &cms, NodeAssignResult &res)
{
    // Assignment shape one by one
    for (auto &p : _rsrcTypeToShapeInfoArray)
    {
        for (auto &shapeInfo : p.second)
        {
            ShapeAssignSolution shapeAssignSol = getShapeAssignment(cms, shapeInfo);
            if (shapeAssignSol.dist == REAL_TYPE_MAX)
            {
                // The shape cannot find a legal position
                res.legal = false;
                return;
            }

            // Apply the shape assignment
            applyShapeAssignment(shapeAssignSol, shapeInfo, res);
        }
    }
}
*/
/// Assign a shape to clock regions given the clock mask constraint
/// Here we do not consider overlapping and only find the closest possible sites to assign
/// \param  cms            the given ClockMaskSet that tells with clock is masked in with clock region
/// \param  shapeInfo      the shape to be assigned
/// \param  nodeAssignRes  results of the node assignment
/// \return                return the shape assignment solution
/*
template<typename T>
ClockNetworkPlanner<T>::ShapeAssignSolution ClockNetworkPlanner<T>::getShapeAssignment(const ClockMaskSet &cms, const ShapeInfo &shapeInfo) const
{
    // Get the assignment set type of this shape
    // Although SLICEL shape can be assigned to both SLICEL and SLICELM sites, here
    // we only consider SLICEL sites for them
    SiteType siteType = rsrcTypeToSiteType(shapeInfo.rsrcType);

    // Find the closest site column in X direction
    auto &scArray = _db->siteColumnArrayOfType(siteType).siteColumnArray();
    auto scIt = std::lower_bound(scArray.begin(), scArray.end(), shapeInfo.x(), [&](const SiteColumn &sc, RealType x){ return sc.x() < x; });
    if (scIt == scArray.end())
    {
        --scIt;
    }
    else if (scIt != scArray.begin())
    {
        scIt = (std::abs(scIt->x() - shapeInfo.x()) < std::abs(std::prev(scIt)->x() - shapeInfo.x()) ? scIt : std::prev(scIt));
    }

    // Find the best assignment in range [scIt, scArray.end())
    ShapeAssignSolution rSol;
    for (auto it = scIt; it != scArray.end(); ++it)
    {
        RealType xDist = Parameters::scaledXLen(std::abs(it->x() - shapeInfo.x()));
        if (xDist >= rSol.dist)
        {
            // This site column and all site columns right to this site column are sub-optimal
            // No need to proceed
            break;
        }
        auto yIt = getShapeToSiteColumnAssignment(cms, shapeInfo, *it);
        RealType dist = xDist + Parameters::scaledYLen(std::abs(*yIt - shapeInfo.yLo()));
        if (dist < rSol.dist)
        {
            rSol.set(it, yIt, dist);
        }
    }

    // Find the best assignment in range [scArray.begin(), scIt)
    ShapeAssignSolution lSol;

    // Note thet a reverse_iterator points to the element before the element that its corresponding forward iterator points to
    std::reverse_iterator<SiteColumnArray::SCIter> rit(scIt);
    std::reverse_iterator<SiteColumnArray::SCIter> rend(scArray.begin());
    for (; rit != rend; ++rit)
    {
        RealType xDist = Parameters::scaledXLen(std::abs(rit->x() - shapeInfo.x()));
        if (xDist >= lSol.dist || xDist >= rSol.dist)
        {
            // This site column and all site columns left to this site column are sub-optimal
            // No need to proceed
            break;
        }
        auto yIt = getShapeToSiteColumnAssignment(cms, shapeInfo, *rit);
        RealType dist = xDist + Parameters::scaledYLen(std::abs(*yIt - shapeInfo.yLo()));
        if (dist < lSol.dist)
        {
            // The base() of a reverse_iterator has an offset of 1
            lSol.set(std::prev(rit.base()), yIt, dist);
        }
    }

    // Pick the better solution between lSol and rSol
    return (lSol.dist < rSol.dist ? lSol : rSol);
}
*/
/// Given a Shape and a SiteColumn, get the closest assignment
/*
template<typename T>
SiteColumn::YIter ClockNetworkPlanner<T>::getShapeToSiteColumnAssignment(const ClockMaskSet &cms,
                                                                      const ShapeInfo &shapeInfo,
                                                                      const SiteColumn &sc) const
{
    // Check if the column has enough capacity
    if (sc.numSites() < shapeInfo.numSites())
    {
        return sc.end();
    }

    // Get the nearest Y coordinate in the column site
    auto yIt = std::lower_bound(sc.begin(), sc.end(), shapeInfo.yLo());
    if (yIt == sc.end())
    {
        --yIt;
    }
    else if (yIt != sc.begin())
    {
        yIt = (std::abs(*yIt - shapeInfo.yLo()) < std::abs(*std::prev(yIt) - shapeInfo.yLo()) ? yIt : std::prev(yIt));
    }

    // Check if placing the shape in its closest site is legal
    if (shapeToSiteIsLegal(cms, shapeInfo, sc, yIt))
    {
        return yIt;
    }

    // If the code hit here, placing the shape in its closest site is not legal
    // We need to shift the shape downward/upward to find a legal solution
    // Note that, given the closest site is clock illegal, an optimal and clock-legal solution
    // must be achieved at one of the following two cases
    //   1) the shape is placed above the best location, and its bottom node is placed at the bottom of a clock region
    //   2) the shape is placed below the best location, and its top node is placed at the top of a clock region
    // So, we just need to find the best sites among all solutions in these two cases

    // Get the best solution in case 1), that is, the solution that is above the best location
    RealType upperDistY = REAL_TYPE_MAX;
    SiteColumn::YIter upperYIt = sc.end();
    IndexType shapeCrYLo = _db->yToCrY(shapeInfo.yLo());
    for (IndexType crY = shapeCrYLo + 1; crY < _db->numCrY(); ++crY)
    {
        auto &seg = sc.segmentOfCrY(crY);
        if (seg.numSites() == 0)
        {
            continue;
        }
        if (shapeToSiteIsLegal(cms, shapeInfo, sc, seg.begin()))
        {
            upperDistY = std::abs(shapeInfo.yLo() - *seg.begin());
            upperYIt = seg.begin();
            break;
        }
    }

    // Get the best solution in case 2), that is, the solution that is below the best location
    RealType lowerDistY = REAL_TYPE_MAX;
    SiteColumn::YIter lowerYIt = sc.end();
    IndexType shapeCrYHi = _db->yToCrY(std::min(shapeInfo.yHi(), _db->numSiteY() - 1.0));
    for (IntType crY = shapeCrYHi - 1; crY >= 0; --crY)
    {
        auto &seg = sc.segmentOfCrY(crY);
        if (seg.end() - sc.begin() < shapeInfo.numSites())
        {
            continue;
        }
        auto it = seg.end() - shapeInfo.numSites();
        RealType distY = std::abs(shapeInfo.yLo() - *it);
        if (distY > upperDistY)
        {
            // This and all sites below are sub-optimal
            // No need to proceed
            break;
        }
        if (shapeToSiteIsLegal(cms, shapeInfo, sc, it))
        {
            lowerDistY = distY;
            lowerYIt = it;
            break;
        }
    }

    // Compare the lower and upper solution and return the better one
    return (lowerDistY < upperDistY ? lowerYIt : upperYIt);
}
*/
/// Check if placing a shape at the given site (site column 'sc' with Y coordinate of *yIt) is legal
/*
template<typename T>
bool ClockNetworkPlanner<T>::shapeToSiteIsLegal(const ClockMaskSet &cms, const ShapeInfo &shapeInfo, const SiteColumn &sc, SiteColumn::YIter yIt) const
{
    // Check if the shape will exceed the site column top boundary
    if (sc.end() - yIt < shapeInfo.numSites())
    {
        return false;
    }

    IndexType crYLo = _db->yToCrY(*yIt);
    IndexType crYHi = _db->yToCrY(*(yIt + shapeInfo.numSites() - 1));
    for (IndexType crY = crYLo; crY <= crYHi; ++crY)
    {
        // Check the clock legality
        if (cms.isMasked(shapeInfo.ckSig, sc.crX(), crY))
        {
            return false;
        }

        // Check the capacity constraint
        const auto &seg = sc.segmentOfCrY(crY);
        RealType rsrcDem = std::min(yIt + shapeInfo.numSites(), seg.end()) - std::max(yIt, seg.begin());
        if (crY == crYHi)
        {
            // The last site can be partially occupied, so here we need specail handling
            auto rbeg = shapeInfo.atomInfoArray.rbegin();
            auto rend = shapeInfo.atomInfoArray.rend();
            IndexType topYIdx = rbeg->yIdx;
            RealType topSiteRsrcDem = rbeg->rsrcDem;
            while (++rbeg != rend && rbeg->yIdx == topYIdx)
            {
                topSiteRsrcDem += rbeg->rsrcDem;
            }
            rsrcDem -= (1.0 - topSiteRsrcDem);
        }
        const auto &crInfo = _crInfoGrid.at(sc.crX(), crY);
        if (crInfo.rsrcDemOfType(shapeInfo.rsrcType) + rsrcDem > crInfo.rsrcCapOfType(shapeInfo.rsrcType))
        {
            return false;
        }
    }
    return true;
}
*/
/// Apply the shape assignment
/*
template<typename T>
void ClockNetworkPlanner<T>::applyShapeAssignment(const ShapeAssignSolution &sol, ShapeInfo &shapeInfo, NodeAssignResult &res)
{
    // Update the clock assignment
    IndexType crYLo = _db->yToCrY(*(sol.yIt));
    IndexType crYHi = _db->yToCrY(*(sol.yIt + shapeInfo.numSites() - 1));
    for (IndexType crY = crYLo; crY <= crYHi; ++crY)
    {
        for (IndexType ckIdx : shapeInfo.ckSig)
        {
            res.clockAssign.add(ckIdx, sol.scIt->crX(), crY);
        }
    }

    // Update the resource demand and set target clock region for each node in the shape
    for (auto &atom : shapeInfo.atomInfoArray)
    {
        IndexType crY = _db->yToCrY(*(sol.yIt + atom.yIdx));
        atom.tgtCrId = _db->crGrid().xyToIndex(sol.scIt->crX(), crY);
        auto &crInfo = _crInfoGrid.at(atom.tgtCrId);
        crInfo.rsrcDemOfType(shapeInfo.rsrcType) += atom.rsrcDem;
    }

    // Update the cost
    res.cost += sol.dist * shapeInfo.wt;
}
*/
/// The kernel function to perform NodeSet to clock region assignment
/// This function is based on the regret-minimization heuristic for general min-cost assignment problem
/// \param  cms  the given ClockMaskSet that tells with clock is masked in with clock region
/// \param  res  results of the node assignment
template<typename T>
void ClockNetworkPlanner<T>::nodeSetAssignmentRGTKernel(const ClockMaskSet &cms, NodeAssignResult &res)
{
    for (auto &p : _rsrcTypeToNodeSetInfoArray)
    {
        // Reset node assignment
        auto &nsInfoArray = p.second;
        for (auto &nsInfo : nsInfoArray)
        {
            nsInfo.tgtCrDemArray.clear();
        }

        // Build a general assignment problem
        GeneralAssignmentProblem gap(nsInfoArray.size(), _crInfoGrid.size());

        // Initialize item demands
        for (IndexType i = 0; i < nsInfoArray.size(); ++i)
        {
            gap.addItem(nsInfoArray.at(i).rsrcDem);
        }

        // Initialize bin capacity
        for (IndexType i = 0; i < _crInfoGrid.size(); ++i)
        {
            gap.addBin(_crInfoGrid.at(i).rsrcCapOfType(p.first));
        }

        // Fill item-to-bin cost map
        for (IndexType itemIdx = 0; itemIdx < gap.numItems(); ++itemIdx)
        {
            const auto &nsInfo = nsInfoArray.at(itemIdx);
            for (IndexType binIdx = 0; binIdx < gap.numBins(); ++binIdx)
            {
                const auto &crInfo = _crInfoGrid.at(binIdx);
                if (cms.isMasked(nsInfo.ckSig, binIdx) ||
                    crInfo.rsrcDemOfType(nsInfo.rsrcType) + nsInfo.rsrcDem > crInfo.rsrcCapOfType(nsInfo.rsrcType))
                {
                    // Violate clock mask constraint or the capacity constraint
                    continue;
                }
                gap.addCost(itemIdx, binIdx, getXYToClockRegionDist(nsInfo.xy, crInfo, nsInfo.rsrcType) * nsInfo.wt);
            }
        }

        // Run the assignment
        RegretMinimizationSolver rms(gap);
        rms.init();
        if (! rms.run())
        {
            // Cannot find a legal solution
            res.legal = false;
            return;
        }

        // Commit the legal assignment
        for (IndexType itemIdx = 0; itemIdx < gap.numItems(); ++itemIdx)
        {
            auto &nsInfo = nsInfoArray.at(itemIdx);
            nsInfo.tgtCrDemArray.emplace_back(rms.targetBinIndexOfItemIndex(itemIdx), 1.0);
            for (IndexType ckIdx : nsInfo.ckSig)
            {
                res.clockAssign.add(ckIdx, nsInfo.tgtCrDemArray.at(0).crId);
            }
        }
        res.cost += rms.cost();
    }
}

/// The kernel function to perform NodeSet to clock region assignment
/// This function solve the general assignment problem approximately using min-cost flow
/// \param  cms  the given ClockMaskSet that tells with clock is masked in with clock region
/// \param  res  results of the node assignment
template<typename T>
void ClockNetworkPlanner<T>::nodeSetAssignmentMCFKernel(const ClockMaskSet &cms, NodeAssignResult &res)
{
    for (auto &p : _rsrcTypeToNodeSetInfoArray)
    {
        // Reset node assignment
        RsrcType rsrcType = p.first;

        auto &nsInfoArray = p.second;
        for (auto &nsInfo : nsInfoArray)
        {
            nsInfo.tgtCrDemArray.clear();
        }

        // Build a min-cost bipartite matching problem, where node sets are left nodes and clock regions are right node
        LemonGraph g;
        LemonGraph::ArcMap<FlowIntType> lowerCap(g);
        LemonGraph::ArcMap<FlowIntType> upperCap(g);
        LemonGraph::ArcMap<FlowIntType> costMap(g);
        std::vector<LemonGraph::Node> lNodes, rNodes;
        std::vector<LemonGraph::Arc> lArcs, rArcs, mArcs;
        std::vector<ArcInfo> mArcInfos;

        // Source and target nodes
        LemonGraph::Node s = g.addNode(), t = g.addNode();

        // Add arcs between source and left (node sets)
        FlowIntType supply = 0;
        for (const auto &nsInfo : nsInfoArray) {
            lNodes.emplace_back(g.addNode());
            lArcs.emplace_back(g.addArc(s, lNodes.back()));
            costMap[lArcs.back()]  = 0;
            lowerCap[lArcs.back()] = 0;
            FlowIntType sup;
            if (rsrcType == RsrcType::SLICEL) {
                sup = nsInfo.rsrcDem * Parameters::cnpMinCostFlowCostScaling;
            } else {
                sup = nsInfo.rsrcDem;
            }
            upperCap[lArcs.back()] = sup;
            supply += sup;
        }
        //openparfPrint(kDebug, "Demand for resource %s is %i\n", RsrcTypeName[(uint32_t)rsrcType], supply);
        // Add arcs between right (clock regions) and target
        FlowIntType q = 0;
        for (const auto &crInfo : _crInfoGrid) {
            rNodes.emplace_back(g.addNode());
            rArcs.emplace_back(g.addArc(rNodes.back(), t));
            costMap[rArcs.back()]  = 0;
            lowerCap[rArcs.back()] = 0;
            FlowIntType r;
            if (rsrcType == RsrcType::SLICEL) {
                r = crInfo.rsrcCapOfType(rsrcType) * Parameters::cnpMinCostFlowCostScaling;
            } else {
                r = crInfo.rsrcCapOfType(rsrcType);
            }

            // LUT and FF do not interfere with each other when occupying slices.
            // So the effective cap is doubled for LUT and FF
            upperCap[rArcs.back()] = r;
            q += r;
            if (rsrcType == RsrcType::SLICEL) {
                upperCap[rArcs.back()] *= 2;
                q += r;
            }
        }
        openparfPrint(kDebug, "============================\n");
        openparfPrint(kDebug, "Supply for resource %s is %i\n", RsrcTypeName[(uint32_t) rsrcType],
                      supply);
        openparfPrint(kDebug, "Capacity for resource %s is %i\n", RsrcTypeName[(uint32_t) rsrcType],
                      q);

        // We incrementally add arcs between left (node sets) and right (clock regions) nodes
        // Each time we only add KNN feasible clock regions for each node set
        IndexType tgtNumCr = Parameters::cnpMinCostFlowNodeAssignNumNearestClockRegionInit;
        std::vector<IndexType> crIdxArray(nsInfoArray.size(), 0);
        bool                   feasible = false;
        while (!feasible) {
            // Add arcs between left and right
            IndexType numArcs = mArcs.size();
            for (IndexType l = 0; l < nsInfoArray.size(); ++l) {
                const auto &nsInfo = nsInfoArray.at(l);
                IndexType   numCr  = 0;
                IndexType & rIdx   = crIdxArray.at(l);
                FlowIntType sup;
                if (rsrcType == RsrcType::SLICEL) {
                    sup = (FlowIntType) ((RealType) nsInfo.rsrcDem *
                                         (RealType) Parameters::cnpMinCostFlowCostScaling);
                } else {
                    sup = nsInfo.rsrcDem;
                }
                while (numCr < tgtNumCr && rIdx < _crInfoGrid.size()) {
                    IndexType r = nsInfo.crIdArray.at(rIdx);
                    if (!cms.isMasked(nsInfo.ckSig, r)) {
                        mArcs.emplace_back(g.addArc(lNodes.at(l), rNodes.at(r)));
                        mArcInfos.emplace_back(l, r);
                        auto d = getXYToClockRegionDist(nsInfo.xy, _crInfoGrid.at(r),
                                                        nsInfo.rsrcType);
                        costMap[mArcs.back()] =
                                d * nsInfo.wt / sup * Parameters::cnpMinCostFlowCostScaling;
                        lowerCap[mArcs.back()] = 0;
                        upperCap[mArcs.back()] = sup;
                        ++numCr;
                    }
                    ++rIdx;
                }
            }
            if (numArcs == (IndexType) mArcs.size()) {
                // No new arc is added and current arcs cannot give a legal solution
                res.legal = false;
                return;
            }
            // Run the MCF
            LemonMinCostMaxFlowAlgorithm mcf(g);
            mcf.stSupply(s, t, supply);
            mcf.lowerMap(lowerCap).upperMap(upperCap).costMap(costMap);
            auto        rv       = mcf.run();
            FlowIntType flowSize = 0;
            for (const auto &arc : rArcs) { flowSize += mcf.flow(arc); }
            openparfPrint(kDebug,
                          "Min-cost Flow: flow_size: = %ld, flow_supply = %ld, cost = %ld\n",
                          flowSize, supply, mcf.totalCost());
            openparfAssert(rv != LemonMinCostMaxFlowAlgorithm::ProblemType::UNBOUNDED);
            if (flowSize == supply) {
                // A legal solution is found, commit the legal solution
                for (IndexType arcIdx = 0; arcIdx < mArcs.size(); ++arcIdx) {
                    const auto &arc = mArcs.at(arcIdx);
                    if (mcf.flow(arc)) {
                        const auto &ai     = mArcInfos.at(arcIdx);
                        auto &      nsInfo = nsInfoArray.at(ai.lIdx);
                        FlowIntType sup    = nsInfo.rsrcDem * Parameters::cnpMinCostFlowCostScaling;
                        nsInfo.tgtCrDemArray.emplace_back(ai.rIdx, (RealType) mcf.flow(arc) / sup);
                        for (IndexType ckIdx : nsInfo.ckSig) {
                            res.clockAssign.add(ckIdx, ai.rIdx);
                        }
                    }
                }
                auto cost_for_this = mcf.totalCost() / Parameters::cnpMinCostFlowCostScaling;
                res.cost += cost_for_this;
                openparfPrint(kDebug, "Cost for resource type %s is %f\n",
                              RsrcTypeName[(uint32_t) rsrcType], cost_for_this);
                feasible = true;
            } else {
                // Cannot find a legal solution, need to add more arcs
                openparfPrint(kInfo, "Incremental Legality Check: need to add more arcs");
                tgtNumCr = Parameters::cnpMinCostFlowNodeAssignNumNearestClockRegionIncr;
            }
        }
    }
}

/// Get the distance between a given XY to the closest site of given type in a given clock region
/// Return REAL_TYPE_MAX if no such site is in the clock region
template<typename T>
XY<RealType> ClockNetworkPlanner<T>::getDistXYToCrSiteOfType(const XY<RealType> &xy, const XY<IndexType> &crXY, SiteType st) const
{
    const auto &cr = _db->cr(crXY);
    IndexType siteId = _db->siteTypeToId(st);
    if (cr.numSites(siteId) == 0) { // MOD
        return XY<RealType>(REAL_TYPE_MAX, REAL_TYPE_MAX);
    }

    const auto &range = _db->siteColumnArrayOfType(st).rangeOfCrX(crXY.x()); // MOD
    auto scIt = range.end();

    // X distance
    RealType xDist = REAL_TYPE_MAX;
    if (xy.x() <= range.begin()->x())
    {
        xDist = range.begin()->x() - xy.x();
        scIt = range.begin();
    }
    else if (xy.x() >= std::prev(range.end())->x())
    {
        xDist = xy.x() - std::prev(range.end())->x();
        scIt = std::prev(range.end());
    }
    else
    {
        for (scIt = range.begin(); scIt != range.end(); ++scIt)
        {
            RealType dist = std::abs(scIt->x() - xy.x());
            if (dist >= xDist)
            {
                break;
            }
            xDist = dist;
        }
        --scIt;
    }

    const auto &seg = scIt->segmentOfCrY(crXY.y()); // MOD
    openparfAssert(scIt != range.end());
    openparfAssert(seg.numSites());

    // Y distance
    RealType yDist = REAL_TYPE_MAX;
    if (xy.y() <= *seg.begin())
    {
        yDist = *seg.begin() - xy.y();
    }
    else if (xy.y() >= *std::prev(seg.end()))
    {
        yDist = xy.y() - *std::prev(seg.end());
    }
    else
    {
        for (auto it = seg.begin(); it != seg.end(); ++it)
        {
            RealType dist = std::abs(*it - xy.y());
            if (dist >= yDist)
            {
                break;
            }
            yDist = dist;
        }
    }
    return XY<RealType>(xDist, yDist);
}

/// Run distribution layer routing for a given clock region assignment solution
/// \param  clockAssign  the given clock assignment
/// \param  dRouteArray  the resulting DRoute solution for each clock
/// \return              if the routing solution is overflow-free
template<typename T>
bool ClockNetworkPlanner<T>::runDLayerRouting(const ClockAssignment &clockAssign, std::vector<DRoute> &dRouteArray)
{
    buildDRoutesAndInitDRouteTracker(clockAssign);
    return selectDRoutesLR(dRouteArray);
}


/// Build all DRoutes for the given clock region assignment solution and use them to initialize the DRoute tracker
/// \param  clockAssign  the given clock assignment solution
template<typename T>
void ClockNetworkPlanner<T>::buildDRoutesAndInitDRouteTracker(const ClockAssignment &clockAssign)
{
    // Reset the DRoute tracker
    _drtk.candArray.clear();

    _drtk.selIdxArray.clear();
    _drtk.selIdxArray.resize(_nlPtr->numClockNets(), INDEX_TYPE_MAX);

    _drtk.ckIdxToDRouteIdxArray.resize(_nlPtr->numClockNets());
    for (auto &arr : _drtk.ckIdxToDRouteIdxArray) { arr.clear(); }

    _drtk.crToHoriDRouteIdxArray.resize(_db->numCrX(), _db->numCrY());
    for (auto &arr : _drtk.crToHoriDRouteIdxArray) { arr.clear(); }
    _drtk.crToVertDRouteIdxArray.resize(_db->numCrX(), _db->numCrY());
    for (auto &arr : _drtk.crToVertDRouteIdxArray) { arr.clear(); }

    // Build and add Droute candidates
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx) {
        for (IndexType tkX = 0; tkX < _db->numCrX(); ++tkX) {
            _drtk.addDRoute(ckIdx, buildDRoute(clockAssign, ckIdx, tkX));
        }
    }
}

template<typename T>
bool ClockNetworkPlanner<T>::runBBoxRouting(const ClockAssignment &clockAssign,
                                            ClockAvailCR &         clkAvailCR) {
    IndexType cks_num   = _nlPtr->numClockNets();
    IndexType x_cr_size = _db->numCrX();
    IndexType y_cr_size = _db->numCrY();
    clkAvailCR.ckIdxToAvailGrid().clear();
    clkAvailCR.ckIdxToAvailGrid().resize(cks_num, Vector2D<Byte>(x_cr_size, y_cr_size, 0));
    for (IndexType ckIdx = 0; ckIdx < cks_num; ++ckIdx) {
        IndexType cr_xl = std::numeric_limits<IndexType>::max();
        IndexType cr_yl = std::numeric_limits<IndexType>::max();
        IndexType cr_xh = std::numeric_limits<IndexType>::lowest();
        IndexType cr_yh = std::numeric_limits<IndexType>::lowest();
        for (IndexType crI = 0; crI < x_cr_size; crI++) {
            for (IndexType crJ = 0; crJ < y_cr_size; crJ++) {
                if (clockAssign.has(ckIdx, crI, crJ)) {
                    cr_xl = std::min(cr_xl, crI);
                    cr_xh = std::max(cr_xh, crI);
                    cr_yl = std::min(cr_yl, crJ);
                    cr_yh = std::max(cr_yh, crJ);
                }
            }
        }
        openparfAssert(cr_xl != std::numeric_limits<decltype(cr_xl)>::max());
        auto &availGrid = clkAvailCR.ckIdxToAvailGrid().at(ckIdx);
        for (IndexType crI = cr_xl; crI <= cr_xh; crI++) {
            for (IndexType crJ = cr_yl; crJ <= cr_yh; crJ++) { availGrid.at(crI, crJ) = 1; }
        }
    }
    return true;
}

/// Generate a distribution layer route for a given clock and a given clock region assignment
/// solution \param   clockAssign  the given clock assignment solution \param   tkX          the x
/// index for the given clock \return               the resulting route
template<typename T>
typename ClockNetworkPlanner<T>::DRoute
ClockNetworkPlanner<T>::buildDRoute(const ClockAssignment &clockAssign, IndexType ckIdx,
                                    IndexType tkX) const {
    DRoute res;
    res.tkX = tkX;

    for (IndexType y = 0; y < _db->numCrY(); ++y) {
        IndexType xLo = INDEX_TYPE_MAX, xHi = 0;
        for (IndexType x = 0; x < _db->numCrX(); ++x) {
            if (clockAssign.has(ckIdx, x, y)) {
                xLo = std::min(xLo, x);
                xHi = x;
            }
        }

        if (xLo != INDEX_TYPE_MAX) {
            // This row has clock loads
            res.tkYLo = std::min(res.tkYLo, y);
            res.tkYHi = y;

            // Add branches
            if (xLo <= tkX && xHi >= tkX) {
                res.branches.emplace_back(y, xLo, xHi);
            } else if (xLo < tkX) {
                res.branches.emplace_back(y, xLo, tkX);
            } else   // xHi > tkX
            {
                res.branches.emplace_back(y, tkX, xHi);
            }
            res.topoCost += res.branches.back().len();
        }
    }
    res.topoCost += res.tkYHi - res.tkYLo + 1;
    return res;
}

/// Perform DRoute selection using lagrangian relaxation (LR)
//. \output  dRouteArray  the DRoute solution of each clock
/// \return               if an overflow-free solution is found
template<typename T>
bool ClockNetworkPlanner<T>::selectDRoutesLR(std::vector<DRoute> &dRouteArray)
{
    DRouteGrid drg(_db->numCrX(), _db->numCrY());
    RealType topoCost = 0.0;
    dRouteArray.clear();
    dRouteArray.reserve(_nlPtr->numClockNets());

    // Get the initial solution
    // Pick the DRoute with the lowest cost from each clock
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {
        IndexType drIdx = _drtk.bestDRouteIdx(ckIdx);
        const DRoute &dr = _drtk.dRoute(drIdx);
        _drtk.setSelDRouteIdx(ckIdx, drIdx);
        dRouteArray.emplace_back(dr);
        drg.addDRouteDemand(dr);
        topoCost += dr.topoCost;
    }

    IndexType vdOverflow = drg.vdOverflow();
    IndexType hdOverflow = drg.hdOverflow();

    // Record the best solution so far
    RealType bestTopoCost = topoCost;
    IndexType bestOverflowCost = vdOverflow * Parameters::cnpVDOverflowCost + hdOverflow * Parameters::cnpHDOverflowCost;

    // For debugging
    //DBG("Init : Total VD/HD Overflow = %u/%u, topoCost = %.2lf, overflowCost = %u\n", vdOverflow, hdOverflow, bestTopoCost, bestOverflowCost);

    // Iterative update the initial solution
    // Change one DRoute at a time, until a feasible solution is found or the maximum iteration limit is reached
    _drtk.resetDRouteDeltaPenalty();
    IndexType iter = 0;
    while (iter < Parameters::cnpMaxDRouteLRIter && bestOverflowCost)
    {
        // Calculate base delta penalty of each DRoute based on the CR overflow
        for(IndexType crI = 0; crI < _db->numCrX(); crI++)
        for(IndexType crJ = 0; crJ < _db->numCrY(); crJ++)
        {
        XY<IndexType> xy(crI, crJ);
            // Calculate base delta penalty for VD
            if (drg.vdOverflow(xy))
            {
                RealType vddp = Parameters::cnpVDOverflowCost * drg.vdOverflow(xy) / _drtk.crToVertDRouteIdxArray.at(xy).size();
                for (IndexType drIdx : _drtk.crToVertDRouteIdxArray.at(xy))
                {
                    DRoute &dr = _drtk.dRoute(drIdx);
                    dr.deltaPenalty += vddp;
                }
            }

            // Calculate base delta penalty for HD
            if (drg.hdOverflow(xy))
            {
                RealType hddp = Parameters::cnpHDOverflowCost * drg.hdOverflow(xy) / _drtk.crToHoriDRouteIdxArray.at(xy).size();
                for (IndexType drIdx : _drtk.crToHoriDRouteIdxArray.at(xy))
                {
                    DRoute &dr = _drtk.dRoute(drIdx);
                    dr.deltaPenalty += hddp;
                }
            }
        }

        // Find the minium scaling factor for deltaPenalty to make at least one
        // unselected DRoute becomes as good as the corresponding selecetd one
        RealType scale = REAL_TYPE_MAX;
        IndexType tgtCkIdx = INDEX_TYPE_MAX;
        IndexType tgtDrIdx = INDEX_TYPE_MAX;
        for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
        {
            const auto &sel = _drtk.selDRoute(ckIdx);
            for (IndexType drIdx : _drtk.ckIdxToDRouteIdxArray.at(ckIdx))
            {
                // Calculate the min scale to make this DRoute become as good as 'sel'
                const auto &dr = _drtk.dRoute(drIdx);
                RealType dpDiff = sel.deltaPenalty - dr.deltaPenalty;
                if (dpDiff > Parameters::cnpDRouteDeltaPenaltyTol)
                {
                    // 'reqScale' is the required scale to reach even
                    RealType reqScale = (dr.cost() - sel.cost()) / dpDiff;
                    if (reqScale < scale)
                    {
                        scale = reqScale;
                        tgtCkIdx = ckIdx;
                        tgtDrIdx = drIdx;
                    }
                }
            }
        }

        if (tgtDrIdx == INDEX_TYPE_MAX)
        {
            // New solutions won't be reached even if DRoute costs are further adjusted
            break;
        }

        // Apply the scale to all deltaPenalty
        // Avoid negative sacle caused by round off error
        if (scale > 0)
        {
            _drtk.applyAndResetDRouteDeltaPenalty(scale);
        }

        // Substitue the current selected DRoute of 'tgtCkIdx' to DRoute 'tgtDrIdx'
        const DRoute &oldSel = _drtk.selDRoute(tgtCkIdx);
        const DRoute &newSel = _drtk.dRoute(tgtDrIdx);
        drg.removeDRouteDemand(oldSel);
        drg.addDRouteDemand(newSel);
        _drtk.setSelDRouteIdx(tgtCkIdx, tgtDrIdx);
        topoCost = topoCost - oldSel.topoCost + newSel.topoCost;
        vdOverflow = drg.vdOverflow();
        hdOverflow = drg.hdOverflow();
        IndexType overflowCost = vdOverflow * Parameters::cnpVDOverflowCost + hdOverflow * Parameters::cnpHDOverflowCost;

        // Update the best solution if applicable
        if (overflowCost < bestOverflowCost || (overflowCost <= bestOverflowCost && topoCost <= bestTopoCost))
        {
            bestTopoCost = topoCost;
            bestOverflowCost = overflowCost;
            dRouteArray.at(tgtCkIdx) = newSel;
        }
        ++iter;
        //DBG("Iter %u : Total VD/HD Overflow = %u/%u, topoCost = %.2lf, overflowCost = %u\n", iter, vdOverflow, hdOverflow, topoCost, overflowCost);
    }
    //DBG("Best : topoCost = %.2lf, overflowCost = %u\n", bestTopoCost, bestOverflowCost);
    return bestOverflowCost == 0;
}

/// Run routing-layer (R-layer) routing for a given D-layer routing solution
/// \param  dRouteArray  the given DRoute solution of each clock
/// \param  rRouteArray  the resulting RRoute solution of each clock
/// \return              if the routing solution is overflow-free
template<typename T>
bool ClockNetworkPlanner<T>::runRLayerRouting(const std::vector<DRoute> &dRouteArray, std::vector<RRoute> &rRouteArray) const
{
    // Determine routing ordering
    std::vector<IndexType> order;
    orderRLayerNetRouting(dRouteArray, order);

    // Initialization
    RRouteGrid rrg(_db->numCrX(), _db->numCrY());
    rRouteArray.resize(_nlPtr->numClockNets());

    // Perform initial routing
    for (IndexType ckIdx : order)
    {
        RRoute &rr = rRouteArray.at(ckIdx);
        routeRLayerNet(ckIdx, dRouteArray.at(ckIdx), rrg, rr);
        rrg.addRRouteDemand(rr);
        updateRRouteCostGrid(rr, rrg);
    }

    // Ripup and reroute phase
    std::vector<IndexType> ripupCkIdxArray;
    collectRipupClockNets(rrg, rRouteArray, ripupCkIdxArray);

    IndexType iter = 0;
    while (! ripupCkIdxArray.empty() && iter < Parameters::cnpMaxRRouteRipupIter)
    {
        for (IndexType ckIdx : ripupCkIdxArray)
        {
            RRoute &rr = rRouteArray.at(ckIdx);
            rrg.removeRRouteDemand(rr);
            routeRLayerNet(ckIdx, dRouteArray.at(ckIdx), rrg, rr);
            rrg.addRRouteDemand(rr);
            updateRRouteCostGrid(rr, rrg);
        }
        collectRipupClockNets(rrg, rRouteArray, ripupCkIdxArray);
        ++iter;
    }
    return ripupCkIdxArray.empty();
}

/// Determine routing ordering of R-layer clock nets for a given DRoute solution
/// \param   dRouteArray  the given DRoute solution
/// \output  order        the result clock order
template<typename T>
void ClockNetworkPlanner<T>::orderRLayerNetRouting(const std::vector<DRoute> &dRouteArray, std::vector<IndexType> &order) const
{
    // Get the minimum wirelength for each clock RRoute
    std::vector<IndexType> minWireLen(_nlPtr->numClockNets());
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {
        // Get the clock source crXY
        const auto &srcXY = _ckSrcInfoArray.at(ckIdx).crXY;

        const auto &dr = dRouteArray.at(ckIdx);
        IndexType xLen = (srcXY.x() >= dr.tkX ? srcXY.x() - dr.tkX : dr.tkX - srcXY.x());
        IndexType yLen = 0;
        if (srcXY.y() < dr.tkYLo)
        {
            yLen = dr.tkYLo - srcXY.y();
        }
        else if (srcXY.y() > dr.tkYHi)
        {
            yLen = srcXY.y() - dr.tkYHi;
        }

        // The first edge from source must be horizontal
        if ((xLen > 0 && yLen > 0) || (xLen == 0 && yLen > 0))
        {
            minWireLen.at(ckIdx) = xLen + yLen + 2;
        }
        else
        {
            minWireLen.at(ckIdx) = xLen + yLen + 1;
        }
    }

    // Sort all clock by their minimum wirelength from low to high
    order.resize(_nlPtr->numClockNets());
    std::iota(order.begin(), order.end(), 0);
    auto cmp = [&](IndexType l, IndexType r){ return minWireLen.at(l) < minWireLen.at(r); };
    std::sort(order.begin(), order.end(), cmp);
}

/// Route a clock in R layer given its corresponding DRoute solution
/// This is a A-star search based routing
/// \param   ckIdx  the index of the clock to route
/// \param   dr     the given DRoute solution of the clock
/// \param   rrg    the R-layer routing grid to use
/// \output  rr     the resulting RRoute solution
template<typename T>
void ClockNetworkPlanner<T>::routeRLayerNet(IndexType ckIdx, const DRoute &dr, RRouteGrid &rrg, RRoute &rr) const
{
    // reset all path cost to +INF
    rrg.resetPathCosts();

    // Generate a min heap to track all frontier points in A-star search
    boost::heap::fibonacci_heap<RRouteFrontier, boost::heap::compare<RRouteFrontierComp>> frontiers;

    // Push the clock source into the frontier set
    // In this clock architecture, the first routing edge from the source must be horizontal
    const auto &srcXY = _ckSrcInfoArray.at(ckIdx).crXY;
    RRouteFrontier src(srcXY, Orient::H, rrg.cost(srcXY, Orient::H));
    setRRouteEstCost(dr, src);
    frontiers.push(src);

    // Record the cost of the best legal routing found so far
    RealType minPathCost = REAL_TYPE_MAX;

    // Keep updating path costs until no active frontier points exist
    while (! frontiers.empty())
    {
        auto ft = frontiers.top();
        frontiers.pop();

        if (ft.detCost + ft.estCost >= minPathCost || ft.detCost >= rrg.pathCost(ft.xy, ft.orient))
        {
            // This is a sub-optimal route
            continue;
        }

        // Update the path cost
        rrg.pathCost(ft.xy, ft.orient) = ft.detCost;
        if (ft.x() == dr.tkX && ft.y() >= dr.tkYLo && ft.y() <= dr.tkYHi)
        {
            // This route hits the target, update the minPatCost,
            // and no need to continue this route
            minPathCost = ft.detCost;
            continue;
        }

        // Add neighbor grids into 'frontiers'
        if (ft.orient == Orient::H)
        {
            // Add left neighbor
            if (ft.x() > 0)
            {
                RRouteFrontier l(ft.xy.left(), Orient::H, ft.detCost + rrg.cost(ft.xy.left(), Orient::H));
                setRRouteEstCost(dr, l);
                frontiers.push(l);
            }
            // Add right neighbor
            if (ft.x() < _db->numCrX() - 1)
            {
                RRouteFrontier r(ft.xy.right(), Orient::H, ft.detCost + rrg.cost(ft.xy.right(), Orient::H));
                setRRouteEstCost(dr, r);
                frontiers.push(r);
            }
            // Add the neighbor that has different (here is Orient::V) orientation
            RRouteFrontier v(ft.xy, Orient::V, ft.detCost + rrg.cost(ft.xy, Orient::V));
            setRRouteEstCost(dr, v);
            frontiers.push(v);
        }
        else // ft.orient == Orient::V
        {
            // Add bottom neighbor
            if (ft.y() > 0)
            {
                RRouteFrontier b(ft.xy.bottom(), Orient::V, ft.detCost + rrg.cost(ft.xy.bottom(), Orient::V));
                setRRouteEstCost(dr, b);
                frontiers.push(b);
            }
            // Add top neighbor
            if (ft.y() < _db->numCrY() - 1)
            {
                RRouteFrontier t(ft.xy.top(), Orient::V, ft.detCost + rrg.cost(ft.xy.top(), Orient::V));
                setRRouteEstCost(dr, t);
                frontiers.push(t);
            }
            // Add the neighbor that has different (here is Orient::H) orientation
            RRouteFrontier h(ft.xy, Orient::H, ft.detCost + rrg.cost(ft.xy, Orient::H));
            setRRouteEstCost(dr, h);
            frontiers.push(h);
        }
    }

    // Find the routing solution
    //

    // Find the end point with the minimum cost
    RealType minCost = REAL_TYPE_MAX;
    IndexType endY = INDEX_TYPE_MAX;
    Orient endOrient = Orient::H;
    for (IndexType y = dr.tkYLo; y <= dr.tkYHi; ++y)
    {
        for (Orient orient : {Orient::H, Orient::V})
        {
            RealType cost = rrg.pathCost(XY<IndexType>(dr.tkX, y), orient);
            if (cost < minCost)
            {
                minCost = cost;
                endY = y;
                endOrient = orient;
            }
        }
    }

    // Backtrace from (dr.tkX, endY, endOrient) to the clock source
    XY<IndexType> curXY(dr.tkX, endY);
    Orient curOrient = endOrient;

    rr.edgeArray.clear();
    rr.edgeArray.reserve(_db->numCrX() + _db->numCrY());
    rr.edgeArray.emplace_back(curXY, curOrient);

    while (curXY != srcXY || curOrient != Orient::H)
    {
        if (curOrient == Orient::H)
        {
            RealType lCost = (curXY.x() > 0 ? rrg.pathCost(curXY.left(), Orient::H) : REAL_TYPE_MAX);
            RealType rCost = (curXY.x() < _db->numCrX() - 1 ? rrg.pathCost(curXY.right(), Orient::H) : REAL_TYPE_MAX);
            RealType vCost = rrg.pathCost(curXY, Orient::V);

            if (lCost <= rCost && lCost <= vCost)
            {
                // The left neighbor has the min cost
                curXY.toLeft();
            }
            else if (rCost <= lCost && rCost <= vCost)
            {
                // The right neighbor has the min cost
                curXY.toRight();
            }
            else // vCost <= lCost && vCost <= rCost
            {
                // The neighbor with different orientation has the min cost
                curOrient = Orient::V;
            }
        }
        else // curOrient == Orient::V
        {
            RealType bCost = (curXY.y() > 0 ? rrg.pathCost(curXY.bottom(), Orient::V) : REAL_TYPE_MAX);
            RealType tCost = (curXY.y() < _db->numCrY() - 1 ? rrg.pathCost(curXY.top(), Orient::V) : REAL_TYPE_MAX);
            RealType hCost = rrg.pathCost(curXY, Orient::H);

            if (bCost <= tCost && bCost <= hCost)
            {
                // The bottom neighbor has the min cost
                curXY.toBottom();
            }
            else if (tCost <= bCost && tCost <= hCost)
            {
                // The top neighbor has the min cost
                curXY.toTop();
            }
            else // hCost <= bCost && hCost <= tCost
            {
                // The neighbor with different orientation has the min cost
                curOrient = Orient::H;
            }
        }
        rr.edgeArray.emplace_back(curXY, curOrient);
    }

    // Make edges sorted from source to sink
    std::reverse(rr.edgeArray.begin(), rr.edgeArray.end());
    openparfAssert(rr.edgeArray.front().xy == srcXY && rr.edgeArray.front().orient == Orient::H);
}

/// Set the estimated remaining cost for a RRoute frontier point
/// \param   dr  the corresponding DRoute solution
/// \output  rf  the RRoute frontier to be estimated
template<typename T>
void ClockNetworkPlanner<T>::setRRouteEstCost(const DRoute &dr, RRouteFrontier &rf) const
{
    IndexType xLen = (rf.x() >= dr.tkX ? rf.x() - dr.tkX : dr.tkX - rf.x());
    IndexType yLen = 0;
    if (rf.y() < dr.tkYLo)
    {
        yLen = dr.tkYLo - rf.y();
    }
    else if (rf.y() > dr.tkYHi)
    {
        yLen = rf.y() - dr.tkYHi;
    }

    if ((xLen > 0 && yLen > 0) ||
        (xLen == 0 && yLen > 0 && rf.orient == Orient::H) ||
        (xLen > 0 && yLen == 0 && rf.orient == Orient::V))
    {
        // Change direction takes one extra resource
        rf.estCost = xLen + yLen + 1;
    }
    else
    {
        rf.estCost = xLen + yLen;
    }
}

/// Update RRoute cost grid after adding a given RRoute
/// \param  rr   the given RRoute
/// \param  rrg  the RRoute grid to be updated
template<typename T>
void ClockNetworkPlanner<T>::updateRRouteCostGrid(const RRoute &rr, RRouteGrid &rrg) const
{
    for (const auto &e : rr.edgeArray)
    {
        if (rrg.dem(e.xy, e.orient) >= Parameters::archClockRegionClockCapacity)
        {
            rrg.cost(e.xy, e.orient) += Parameters::cnpRRouteOverflowCostIncrValue;
        }
    }
}

/// Collect the set of clocks need to rip up and reroute
/// \param   rrg              the given RRoute grid
/// \param   rRouteArray      the given RRoute solution of each clock
/// \output  ripupCkIdxArray  the set of clock nets needs to rip up, they are sorted in their rip up order
template<typename T>
void ClockNetworkPlanner<T>::collectRipupClockNets(const RRouteGrid &rrg, const std::vector<RRoute> &rRouteArray, std::vector<IndexType> &ripupCkIdxArray) const
{
    ripupCkIdxArray.clear();
    ripupCkIdxArray.reserve(_nlPtr->numClockNets());
    std::vector<IndexType> overflow(_nlPtr->numClockNets(), 0);
    IndexType dbgOverflow = 0;

    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {
        const auto &rr = rRouteArray.at(ckIdx);
        for (const auto &e : rr.edgeArray)
        {
            //if (rrg.dem(e.xy, e.orient) > Parameters::archClockRegionClockCapacity)
            if (rrg.dem(e.xy, e.orient) > 24)
            {
                overflow.at(ckIdx) += 1;
            }
        }
        if (overflow.at(ckIdx))
        {
            ripupCkIdxArray.push_back(ckIdx);
            dbgOverflow += overflow.at(ckIdx);
        }
    }

    // Sort all ripup clock by their overflow from low to high
    auto cmp = [&](IndexType l, IndexType r){ return overflow.at(l) < overflow.at(r); };
    std::sort(ripupCkIdxArray.begin(), ripupCkIdxArray.end(), cmp);
}

/// Given a not yet legal Solution, generate a set of new ClockMaskSets
/// We hope that legal solutions can be found using these new ClockMaskSets
/// These newly generated clock mask sets are sorted by their estimated cost from low to high
/// \param   sol          the given Solution
/// \param   maxCost      we discard solutions with cost larger than this value
/// \param   cmsPool      the pool of existing clock mask sets, for avoiding generating duplicate
/// clock mask sets \output  newCMSArray  the set of newly generated ClockMaskSets based on cm
template<typename T>
void ClockNetworkPlanner<T>::createNewClockMaskSets(const Solution &sol, RealType maxCost,
                                                    ClockMaskSetPool &         cmsPool,
                                                    std::vector<ClockMaskSet> &newCMSArray) const {
    newCMSArray.clear();
    newCMSArray.reserve(_nlPtr->numClockNets());

    //    // Generate DRoute grid using the given DRoute solution
    //    DRouteGrid drg(_db->numCrX(), _db->numCrY());
    //    for (const auto &dr : sol.dRouteArray)
    //    {
    //        drg.addDRouteDemand(dr);
    //    }
    //
    //    // Get the most HD and VD overflowed clock region
    //    XY<IndexType> vdCrXY, hdCrXY;
    //    IndexType maxVdOverflow = 0, maxHdOverflow = 0;
    //    for(IndexType crI = 0; crI < _db->numCrX(); crI++)
    //    for(IndexType crJ = 0; crJ < _db->numCrY(); crJ++)
    //    {
    //        XY<IndexType> xy(crI, crJ);
    //        if (drg.vdOverflow(xy) > maxVdOverflow)
    //        {
    //            maxVdOverflow = drg.vdOverflow(xy);
    //            vdCrXY        = xy;
    //        }
    //        if (drg.hdOverflow(xy) > maxHdOverflow) {
    //            maxHdOverflow = drg.hdOverflow(xy);
    //            hdCrXY        = xy;
    //        }
    //    }

    // Create new clock masks to resolve the most overflowed vertical or horizontal D-layer overflow
    //    if (maxVdOverflow * Parameters::cnpVDOverflowCost >
    //        maxHdOverflow * Parameters::cnpHDOverflowCost) {
    //        createNewClockMaskSetsToResolveVDOverflow(sol, vdCrXY, maxCost, cmsPool, newCMSArray);
    //    } else {
    //        createNewClockMaskSetsToResolveHDOverflow(sol, hdCrXY, maxCost, cmsPool, newCMSArray);
    //    }

    XY<IndexType> targetCrXY;
    IndexType     maxOverflow = 0;
    for (IndexType crI = 0; crI < _db->numCrX(); crI++) {
        for (IndexType crJ = 0; crJ < _db->numCrY(); crJ++) {
            if (sol.crCkCountMap.at(crI, crJ) > maxOverflow) {
                maxOverflow = sol.crCkCountMap.at(crI, crJ);
                targetCrXY  = XY<IndexType>(crI, crJ);
            }
        }
    }
    openparfPrint(kDebug, "target CR: (%d, %d)\n", targetCrXY.x(), targetCrXY.y());
    openparfAssert(maxOverflow > Parameters::archClockRegionClockCapacity);
    createNewClockMaskSetsToResolveBBoxOverflow(sol, targetCrXY, maxCost, cmsPool, newCMSArray);

    // Sort all resulting clock mask sets by their estimated cost from low to high
    // The order determines the solution exploration order
    std::sort(newCMSArray.begin(), newCMSArray.end());
}

/// Given a not yet legal Solution, generate a set of new ClockMaskSets to resolve vertical D-layer overflow
/// \param   sol          the given Solution
/// \param   crXY         the target overflowed clock region to be resolved
/// \param   maxCost      we discard solutions with cost larger than this value
/// \param   cmsPool      the pool of existing clock mask sets, for avoiding generating duplicate clock mask sets
/// \output  newCMSArray  the set of newly generated ClockMaskSets based on cm
template<typename T>
void ClockNetworkPlanner<T>::createNewClockMaskSetsToResolveVDOverflow(const Solution &sol,
                                                                    const XY<IndexType> &crXY,
                                                                    RealType maxCost,
                                                                    ClockMaskSetPool &cmsPool,
                                                                    std::vector<ClockMaskSet> &newCMSArray) const
{
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {
        const auto &dr = sol.dRouteArray.at(ckIdx);
        if (! dr.occupyVD(crXY))
        {
            continue;
        }
        // We generate four clock mask sets
        // 1) Masks all CRs that are below the overflowed CR (including the overflowed CR)
        // 2) Masks all CRs that are above the overflowed CR (including the overflowed CR)
        // 3) Masks all CRs that are in the left of the overflowed CR (including the overflowed CR)
        // 4) Masks all CRs that are in the right of the overflowed CR (including the overflowed CR)
        for (auto bbox : {Box<IndexType>(0, 0, _db->numCrX() - 1, crXY.y()),
                          Box<IndexType>(0, crXY.y(), _db->numCrX() - 1, _db->numCrY() - 1),
                          Box<IndexType>(0, 0, crXY.x(), _db->numCrY() - 1),
                          Box<IndexType>(crXY.x(), 0, _db->numCrX() - 1, _db->numCrY() - 1)})
        {
            ClockMaskSet newCMS(sol.cms);
            for(IndexType i = bbox.xl(); i <= bbox.xh(); i++)
            for(IndexType j = bbox.yl(); j <= bbox.yh(); j++)
            {
                XY<IndexType> ij(i, j);
                newCMS.mask(ckIdx, ij);
            }
            if (cmsPool.addClockMaskSet(newCMS))
            {
                // Add it only if the new clock mask set is not duplicate
                // Before adding, update the cost of this clock mask set
                newCMS.cost -= newCMS.cmCostArray.at(ckIdx);
                newCMS.cmCostArray.at(ckIdx) = getClockMaskCost(ckIdx, newCMS);
                newCMS.cost += newCMS.cmCostArray.at(ckIdx);
                if (newCMS.cost < maxCost)
                {
                    newCMSArray.emplace_back(newCMS);
                }
            }
        }
    }
}

/// Given a not yet legal Solution, generate a set of new ClockMaskSets to resolve horizontal D-layer overflow
/// \param   sol          the given Solution
/// \param   crXY         the target overflowed clock region to be resolved
/// \param   maxCost      we discard solutions with cost larger than this value
/// \param   cmsPool      the pool of existing clock mask sets, for avoiding generating duplicate clock mask sets
/// \output  newCMSArray  the set of newly generated ClockMaskSets based on cm
template<typename T>
void ClockNetworkPlanner<T>::createNewClockMaskSetsToResolveHDOverflow(const Solution &sol,
                                                                    const XY<IndexType> &crXY,
                                                                    RealType maxCost,
                                                                    ClockMaskSetPool &cmsPool,
                                                                    std::vector<ClockMaskSet> &newCMSArray) const
{
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {
        const auto &dr = sol.dRouteArray.at(ckIdx);
        if (! dr.occupyHD(crXY))
        {
            continue;
        }

        // We generate four clock mask sets
        // 1) Masks all CRs that are below the overflowed CR (including the overflowed CR)
        // 2) Masks all CRs that are above the overflowed CR (including the overflowed CR)
        // 3) Masks all CRs that are in the left of the overflowed CR (including the overflowed CR)
        // 4) Masks all CRs that are in the right of the overflowed CR (including the overflowed CR)

        //for (auto bbox : {Box<IndexType>(0, 0, crXY.x(), crXY.y()),
                          //Box<IndexType>(crXY.x(), crXY.y(), _db->numCrX() - 1, _db->numCrY() - 1),
                          //Box<IndexType>(0, crXY.y(), crXY.x(), _db->numCrY() - 1),
                          //Box<IndexType>(crXY.x(), 0, _db->numCrX() - 1, crXY.y())})
        //for (auto bbox : {Box<IndexType>(0, crXY.y(), crXY.x(), crXY.y()),
                          //Box<IndexType>(crXY.x(), crXY.y(), _db->numCrX() - 1, crXY.y())})
        for (auto bbox : {Box<IndexType>(0, 0, _db->numCrX() - 1, crXY.y()),
                          Box<IndexType>(0, crXY.y(), _db->numCrX() - 1, _db->numCrY() - 1),
                          Box<IndexType>(0, 0, crXY.x(), _db->numCrY() - 1),
                          Box<IndexType>(crXY.x(), 0, _db->numCrX() - 1, _db->numCrY() - 1)})
        {
            ClockMaskSet newCMS(sol.cms);
            for(IndexType i = bbox.xl(); i <= bbox.xh(); i++)
            for(IndexType j = bbox.yl(); j <= bbox.yh(); j++)
            {
                XY<IndexType> ij(i, j);
                newCMS.mask(ckIdx, ij);
            }
            if (cmsPool.addClockMaskSet(newCMS))
            {
                // Add it only if the new clock mask set is not duplicate
                // Before adding, update the cost of this clock mask set
                newCMS.cost -= newCMS.cmCostArray.at(ckIdx);
                newCMS.cmCostArray.at(ckIdx) = getClockMaskCost(ckIdx, newCMS);
                newCMS.cost += newCMS.cmCostArray.at(ckIdx);
                if (newCMS.cost < maxCost)
                {
                    newCMSArray.emplace_back(newCMS);
                }
            }
        }
    }
}

/// Given a not yet legal Solution, generate a set of new ClockMaskSets to resolve vertical D-layer
/// overflow \param   sol          the given Solution \param   crXY         the target overflowed
/// clock region to be resolved \param   maxCost      we discard solutions with cost larger than
/// this value \param   cmsPool      the pool of existing clock mask sets, for avoiding generating
/// duplicate clock mask sets \output  newCMSArray  the set of newly generated ClockMaskSets based
/// on cm
template<typename T>
void ClockNetworkPlanner<T>::createNewClockMaskSetsToResolveBBoxOverflow(
        const Solution &sol, const XY<IndexType> &crXY, RealType maxCost, ClockMaskSetPool &cmsPool,
        std::vector<ClockMaskSet> &newCMSArray) const {
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx) {
        const auto &ckAvailGrid = sol.clkAvailCR.ckIdxToAvailGrid()[ckIdx];
        if (!ckAvailGrid.at(crXY)) { continue; }
        // We generate four clock mask sets
        // 1) Masks all CRs that are below the overflowed CR (including the overflowed CR)
        // 2) Masks all CRs that are above the overflowed CR (including the overflowed CR)
        // 3) Masks all CRs that are in the left of the overflowed CR (including the overflowed CR)
        // 4) Masks all CRs that are in the right of the overflowed CR (including the overflowed CR)
        for (auto bbox : {Box<IndexType>(0, 0, _db->numCrX() - 1, crXY.y()),
                          Box<IndexType>(0, crXY.y(), _db->numCrX() - 1, _db->numCrY() - 1),
                          Box<IndexType>(0, 0, crXY.x(), _db->numCrY() - 1),
                          Box<IndexType>(crXY.x(), 0, _db->numCrX() - 1, _db->numCrY() - 1)}) {
            ClockMaskSet newCMS(sol.cms);
            for (IndexType i = bbox.xl(); i <= bbox.xh(); i++)
                for (IndexType j = bbox.yl(); j <= bbox.yh(); j++) {
                    XY<IndexType> ij(i, j);
                    newCMS.mask(ckIdx, ij);
                }
            if (cmsPool.addClockMaskSet(newCMS)) {
                // Add it only if the new clock mask set is not duplicate
                // Before adding, update the cost of this clock mask set
                newCMS.cost -= newCMS.cmCostArray.at(ckIdx);
                newCMS.cmCostArray.at(ckIdx) = getClockMaskCost(ckIdx, newCMS);
                newCMS.cost += newCMS.cmCostArray.at(ckIdx);
                if (newCMS.cost < maxCost) { newCMSArray.emplace_back(newCMS); }
            }
        }
    }
}
/// Calculate clock mask cost for a given clock
/// The cost is defined as the (not tight) lower bound cell movement
/// \param  ckIdx    the index of the given clock
/// \param  cms      the given clock mask set
/// \return          the mask cost
template<typename T>
RealType ClockNetworkPlanner<T>::getClockMaskCost(IndexType ckIdx, const ClockMaskSet &cms) const {
    RealType cost = 0;

    // Estimate shape movement cost
    // We align shape to their legal location for the estimation
    /*
    for (const auto &p : _rsrcTypeToCkIdxToShapeIdxArray)
    {
        const auto &shapeInfoArray = _rsrcTypeToShapeInfoArray.at(p.first);
        for (IndexType shapeIdx : p.second.at(ckIdx))
        {
            const auto &shapeInfo = shapeInfoArray.at(shapeIdx);

            // Get the shape clock region span and check if any of these clock region has been
    masked IndexType crYLo = _db->yToCrY(shapeInfo.yLo()); IndexType crYHi =
    _db->yToCrY(std::min(shapeInfo.yHi(), _db->numSiteY() - 1.0)); for (IndexType crY = crYLo; crY
    <= crYHi; ++crY)
            {
                // Check the clock legality
                if (cms.isMasked(shapeInfo.ckSig, _db->xToCrX(shapeInfo.x()), crY))
                {
                    // The current shape location is not clock legal
                    // We need to find the legal assignment with the minimum cost
                    auto shapeAssignSol = getShapeAssignment(cms, shapeInfo);
                    if (shapeAssignSol.dist == REAL_TYPE_MAX)
                    {
                        // Cannot find a legal position to assign this shape
                        return false;
                    }
                    cost += shapeAssignSol.dist * shapeInfo.wt;
                }
            }
        }
    }*/

    // Estimate node set movement cost
    for (const auto &p : _rsrcTypeToCkIdxToNodeSetIdxArray)
    {
        const auto &nsInfoArray = _rsrcTypeToNodeSetInfoArray.at(p.first);
        for (IndexType nsIdx : p.second.at(ckIdx))
        {
            const auto &nsInfo = nsInfoArray.at(nsIdx);
            if (cms.isMasked(nsInfo.ckSig, _db->siteCrId(nsInfo.xy)))
            {
                // The clock region that his node set currently is in has been masked
                // We need to find the clock region to with the minimum assignment cost
                RealType minDist = REAL_TYPE_MAX;
                for (const auto &crInfo : _crInfoGrid)
                {
                    if (! cms.isMasked(nsInfo.ckSig, crInfo.crXY))
                    {
                        minDist = std::min(minDist, getXYToClockRegionDist(nsInfo.xy, crInfo, nsInfo.rsrcType));
                    }
                }
                if (minDist == REAL_TYPE_MAX)
                {
                    // Cannot find feasible clock region to assign
                    return REAL_TYPE_MAX;
                }
                cost += minDist * nsInfo.wt;
            }
        }
    }
    return cost;
}

/// Commit the the given solution as the final solution
template<typename T>
void ClockNetworkPlanner<T>::commitSolution(const Solution &sol) {
    _clockMaskSet = sol.cms;
    _dRouteArray  = sol.dRouteArray;
    _rRouteArray  = sol.rRouteArray;

    // Fill the clock availability
    // For a clock, all the clock regions that covered by its DRoute is avaiable
    //    _clockAvailCR.ckIdxToAvailGrid().clear();
    //    _clockAvailCR.ckIdxToAvailGrid().resize(_nlPtr->numClockNets(),
    //    Vector2D<Byte>(_db->numCrX(), _db->numCrY(), 0)); for (IndexType ckIdx = 0; ckIdx <
    //    _nlPtr->numClockNets(); ++ckIdx)
    //    {
    //        const auto &dr = sol.dRouteArray.at(ckIdx);
    //        auto &availGrid = _clockAvailCR.ckIdxToAvailGrid().at(ckIdx);
    //        for (const auto &br : dr.branches)
    //        {
    //            for (IndexType x = br.xLo; x <= br.xHi; ++x)
    //            {
    //                availGrid.at(x, br.y) = 1;
    //            }
    //        }
    //    }
    _clockAvailCR = sol.clkAvailCR;

    openparfAssert(commitNodeAssignment(sol.cms));
}

/// Commit the node assignment solution for the given clock mask set
template<typename T>
bool ClockNetworkPlanner<T>::commitNodeAssignment(const ClockMaskSet &cms)
{
    // Perfrom node assignment and get the nodes in each clock region
    // Note that we must be able to get a legal node assignment solution from the sol.cms
    NodeAssignResult nodeAssignRes = runNodeAssignment(cms);
    if (! nodeAssignRes.legal)
    {
        return false;
    }
    openparfPrint(MessageType::kDebug, "commit node assignment cost %.2lf\n", nodeAssignRes.cost);

    _crToNodeIdArray.clear();
    _crToNodeIdArray.resize(_db->numCrX(), _db->numCrY());

    // Assign nodes in shapes
    /*
    for (const auto &p : _rsrcTypeToShapeInfoArray)
    {
        for (const auto &shapeInfo : p.second)
        {
            for (const auto &atom : shapeInfo.atomInfoArray)
            {
                _crToNodeIdArray.at(atom.tgtCrId).push_back(atom.nodeId);
            }
        }
    }*/
    // Assign nodes in node sets
    for (auto &p : _rsrcTypeToNodeSetInfoArray)
    {
        for (auto &nsInfo : p.second)
        {
            realizeNodeAssignment(nsInfo);
            for (const auto &crDem : nsInfo.tgtCrDemArray)
            {
                auto &nodeIdArray = _crToNodeIdArray.at(crDem.crId);
                nodeIdArray.insert(nodeIdArray.end(), crDem.nodeIdArray.begin(), crDem.nodeIdArray.end());
            }
        }
    }
    return true;
}

/// Given a node set, if it is assigned to more than one clock region, this function determines the detailed node to clock region assignment
/// If it is assigned to only one clock region, all the nodes in this node set will be moved to the target clock region
template<typename T>
void ClockNetworkPlanner<T>::realizeNodeAssignment(NodeSetInfo &nsInfo)
{
    if (nsInfo.tgtCrDemArray.size() == 1)
    {
        nsInfo.tgtCrDemArray.at(0).nodeIdArray = nsInfo.nodeIdArray;
        return;
    }

    // If the execution hits here, this node set is assigned to more than clock regions
    // We here use another level of min-cost flow to determine the final node-to-clock region assignment
    //

    // Get the resource demand of each node in this node set
    std::vector<FlowIntType> rsrcDemArray;
    rsrcDemArray.reserve(nsInfo.nodeIdArray.size());
    if (nsInfo.rsrcType == RsrcType::DSP || nsInfo.rsrcType == RsrcType::RAM)
    {
        for (IndexType i : nsInfo.nodeIdArray)
        {
            rsrcDemArray.push_back(_nlPtr->nodeRsrcDem(i) * Parameters::cnpMinCostFlowCostScaling);
        }
    }
    else
    {
        for (IndexType i : nsInfo.nodeIdArray)
        {
            auto area= _nlPtr->nodeArea(i);
            rsrcDemArray.push_back(area * Parameters::cnpMinCostFlowCostScaling);
        }
    }
    FlowIntType totalRsrcDem = std::accumulate(rsrcDemArray.begin(), rsrcDemArray.end(), 0);

    // Get the resource capacity of each clock region
    std::vector<FlowIntType> rsrcCapArray;
    rsrcCapArray.reserve(nsInfo.tgtCrDemArray.size());
    for (const auto &crDem : nsInfo.tgtCrDemArray)
    {
        rsrcCapArray.push_back(std::ceil(crDem.rsrcDemRatio * totalRsrcDem));
    }

    // Build a min-cost bipartite matching problem, where node sets are left nodes and clock regions are right node
    LemonGraph g;
    LemonGraph::ArcMap<FlowIntType> lowerCap(g);
    LemonGraph::ArcMap<FlowIntType> upperCap(g);
    LemonGraph::ArcMap<FlowIntType> costMap(g);
    std::vector<LemonGraph::Node> lNodes, rNodes;
    std::vector<LemonGraph::Arc> lArcs, rArcs, mArcs;
    std::vector<ArcInfo> mArcInfos;

    // Source and target nodes
    LemonGraph::Node s = g.addNode(), t = g.addNode();

    // Add arcs between source and left (node sets)
    for (IndexType i = 0; i < rsrcDemArray.size(); ++i)
    {
        lNodes.emplace_back(g.addNode());
        lArcs.emplace_back(g.addArc(s, lNodes.back()));
        costMap[lArcs.back()] = 0;
        lowerCap[lArcs.back()] = 0;
        upperCap[lArcs.back()] = rsrcDemArray.at(i);
    }

    // Add arcs between right (clock regions) and target
    for (IndexType i = 0; i < rsrcCapArray.size(); ++i)
    {
        rNodes.emplace_back(g.addNode());
        rArcs.emplace_back(g.addArc(rNodes.back(), t));
        costMap[rArcs.back()] = 0;
        lowerCap[rArcs.back()] = 0;
        upperCap[rArcs.back()] = rsrcCapArray.at(i);
    }

    // Add arcs between left and right
    for (IndexType l = 0; l < rsrcDemArray.size(); ++l)
    {
        const auto nodeId = nsInfo.nodeIdArray.at(l);
        for (IndexType r = 0; r < rsrcCapArray.size(); ++r)
        {
            const auto &crInfo = _crInfoGrid.at(nsInfo.tgtCrDemArray.at(r).crId);
            mArcs.emplace_back(g.addArc(lNodes.at(l), rNodes.at(r)));
            mArcInfos.emplace_back(l, r);
            auto rsrcType = _nlPtr->nodeRsrcType(nodeId);
            auto xy = _nlPtr->getXYFromNodeId(nodeId);
            costMap[mArcs.back()] = getXYToClockRegionDist(xy, crInfo, rsrcType) * _nlPtr->nodeNumPins(nodeId) / rsrcDemArray.at(l) * Parameters::cnpMinCostFlowCostScaling;
            lowerCap[mArcs.back()] = 0;
            upperCap[mArcs.back()] = rsrcDemArray.at(l);
        }
    }

    // Run the MCF
    LemonMinCostMaxFlowAlgorithm mcf(g);
    mcf.stSupply(s, t, totalRsrcDem);
    mcf.lowerMap(lowerCap).upperMap(upperCap).costMap(costMap);
    mcf.run();

    FlowIntType flowSize = 0;
    for (const auto &arc : rArcs)
    {
        flowSize += mcf.flow(arc);
    }
    openparfAssert(flowSize == totalRsrcDem);

    // Realize the flow, assign the node to the clock region with the largest flow size
    std::vector<std::pair<FlowIntType, IndexType>> nodeIdxToCrIdx(nsInfo.nodeIdArray.size(), std::pair<FlowIntType, IndexType>(0, 0));
    for (IndexType arcIdx = 0; arcIdx < mArcs.size(); ++arcIdx)
    {
        const auto &arc = mArcs.at(arcIdx);
        const auto &ai = mArcInfos.at(arcIdx);
        auto &p = nodeIdxToCrIdx.at(ai.lIdx);
        if (mcf.flow(arc) > p.first)
        {
            p.second = ai.rIdx;
        }
    }

    // Commit the assignment
    for (auto &crDem : nsInfo.tgtCrDemArray)
    {
        crDem.nodeIdArray.clear();
    }
    for (IndexType nodeIdx = 0; nodeIdx < nodeIdxToCrIdx.size(); ++nodeIdx)
    {
        IndexType nodeId = nsInfo.nodeIdArray.at(nodeIdx);
        IndexType crIdx = nodeIdxToCrIdx.at(nodeIdx).second;
        nsInfo.tgtCrDemArray.at(crIdx).nodeIdArray.push_back(nodeId);
    }
}

/// Given a clock-legal solution (both clock region and half-column region constraints are satisfied),
/// plan the clock availability in each half-column region.
/// This planning is a Gaussian-based probability estimation, we calculate the occupancy probability of each clock net
/// in each half-column region by assuming that each node will be disturbed by a Gaussian noise around its current location
template<typename T>
void ClockNetworkPlanner<T>::planHalfColumnRegionClockAvailability()
{
    // Calculate the probability of each clock in each half-column region
    // ckIdxToInProbArray[ckIdx][hcId] = invProb means the probability that ckIdx does NOT occupy hcId is invProb
    std::vector<std::vector<RealType>> ckIdxToInvProbArray(_nlPtr->numClockNets(), std::vector<RealType>(_db->hcSize(), 1.0));
    for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {

        //const Net &net = _nlPtr->clockNet(ckIdx);
        IndexType netId = _nlPtr->clockNet(ckIdx);
        auto &invProbs = ckIdxToInvProbArray.at(ckIdx);
        auto numPins = _nlPtr->netSize(netId);
        for (IndexType pinIdxInNet = 0; pinIdxInNet < numPins; pinIdxInNet++)
        {
            IndexType pinId = _nlPtr->netPin(netId, pinIdxInNet);
            IndexType nodeId = _nlPtr->pinToNodeId(pinId);
            if (_nlPtr->isNodeClockSource(nodeId))
            {
                continue;
            }
            for (IndexType i = 0; i < _db->hcSize(); i++)
            {
                // Calculate the probability that this node is in this half-column region
                RealType prob = nodeToHcProbability(nodeId, i);
                invProbs.at(i) *= (1.0 - prob);
            }
        }
    }

    // In each half-column region, allow the top Parameter::archHalfColumnRegionClockCapacity number of clocks
    _clockAvailHC.ckIdxToAvailArray().clear();
    _clockAvailHC.ckIdxToAvailArray().resize(_nlPtr->numClockNets(), std::vector<Byte>(_db->hcSize(), 0));
    for (IndexType hcId = 0; hcId < _db->hcSize(); ++hcId)
    {
        // Collect the probability of all clocks in this half-column region
        std::vector<std::pair<IndexType, RealType>> ckIdxProbPairs;
        ckIdxProbPairs.reserve(_nlPtr->numClockNets());
        for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
        {
            ckIdxProbPairs.emplace_back(ckIdx, 1.0 - ckIdxToInvProbArray.at(ckIdx).at(hcId));
        }

        // Sort clock nets by their probability from high to low
        std::sort(ckIdxProbPairs.begin(), ckIdxProbPairs.end(),
                  [](const std::pair<IndexType, RealType> &l, const std::pair<IndexType, RealType> &r){ return l.second > r.second; });
        IndexType nck = std::min(_nlPtr->numClockNets(), Parameters::archHalfColumnRegionClockCapacity);
        openparfAssert(ckIdxToInvProbArray.size() <= nck || ckIdxProbPairs.at(nck).second < 1.0);
        for (IndexType i = 0; i < nck; ++i)
        {
            const auto &p = ckIdxProbPairs.at(i);
            if (p.second > 0.0)
            {
                _clockAvailHC.ckIdxToAvailArray().at(p.first).at(hcId) = 1;
            }
        }
    }
}

/// Export the clock tree solution to a file
template<typename T>
void ClockNetworkPlanner<T>::exportClockTreeToFile(const std::string &fileName) const
{

}

template<typename T>
void ClockNetworkPlanner<T>::transferSolutionToTorchTensor( int32_t* nodeToCr, uint8_t* clkAvailCR)  {
    for (IndexType x = 0; x < _db->numCrX(); x++)
    for (IndexType y = 0; y < _db->numCrY(); y++) {
        auto& nodeArr = _crToNodeIdArray.at(x, y);
        for(IndexType nodeId: nodeArr) {
            // A node can only be assigned to one clock region
            openparfAssert(nodeToCr[nodeId] == InvalidIndex<int32_t>::value);
            nodeToCr[nodeId] = x * _db->numCrY() + y;
        }
    }


     for (IndexType ckIdx = 0; ckIdx < _nlPtr->numClockNets(); ++ckIdx)
    {
        for (IndexType x = 0; x < _db->numCrX(); x++)
        for (IndexType y = 0; y < _db->numCrY(); y++) {
            IndexType idx =  ckIdx * _db->numCrX() * _db->numCrY() +  x * _db->numCrY() + y;
            clkAvailCR[idx] = _clockAvailCR.isAvail(ckIdx, x,y);
        }
        // TODO: assert _db num is same as _clockAvailCR inner data
    }
}


template class ClockNetworkPlanner<float>;
template class ClockNetworkPlanner<double>;

}
OPENPARF_END_NAMESPACE
