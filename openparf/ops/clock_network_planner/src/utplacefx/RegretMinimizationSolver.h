#ifndef __REGRETMINIMIZATIONSOLVER_H__
#define __REGRETMINIMIZATIONSOLVER_H__

#include <vector>
#include <boost/heap/fibonacci_heap.hpp>

#include "util/namespace.h"

//#include "global/global.h"
#include "ops/clock_network_planner/src/utplacefx/Types.h"
#include "ops/clock_network_planner/src/utplacefx/GeneralAssignmentProblem.h"

OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {
/// Class to solve general assignment problem using regret minimization heuristic
class RegretMinimizationSolver
{
private:
    /// Class to store the (item index, item regret) pair
    struct ItemRegret
    {
        explicit ItemRegret(IndexType i, RealType r) : itemIdx(i), regret(r) {}

        IndexType itemIdx = INDEX_TYPE_MAX;
        RealType  regret  = 0;
    };

    // Comparator of ItemRegret objects for max heap
    struct ItemRegretComp
    {
        bool operator()(const ItemRegret &l, const ItemRegret &r) const { return l.regret < r.regret; }
    };

    using ItemRegretPQ = boost::heap::fibonacci_heap<ItemRegret, boost::heap::compare<ItemRegretComp>>;

private:
    /// Item class used in this solver
    struct Item
    {
        RealType regret() const
        {
            if (it1 == costArray.end() || it0 == costArray.end())
            {
                return REAL_TYPE_MAX;
            }
            return it1->cost - it0->cost;
        }

        std::vector<GeneralAssignmentProblem::BinCost>            costArray;            // This array should be sorted by cost from low to high
        std::vector<GeneralAssignmentProblem::BinCost>::iterator  it0;                     // Points to the current best assignment in costArray
        std::vector<GeneralAssignmentProblem::BinCost>::iterator  it1;                     // Points to the current second best assignment in costArray
        ItemRegretPQ::handle_type                                 pqHandle;                // The handle points to the item in PQ
        RealType                                                  dem    = 0;              // The demand of this item
        IndexType                                                 binIdx = INDEX_TYPE_MAX; // The target bin index of this item
    };

    /// Bin class used in this solver
    struct Bin
    {
        std::vector<IndexType>  itemIdxArray0;   // Indices of items that regard this bin as their best bin, sorted by item demands from low to high
        std::vector<IndexType>  itemIdxArray1;   // Indices of items that regard this bin as their second best bin, sorted by item demands from low to high
        RealType                cap  = 0;        // The remaining capacity of this bin
    };

public:
    explicit RegretMinimizationSolver(const GeneralAssignmentProblem &gap)
        : _gap(gap)
    {}

    void       init();
    bool       run();
    RealType   cost() const { return _cost; }
    IndexType  targetBinIndexOfItemIndex(IndexType itemIdx) const { return _itemArray.at(itemIdx).binIdx; }
    RealType   binRemainCap(IndexType binIdx) const               { return _binArray.at(binIdx).cap; }

private:
    void insertItemIndexToItemIndexArray(IndexType itemIdx, std::vector<IndexType> &itemIdxArray);

private:
    const GeneralAssignmentProblem  & _gap;       // The general assignment problem to be solved
    std::vector<Item>                 _itemArray;
    std::vector<Bin>                  _binArray;
    RealType                          _cost = 0;  // The solution cost
};

/// Insert item index into a item index array sorted by item demand from low to high
inline void RegretMinimizationSolver::insertItemIndexToItemIndexArray(IndexType itemIdx, std::vector<IndexType> &itemIdxArray)
{
    const auto &item = _itemArray.at(itemIdx);
    auto rit = itemIdxArray.rbegin();
    while (rit != itemIdxArray.rend() && _itemArray.at(*rit).dem > item.dem)
    {
        ++rit;
    }
    itemIdxArray.insert(rit.base(), itemIdx);
}
}
OPENPARF_END_NAMESPACE

#endif // __REGRETMINIMIZATIONSOLVER_H__
