#include "util/namespace.h"

#include "ops/clock_network_planner/src/utplacefx/RegretMinimizationSolver.h"

OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {
/// Initialize the solver
void RegretMinimizationSolver::init()
{
    // Initialize bins
    _binArray.clear();
    _binArray.resize(_gap.numBins());
    for (IndexType binIdx = 0; binIdx < _gap.numBins(); ++binIdx)
    {
        auto &bin = _binArray.at(binIdx);
        bin.cap = _gap.bin(binIdx).cap;
        bin.itemIdxArray0.clear();
        bin.itemIdxArray1.clear();
    }

    // Initialize items
    _itemArray.clear();
    _itemArray.resize(_gap.numItems());
    for (IndexType itemIdx = 0; itemIdx < _gap.numItems(); ++itemIdx)
    {
        auto &item = _itemArray.at(itemIdx);
        item.dem = _gap.item(itemIdx).dem;
        item.costArray = _gap.costArray(itemIdx);

        // Sort costArray and set it0 and it1
        std::sort(item.costArray.begin(), item.costArray.end());
        if (item.costArray.size() > 0)
        {
            item.it0 = item.costArray.begin();
            _binArray.at(item.it0->binIdx).itemIdxArray0.push_back(itemIdx);
        }
        else
        {
            item.it0 = item.costArray.end();
        }
        if (item.costArray.size() > 1)
        {
            item.it1 = std::next(item.costArray.begin());
            _binArray.at(item.it1->binIdx).itemIdxArray1.push_back(itemIdx);
        }
        else
        {
            item.it1 = item.costArray.end();
        }
    }

    // Sort bin.itemIdxArray0/1 by item demands from low to high
    auto comp = [&](IndexType l, IndexType r){ return _itemArray.at(l).dem < _itemArray.at(r).dem; };
    for (auto &bin : _binArray)
    {
        std::sort(bin.itemIdxArray0.begin(), bin.itemIdxArray0.end(), comp);
        std::sort(bin.itemIdxArray1.begin(), bin.itemIdxArray1.end(), comp);
    }
}

/// Run regret minimization
/// Return true if a legal assignment is found, otherwise, return false
bool RegretMinimizationSolver::run()
{
    _cost = 0;

    // Initialize PQ sorted by items' regrets, pq.top() has the largest regret
    ItemRegretPQ pq;
    for (IndexType itemIdx = 0; itemIdx < _itemArray.size(); ++itemIdx)
    {
        auto &item = _itemArray.at(itemIdx);
        item.pqHandle = pq.emplace(itemIdx, item.regret());
    }

    // Iteratively fetch PQ top and assign items
    while (! pq.empty())
    {
        auto top = pq.top();
        pq.pop();

        auto &topItem = _itemArray.at(top.itemIdx);
        if (topItem.it0 == topItem.costArray.end())
        {
            // Fail to find a legal assignment for this top item
            return false;
        }

        // Assign this top item to its current best bin
        topItem.binIdx = topItem.it0->binIdx;
        _cost += topItem.it0->cost;

        // Update bin remaining capacity
        IndexType tgtBinIdx = topItem.binIdx;
        auto &tgtBin = _binArray.at(tgtBinIdx);
        tgtBin.cap -= topItem.dem;

        openparfAssert(tgtBin.cap >= 0);

        // After adding this top item into this bin, items that are previously legal for this bin might become illegal
        // We need to remove them and find their new best/second best bins
        // Note that bin.itemIdxArray0/1 are sorted by items' demand from low to high,
        // so we check the legality from back to the front

        // Update items that regard tgtBin as the best bin
        while (! tgtBin.itemIdxArray0.empty())
        {
            IndexType itemIdx = tgtBin.itemIdxArray0.back();
            auto &item = _itemArray.at(itemIdx);
            if (item.dem <= tgtBin.cap)
            {
                // All remaining items in bin.itemIdxArray0 has demand no greater than tgtBin.cap
                break;
            }

            tgtBin.itemIdxArray0.pop_back();
            if (item.binIdx != INDEX_TYPE_MAX)
            {
                // This item has been assigned
                continue;
            }

            // This item has its demand larger than the remaining capacity in tgtBin
            // So this item->tgtBin assignment is no more legal
            // The current second best bin for this item becomes the best one, and we need to find the new second best bin for this item

            // Set the previous second best bin as the best bin
            item.it0 = item.it1;
            if (item.it1 != item.costArray.end())
            {
                insertItemIndexToItemIndexArray(itemIdx, _binArray.at(item.it0->binIdx).itemIdxArray0);

                // Find the new second best bin
                while (++item.it1 != item.costArray.end())
                {
                    auto &bin = _binArray.at(item.it1->binIdx);
                    if (item.dem <= bin.cap)
                    {
                        insertItemIndexToItemIndexArray(itemIdx, bin.itemIdxArray1);
                        break;
                    }
                }
            }

            // Update regret of the item
            (*item.pqHandle).regret = item.regret();
            pq.update(item.pqHandle);
        }

        // Update items that regard tgtBin as the second best bin
        while (! tgtBin.itemIdxArray1.empty())
        {
            IndexType itemIdx = tgtBin.itemIdxArray1.back();
            auto &item = _itemArray.at(itemIdx);
            if (item.dem <= tgtBin.cap)
            {
                // All remaining items in bin.itemIdxArray1 has demand no greater than tgtBin.cap
                break;
            }

            tgtBin.itemIdxArray1.pop_back();
            if (item.binIdx != INDEX_TYPE_MAX || item.it1->binIdx != tgtBinIdx)
            {
                // This item has been assigned OR
                // This item does not regard tgtBin as the second best bin any more
                // (this could happen when the best bin of this item becomes invalid and use the second best as the best)
                continue;
            }

            // This item has its demand larger than the remaining capacity in tgtBin
            // So this item->tgtBin assignment is no more legal
            // We need to find the new best bin for this item
            if (item.it1 != item.costArray.end())
            {
                while (++item.it1 != item.costArray.end())
                {
                    auto &bin = _binArray.at(item.it1->binIdx);
                    if (item.dem <= bin.cap)
                    {
                        insertItemIndexToItemIndexArray(itemIdx, bin.itemIdxArray1);
                        break;
                    }
                }
            }

            // Update regret of the item
            (*item.pqHandle).regret = item.regret();
            pq.update(item.pqHandle);
        }
    }
    return true;
}

}
OPENPARF_END_NAMESPACE
