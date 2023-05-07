#ifndef __GENERALASSIGNMENTPROBLEM_H__
#define __GENERALASSIGNMENTPROBLEM_H__

#include <vector>
//#include "global/global.h"
#include "util/util.h"
#include "ops/clock_network_planner/src/utplacefx/Types.h"

OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {
/// This class defines a general assignment problem
class GeneralAssignmentProblem
{
public:
    /// Class for items in general assignment problems
    struct Item
    {
        explicit Item(RealType d) : dem(d) {}
        RealType dem = 0; // The demand of this item
    };

    /// Class for bins in general assignment problems
    struct Bin
    {
        explicit Bin(RealType c) : cap(c) {}
        RealType cap = 0; // The capacity of this bin
    };

    // Class to store the (bin index, item to bin cost) pair
    struct BinCost
    {
        BinCost(IndexType i, RealType c) : binIdx(i), cost(c) {}
        bool operator<(const BinCost &rhs) const { return cost < rhs.cost; }

        IndexType binIdx = INDEX_TYPE_MAX;
        RealType  cost   = REAL_TYPE_MAX;
    };

public:
    explicit GeneralAssignmentProblem() = default;
    explicit GeneralAssignmentProblem(IndexType numItems, IndexType numBins)
    {
        _itemArray.reserve(numItems);
        _binArray.reserve(numBins);
        _costMap.reserve(numItems);
    }

    IndexType                          numItems() const                                             { return _itemArray.size(); }
    IndexType                          numBins() const                                              { return _binArray.size(); }
    const Item &                       item(IndexType i) const                                      { return _itemArray.at(i); }
    Item &                             item(IndexType i)                                            { return _itemArray.at(i); }
    const Bin &                        bin(IndexType i) const                                       { return _binArray.at(i); }
    Bin &                              bin(IndexType i)                                             { return _binArray.at(i); }
    const std::vector<Item> &          itemArray() const                                            { return _itemArray; }
    const std::vector<Bin> &           binArray() const                                             { return _binArray; }
    const std::vector<BinCost> &       costArray(IndexType itemIdx) const                           { return _costMap.at(itemIdx); }

    void                               clear()                                                      { _itemArray.clear(); _binArray.clear(); _costMap.clear(); }
    void                               addItem(RealType dem)                                        { _itemArray.emplace_back(dem); _costMap.emplace_back(); }
    void                               addBin(RealType cap)                                         { _binArray.emplace_back(cap); }
    void                               addCost(IndexType itemIdx, IndexType binIdx, RealType cost)  { _costMap.at(itemIdx).emplace_back(binIdx, cost); }

private:
    std::vector<Item>                  _itemArray;
    std::vector<Bin>                   _binArray;
    std::vector<std::vector<BinCost>>  _costMap;
};
}
OPENPARF_END_NAMESPACE

#endif // __GENERALASSIGNMENTPROBLEM_H__
