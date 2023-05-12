#ifndef OPENPARF_CLOCK_DATABASE_H
#define OPENPARF_CLOCK_DATABASE_H

#include "container/index_wrapper.hpp"
#include "database/clock_region.h"
#include "database/half_column_region.h"
#include "database/layout_map.hpp"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace database {

/**
 * Interface that provides clock-related database.
 */
template<typename IndexType, typename = typename std::enable_if<std::is_integral<IndexType>::value>::type>
class ClockDataBase {
public:
    using ClockRegionRefType      = container::IndexWrapper<ClockRegion, IndexType>;
    using HalfColumnRegionRefType = container::IndexWrapper<HalfColumnRegion, IndexType>;

    /// @brief Const getter of 2D map for clock regions
    virtual const ClockRegionMap &clock_region_map() = 0;

    /// &brief Const getter of  collection of half column regions
    virtual const std::vector<HalfColumnRegion> &half_column_regions() = 0;

    /**
     * @brief Const getter for `layout to clock region map`
     * @detail Support we want to get a reference to the corresponding `ClockRegion` instance
     * at location (x,y), use `layout2clock_region_map().at(x,y).Ref()`
     */
    virtual const LayoutMap2D<ClockRegionRefType> &layout2clock_region_map() = 0;

    /**
     * @brief Const getter for `layout to half column region map`
     * @detail Support we want to get a reference to the corresponding `ClockRegion` instance
     * at location (x,y), use `layout2_half_column_region_map().at(x,y).Ref()`
     */
    virtual const LayoutMap2D<HalfColumnRegionRefType> &layout2_half_column_region_map() = 0;

    /// @brief Number of clock nets
    virtual const IndexType &numClockNets() = 0;

    /**
     * @brief Net to clock index mapping.
     * @return a (#num_nets, ) vector `res`. If for `net i`, the value of `res[i]` isn't
     * InvalidIndex<IndexType>::value, then `net i` is a clock net ,and the clock id of `net i` is `res[i]`.
     * The clock id is within [0, num_clock_nets).
     */
    virtual const std::vector<IndexType> &getClockNetIndex() = 0;

    /**
     * @brief Instances to clocks mapping.
     * Note that One instance may be connected to multiple clock nets.
     * @return
     */
    virtual const std::vector<std::vector<IndexType>> &instToClocks() = 0;

    virtual ~ClockDataBase() = default;
};

template<typename IndexType, typename = typename std::enable_if<std::is_integral<IndexType>::value>::type>
class ClockRegionAssignment {
public:
    using Vector1DRefType = std::vector<IndexType>;

    virtual const Vector1DRefType &crToNodeIdArray(IndexType cr_ix, IndexType cr_iy) = 0;

    virtual bool ckIdxToCrIsAvail(IndexType ckIdx, IndexType crX, IndexType crY) = 0;
    virtual bool ckIdxToCrIsAvail(IndexType ckIdx, IndexType crId)               = 0;

    virtual bool ckIdxToHcIsAvail(IndexType ckIdx, IndexType hcId) = 0;

    virtual ~ClockRegionAssignment() = default;
};

}   // namespace database
OPENPARF_END_NAMESPACE
#endif   //OPENPARF_CLOCK_DATABASE_H
