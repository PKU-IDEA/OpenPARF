#include "container/index_wrapper.hpp"
#include "gtest/gtest.h"
#include <vector>

OPENPARF_BEGIN_NAMESPACE

namespace unittest {
class IndexWrapperUnittest : public ::testing::Test {
public:
    void testInt32() const {
        using IndexType = std::size_t;
        using ElementType = int32_t;
        using ClockRegionRefType                   = container::IndexWrapper<ElementType, IndexType>;
        IndexType                width             = 10;
        IndexType                height            = 10;
        IndexType                cr_width          = 5;
        IndexType                cr_height         = 5;
        IndexType                cr_num_x          = width / cr_width;
        IndexType                cr_num_y          = width / cr_height;
        IndexType                num_clock_regions = cr_num_x * cr_num_y;
        std::vector<ElementType> arr(num_clock_regions);
        for (IndexType ix = 0; ix < cr_num_x; ix++) {
            for (IndexType iy = 0; iy < cr_num_y; iy++) {
                arr[ix * cr_num_y + iy] = ix * cr_num_y + iy;
            }
        }
        std::vector<ClockRegionRefType> refs(num_clock_regions);
        for (IndexType i = 0; i < num_clock_regions; i++) {
            refs[i].setRef(arr[i]);
        }
    }
};
TEST_F(IndexWrapperUnittest, ClockRegion) { testInt32(); }
}   // namespace unittest
OPENPARF_END_NAMESPACE
