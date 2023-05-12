#include "util/torch.h"
#include "util/util.h"
#include <iostream>

OPENPARF_BEGIN_NAMESPACE


template<typename scalar_t,
         typename = typename std::enable_if<std::is_scalar<scalar_t>::value>::type>
static OPENPARF_NOINLINE void
FenceRegionCheckerKernel(const at::TensorAccessor<scalar_t, 2> &  fence_region_boxes_a,
                         const at::TensorAccessor<scalar_t, 2> &  half_inst_sizes_a,
                         const std::vector<std::vector<int32_t>> &inst_avail_crs,
                         const at::TensorAccessor<scalar_t, 2> &  inst_pos_a,
                         at::TensorAccessor<scalar_t, 1> &displacement_a, int32_t num_insts,
                         int32_t num_threads) {
    auto dist_func = [](scalar_t l, scalar_t r, scalar_t x) {
        if (x < l) return (l - x) * (l - x);
        if (r < x) return (r - x) * (r - x);
        return scalar_t(0);
    };
    int32_t chunk_size = std::max(int32_t(num_insts / num_threads / 16), 1);
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int32_t i = 0; i < num_insts; ++i) {
        const std::vector<int32_t> &avail_crs = inst_avail_crs[i];
        openparfAssert(!avail_crs.empty());
        scalar_t min_dist = std::numeric_limits<scalar_t>::max();

        scalar_t center_x = inst_pos_a[i][0];
        scalar_t center_y = inst_pos_a[i][1];
        for (auto &region_index : avail_crs) {
            scalar_t box_xl = fence_region_boxes_a[region_index][0];
            scalar_t box_yl = fence_region_boxes_a[region_index][1];
            scalar_t box_xh = fence_region_boxes_a[region_index][2];
            scalar_t box_yh = fence_region_boxes_a[region_index][3];

            scalar_t dist = std::sqrt(dist_func(box_xl, box_xh, center_x) +
                                      dist_func(box_yl, box_yh, center_y));
            min_dist      = std::min(min_dist, dist);
        }
        displacement_a[i] = min_dist;
    }
}

at::Tensor FenceRegionChecker(at::Tensor fence_region_boxes, at::Tensor half_inst_sizes,
                              std::vector<std::vector<int32_t>> &inst_avail_crs,
                              at::Tensor                         inst_pos) {
    CHECK_FLAT_CPU(fence_region_boxes);
    CHECK_DIVISIBLE(fence_region_boxes, 4);
    CHECK_FLAT_CPU(half_inst_sizes);
    CHECK_EVEN(half_inst_sizes);
    CHECK_FLAT_CPU(inst_pos);
    CHECK_EVEN(inst_pos);
    CHECK_CONTIGUOUS(inst_pos);

    int32_t num_insts   = inst_pos.numel() >> 1;
    int32_t num_threads = at::get_num_threads();

    at::Tensor displacement = torch::zeros({num_insts}, inst_pos.options());

    OPENPARF_DISPATCH_FLOATING_TYPES(inst_pos, "Fence Region Checker", [&] {
        auto       displacement_a = displacement.accessor<scalar_t, 1>();
        auto fence_region_boxes_a = fence_region_boxes.accessor<scalar_t, 2>();
        auto half_inst_sizes_a    = half_inst_sizes.accessor<scalar_t, 2>();
        auto inst_pos_a           = inst_pos.accessor<scalar_t, 2>();
        FenceRegionCheckerKernel<scalar_t>(fence_region_boxes_a, half_inst_sizes_a, inst_avail_crs,
                                           inst_pos_a, displacement_a, num_insts, num_threads);
    });
    return displacement;
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::FenceRegionChecker, "Fence Region Checker");
}
