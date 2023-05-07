#include "database/clock_availability.h"
#include "database/placedb.h"
#include "geometry/box.hpp"
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

using CoordinateType = database::PlaceDB::CoordinateType;

template<typename scalar_t>
static OPENPARF_NOINLINE void
CrCkCounterKernel(at::TensorAccessor<scalar_t, 2>                inst_pos_a,
                  const std::vector<std::vector<int32_t>> &      inst_cks,
                  LayoutXy2GridIndexFunctorType<CoordinateType> &XyToCrIndex,
                  at::TensorAccessor<int32_t, 2> cr_ck_counts_a, int32_t num_insts,
                  database::PlaceDB &placedb, int32_t num_threads) {
    using openparf::geometry::Box;
    int32_t cks_num = placedb.numClockNets();
    std::vector<Box<int32_t>> ck_avail_cr_boxes(cks_num);
    for (auto &box : ck_avail_cr_boxes) box.reset();
    int32_t y_crs_num  = placedb.numCrY();
    int     chunk_size = std::max(int32_t(num_insts / num_threads / 16), 1);
    for (int32_t i = 0; i < num_insts; i++) {
        if (placedb.isInstClockSource(i)) {
            // ignore instance source
            continue;
        }
        scalar_t     center_x = inst_pos_a[i][0];
        scalar_t     center_y = inst_pos_a[i][1];
        int32_t      cr_id    = XyToCrIndex(center_x, center_y);
        int32_t      x_cr_id  = cr_id / y_crs_num;
        int32_t      y_cr_id  = cr_id % y_crs_num;
        Box<int32_t> cr_box(x_cr_id, y_cr_id, x_cr_id, y_cr_id);
        for (auto &ck : inst_cks[i]) {ck_avail_cr_boxes[ck].join(cr_box); }
    }
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, chunk_size)
    for (int32_t ck_id = 0; ck_id < cks_num; ck_id++) {
        auto &cr_box = ck_avail_cr_boxes.at(ck_id);
        for (int32_t x = cr_box.xl(); x <= cr_box.xh(); x++) {
            for (int32_t y = cr_box.yl(); y <= cr_box.yh(); y++) {
                int32_t cr_id                = x * y_crs_num + y;
                cr_ck_counts_a[cr_id][ck_id] = 1;
            }
        }
    }
}

at::Tensor CrCkCounter(at::Tensor inst_pos, const std::vector<std::vector<int32_t>> &inst_cks,
                       int32_t num_crs, int32_t num_cks, database::PlaceDB &placedb) {
    CHECK_FLAT_CPU(inst_pos);
    CHECK_EVEN(inst_pos);

    int32_t num_insts   = inst_pos.numel() >> 1;
    int32_t num_threads = at::get_num_threads();

    auto options = torch::TensorOptions()
                           .dtype(torch::kInt32)
                           .layout(torch::kStrided)
                           .device(torch::kCPU)
                           .requires_grad(false);

    namespace arg = std::placeholders;

    at::Tensor cr_ck_counts   = torch::zeros({num_crs, num_cks}, options);
    auto       cr_ck_counts_a = cr_ck_counts.accessor<int32_t, 2>();
    LayoutXy2GridIndexFunctorType<CoordinateType> xy_to_cr_func =
            [ObjectPtr = &placedb](auto &&PH1, auto &&PH2) {
                return ObjectPtr->XyToCrIndex(std::forward<decltype(PH1)>(PH1),
                                              std::forward<decltype(PH2)>(PH2));
            };

    OPENPARF_DISPATCH_FLOATING_TYPES(inst_pos, "Cr Ck counter", [&] {
        auto inst_pos_a = inst_pos.accessor<scalar_t, 2>();
        CrCkCounterKernel<scalar_t>(inst_pos_a, inst_cks, xy_to_cr_func, cr_ck_counts_a, num_insts,
                                    placedb, num_threads);
    });

    return cr_ck_counts.sum(/*dim=*/1);
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::CrCkCounter, "CR-CK Counter");
}
