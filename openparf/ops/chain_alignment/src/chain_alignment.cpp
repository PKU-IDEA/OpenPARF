// project headers
#include "util/torch.h"
#include "util/util.h"

// local headers
#include "chain_alignment_kernel.h"

OPENPARF_BEGIN_NAMESPACE

namespace chain_alignment {

torch::Tensor buildOffset(torch::Tensor chain2inst_data, torch::Tensor chain2inst_index_start, torch::Tensor inst_sizes) {
  CHECK_FLAT_CPU(chain2inst_data);
  CHECK_CONTIGUOUS(chain2inst_data);
  CHECK_EQ(chain2inst_data.dtype(), torch::kInt32);
  CHECK_FLAT_CPU(chain2inst_index_start);
  CHECK_CONTIGUOUS(chain2inst_index_start);
  CHECK_EQ(chain2inst_index_start.dtype(), torch::kInt32);
  CHECK_FLAT_CPU(inst_sizes);
  CHECK_CONTIGUOUS(inst_sizes);
  CHECK_EVEN(inst_sizes);

  int32_t    num_chains            = chain2inst_index_start.numel() - 1;
  int32_t    num_insts             = chain2inst_data.numel();
  torch::Tensor offset_from_grav_core = torch::zeros({num_insts, 2},
                                                  torch::TensorOptions()
                                                          .dtype(inst_sizes.dtype())
                                                          .layout(torch::kStrided)
                                                          .device(torch::kCPU)
                                                          .requires_grad(false));
  auto       data                  = chain2inst_data.accessor<int32_t, 1>();
  auto       index_start           = chain2inst_index_start.accessor<int32_t, 1>();
  OPENPARF_DISPATCH_FLOATING_TYPES(inst_sizes, "build chain offset", [&] {
    auto sizes  = inst_sizes.accessor<scalar_t, 2>();
    auto offset = offset_from_grav_core.accessor<scalar_t, 2>();
    for (int32_t i = 0; i < num_chains; i++) {
      int32_t  st         = index_start[i];
      int32_t  en         = index_start[i + 1];
      scalar_t height_sum = 0;
      for (int j = st; j < en; j++) {
        int32_t  inst_id = data[j];
        scalar_t height  = sizes[inst_id][1];
        height_sum += height;
      }
      scalar_t core_height = height_sum * 0.5;
      height_sum           = 0;
      for (int j = st; j < en; j++) {
        int32_t  inst_id = data[j];
        scalar_t height  = sizes[inst_id][1];
        offset[j][0]     = 0;
        offset[j][1]     = (height_sum + height * 0.5) - core_height;
        height_sum += height;
      }
    }
  });
  return offset_from_grav_core;
}

void forward(torch::Tensor pos,
             torch::Tensor chain2inst_data,
             torch::Tensor chain2inst_index_start,
             torch::Tensor offset_from_grav_core,
             int32_t    num_threads) {
  CHECK_FLAT_CPU(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CPU(chain2inst_data);
  CHECK_CONTIGUOUS(chain2inst_data);
  CHECK_EQ(chain2inst_data.dtype(), torch::kInt32);
  CHECK_FLAT_CPU(chain2inst_index_start);
  CHECK_CONTIGUOUS(chain2inst_index_start);
  CHECK_EQ(chain2inst_index_start.dtype(), torch::kInt32);
  CHECK_FLAT_CPU(offset_from_grav_core);
  CHECK_CONTIGUOUS(offset_from_grav_core);
  CHECK_EQ(pos.dtype(), offset_from_grav_core.dtype());

  int32_t num_chains = chain2inst_index_start.numel() - 1;

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "chain alignment forward", [&] {
    dispatchedForward<scalar_t>(OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
                                OPENPARF_TENSOR_DATA_PTR(offset_from_grav_core, scalar_t),
                                OPENPARF_TENSOR_DATA_PTR(chain2inst_data, int32_t),
                                OPENPARF_TENSOR_DATA_PTR(chain2inst_index_start, int32_t),
                                num_chains,
                                num_threads);
  });
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void dispatchedForward<T>(T * pos,                                                                          \
                                     T * offset,                                                                       \
                                     int32_t * index_data,                                                             \
                                     int32_t * index_start,                                                            \
                                     int32_t num_chains,                                                               \
                                     int32_t num_threads);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace chain_alignment
OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("buildOffset",
        &OPENPARF_NAMESPACE::chain_alignment::buildOffset,
        "compute the offset from the chain gravity core.");
  m.def("forward", &OPENPARF_NAMESPACE::chain_alignment::forward, "chain alignment forward(CPU)");
}
