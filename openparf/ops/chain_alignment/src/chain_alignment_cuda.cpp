// project headers
#include "util/torch.h"
#include "util/util.cuh"


OPENPARF_BEGIN_NAMESPACE

namespace chain_alignment {

template<class T>
void ChainAlignmentKernelLauncher(T *pos, T *offset, int32_t *index_data, int32_t *index_start, int32_t num_chains);

void            forward(at::Tensor pos,
                        at::Tensor chain2inst_data,
                        at::Tensor chain2inst_index_start,
                        at::Tensor offset_from_grav_core,
                        int32_t    num_threads) {
  CHECK_FLAT_CUDA(pos);
  CHECK_CONTIGUOUS(pos);
  CHECK_FLAT_CUDA(chain2inst_data);
  CHECK_CONTIGUOUS(chain2inst_data);
  CHECK_EQ(chain2inst_data.dtype(), torch::kInt32);
  CHECK_FLAT_CUDA(chain2inst_index_start);
  CHECK_CONTIGUOUS(chain2inst_index_start);
  CHECK_EQ(chain2inst_index_start.dtype(), torch::kInt32);
  CHECK_FLAT_CUDA(offset_from_grav_core);
  CHECK_CONTIGUOUS(offset_from_grav_core);
  CHECK_EQ(pos.dtype(), offset_from_grav_core.dtype());

  int32_t num_chains = chain2inst_index_start.numel() - 1;

  OPENPARF_DISPATCH_FLOATING_TYPES(pos, "chain alignment forward", [&] {
    int32_t thread_count = 256;
    int32_t block_count  = ceilDiv(num_chains, thread_count);
    ChainAlignmentKernelLauncher<scalar_t>(
            OPENPARF_TENSOR_DATA_PTR(pos, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(offset_from_grav_core, scalar_t),
            OPENPARF_TENSOR_DATA_PTR(chain2inst_data, int32_t),
            OPENPARF_TENSOR_DATA_PTR(chain2inst_index_start, int32_t),
            num_chains);
  });
}


}   // namespace chain_alignment

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &OPENPARF_NAMESPACE::chain_alignment::forward, "chain alignment forward(CUDA)");
}
