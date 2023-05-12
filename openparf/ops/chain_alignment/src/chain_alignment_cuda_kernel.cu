// C system library headers
#include <cuda_runtime.h>

// project headers
#include "util/util.cuh"

OPENPARF_BEGIN_NAMESPACE

namespace chain_alignment {

template<class T>
__global__ void ChainAlignmentKernel(T *pos, T *offset, int32_t *index_data, int32_t *index_start, int32_t num_chains) {
  int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= num_chains) return;
  int32_t st            = index_start[i];
  int32_t en            = index_start[i + 1];
  T inv_num_insts = 1.0 / (en - st);
  T       xx_sum        = 0;
  T       yy_sum        = 0;
  T       grav_xx, grav_yy;

  for (int j = st; j < en; j++) {
    int32_t inst_id = index_data[j];
    xx_sum += pos[inst_id << 1];
    yy_sum += pos[inst_id << 1 | 1];
  }
  grav_xx = xx_sum * inv_num_insts;
  grav_yy = yy_sum * inv_num_insts;
  for (int j = st; j < en; j++) {
    int32_t inst_id       = index_data[j];
    pos[inst_id << 1]     = grav_xx + offset[j << 1];
    pos[inst_id << 1 | 1] = grav_yy + offset[j << 1 | 1];
  }
}

template<class T>
void ChainAlignmentKernelLauncher(T *pos, T *offset, int32_t *index_data, int32_t *index_start, int32_t num_chains) {
  if (num_chains) {
    int32_t thread_count = 256;
    int32_t block_count  = ceilDiv(num_chains, thread_count);
    ChainAlignmentKernel<<<(uint32_t) block_count, {(uint32_t) thread_count, 1u, 1u}>>>(pos,
        offset,
        index_data,
        index_start,
        num_chains);
  }
}

#define REGISTER_KERNEL_LAUNCHER(T)                                                                                    \
  template void ChainAlignmentKernelLauncher<T>(T * pos,                                                               \
                                                T * offset,                                                            \
                                                int32_t * index_data,                                                  \
                                                int32_t * index_start,                                                 \
                                                int32_t num_chains);

REGISTER_KERNEL_LAUNCHER(float)
REGISTER_KERNEL_LAUNCHER(double)

#undef REGISTER_KERNEL_LAUNCHER

}   // namespace chain_alignment
OPENPARF_END_NAMESPACE
