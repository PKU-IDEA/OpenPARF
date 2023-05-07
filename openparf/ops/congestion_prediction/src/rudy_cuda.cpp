/**
 * File              : rudy_cuda.cpp
 * Author            : Jing Mai <magic3007@pku.edu.cn>
 * Date              : 07.10.2020
 * Last Modified Date: 07.10.2020
 * Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
 */
#include "util/torch.h"
#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

template<typename T>
int32_t RudyCudaLauncher(T *pin_pos, int32_t *netpin_start, int32_t *flat_netpin, T *net_weights,
                         T bin_size_x, T bin_size_y, T xl, T yl, T xh, T yh, int32_t num_bins_x,
                         int32_t num_bins_y, int32_t num_nets, T *pinDirects,
                         T *horizontal_utilization_map, T *vertical_utilization_map,
                         T *pin_density_map);

void rudy_forward(at::Tensor pin_pos, at::Tensor netpin_start, at::Tensor flat_netpin,
                  at::Tensor net_weights, double bin_size_x, double bin_size_y, double xl,
                  double yl, double xh, double yh, int32_t num_bins_x, int32_t num_bins_y,
                  at::Tensor pinDirects, at::Tensor horizontal_utilization_map,
                  at::Tensor vertical_utilization_map, at::Tensor pin_density_map) {
    CHECK_FLAT_CUDA(pin_pos);
    CHECK_EVEN(pin_pos);
    CHECK_CONTIGUOUS(pin_pos);

    CHECK_FLAT_CUDA(netpin_start);
    CHECK_CONTIGUOUS(netpin_start);

    CHECK_FLAT_CUDA(flat_netpin);
    CHECK_CONTIGUOUS(flat_netpin);

    CHECK_FLAT_CUDA(net_weights);
    CHECK_CONTIGUOUS(net_weights);

    CHECK_FLAT_CUDA(horizontal_utilization_map);
    CHECK_CONTIGUOUS(horizontal_utilization_map);

    CHECK_FLAT_CUDA(vertical_utilization_map);
    CHECK_CONTIGUOUS(vertical_utilization_map);

    CHECK_FLAT_CUDA(pin_density_map);
    CHECK_CONTIGUOUS(pin_density_map);

    CHECK_FLAT_CUDA(pinDirects);
    CHECK_CONTIGUOUS(pinDirects);


    /**
     * |netpin_start| is similar to the IA array in CSR format, IA[i+1]-IA[i] is
     * the number of pins in each net, the length of IA is number of nets + 1
     */
    int num_nets = netpin_start.numel() - 1;

    OPENPARF_DISPATCH_FLOATING_TYPES(pin_pos, "RudyCudaLauncher", [&] {
        RudyCudaLauncher<scalar_t>(
                OPENPARF_TENSOR_DATA_PTR(pin_pos, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(netpin_start, int32_t),
                OPENPARF_TENSOR_DATA_PTR(flat_netpin, int32_t),
                net_weights.numel() > 0 ? OPENPARF_TENSOR_DATA_PTR(net_weights, scalar_t) : nullptr,
                bin_size_x, bin_size_y, xl, yl, xh, yh, num_bins_x, num_bins_y, num_nets,
                OPENPARF_TENSOR_DATA_PTR(pinDirects, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(horizontal_utilization_map, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(vertical_utilization_map, scalar_t),
                OPENPARF_TENSOR_DATA_PTR(pin_density_map, scalar_t));
    });
}

OPENPARF_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &OPENPARF_NAMESPACE::rudy_forward, "compute RUDY map (CUDA)");
}
