import torch
from torch import nn
import pdb

from openparf import configure
from . import resource_area_cpp


# Functors to be used at the CPP level
# if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
#    from . import resource_area_cpp
def compute_control_sets(
        inst_pins,
        inst_pins_start,
        pin2net_map,
        pin_signal_types,
        is_inst_ffs
):
    ff_ctrlsets, num_cksr, num_ce = resource_area_cpp.compute_control_sets(
        inst_pins,
        inst_pins_start,
        pin2net_map,
        is_inst_ffs,
        pin_signal_types
    )
    return ff_ctrlsets, num_cksr, num_ce


class ResourceArea(nn.Module):
    def __init__(self,
                 is_inst_luts,  # The type of luts
                 is_inst_ffs,
                 ff_ctrlsets,
                 num_cksr,
                 num_ce,
                 num_bins_x: int,
                 num_bins_y: int,
                 stddev_x,
                 stddev_y,
                 stddev_trunc,
                 slice_capacity: int,
                 gp_adjust_packing_rule: str):
        """
        :param is_inst_luts: a (#insts, ) tensor. If is_inst_luts[i] == 0, then i is not a lut. If it's greater than zero, then it denotes
        the type of lut for that instance
        :param is_inst_ffs: a (#insts, ) tensor. If if_inst_ffs[i] == 0, the i is not a FF. If_inst_ff[i]=1, the i is a FF.
        :param num_bins_x: the number of bin in the x-axis direction
        :param num_bins_y: the number of bin in the y-axis direction
        :param stddev_x: std. derivation of the Gaussian distribution in the X direction
        :param stddev_y: std. derivation of the Gaussian distribution in the Y direction
        :param stddev_trunc: Parameter for the calculation of demand map.
        :param slice_capacity: parameter based on fpga's target architecture.
        """
        super(ResourceArea, self).__init__()
        self.is_inst_luts = is_inst_luts
        self.is_inst_ffs = is_inst_ffs
        self.ff_ctrlsets = ff_ctrlsets
        self.ff_ctrlsets_cksr_size = num_cksr
        self.ff_ctrlsets_ce_size = num_ce
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.stddev_x = stddev_x
        self.stddev_y = stddev_y
        self.stddev_trunc = stddev_trunc
        self.slice_capacity = slice_capacity
        self.gp_adjust_packing_rule = gp_adjust_packing_rule

        assert torch.all(self.is_inst_luts != 1)  # LUT1 does not exist.
        assert torch.all(self.is_inst_ffs <= 6)  # We only have LUT2 to LUT6.
        assert torch.all(0 <= self.is_inst_ffs) and torch.all(
            self.is_inst_ffs <= 1)
        assert gp_adjust_packing_rule == "ultrascale" or gp_adjust_packing_rule == "xarch"

    def forward(self,
                inst_pos: torch.Tensor,
                ) -> torch.Tensor:
        """
        :param inst_pos: center of instances, array of (x, y) pairs
        :return shape of #(num_instances), with the resource area of each lut or ff, in inst_pos.dtype. Unrelated instances will take the value of zero.
        """
        local_pos = inst_pos.cpu() if inst_pos.is_cuda else inst_pos
        rv = resource_area_cpp.compute_resource_areas(local_pos,
                                                      self.is_inst_luts,
                                                      self.is_inst_ffs,
                                                      self.ff_ctrlsets,
                                                      self.ff_ctrlsets_cksr_size,
                                                      self.ff_ctrlsets_ce_size,
                                                      self.num_bins_x,
                                                      self.num_bins_y,
                                                      self.stddev_x,
                                                      self.stddev_y,
                                                      self.stddev_trunc,
                                                      self.slice_capacity,
                                                      self.gp_adjust_packing_rule)
        return rv.to(inst_pos.device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
