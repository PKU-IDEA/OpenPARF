import torch
from torch import nn

from openparf import configure
from . import pin_utilization_cpp
import math

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import pin_utilization_cuda


class PinUtilization(nn.Module):
    def __init__(self,
                 inst_pin_weights,
                 xl,
                 xh,
                 yl,
                 yh,
                 num_bins_x,
                 num_bins_y,
                 inst_range,
                 unit_pin_capacity,
                 pin_stretch_ratio,
                 deterministic_flag):
        """

        :param inst_pin_weights: pin weights for each instances, shape of (#instance,).
        :param xl: minimum x-coordinates of the layout
        :param xh: maximum x-coordinates of the layout
        :param yl: minimum y-coordinates of the layout
        :param yh: maximum y-coordinates of the layout
        :param num_bins_x: number of bins in the x-axis direction
        :param num_bins_y: number of bins in the y-axis direction
        :param inst_range: index pair [lower bound, higher bound) of associated cells
        :param unit_pin_capacity: number of pins per unit area
        :param pin_stretch_ratio: stretch each pin to a ratio of the pin utilization bin
        """
        super(PinUtilization, self).__init__()

        self.inst_pin_weights = inst_pin_weights
        self.xl = xl
        self.xh = xh
        self.yl = yl
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        self.inst_range = inst_range

        self.unit_pin_capacity = unit_pin_capacity
        self.pin_stretch_ratio = pin_stretch_ratio
        self.deterministic_flag = deterministic_flag

    def forward(self,
                inst_sizes: torch.Tensor,
                inst_pos: torch.Tensor):
        """

        :param inst_sizes: pair (width, height) of cell sizes, shape of (#instance, 2)
        :param inst_pos: center of instances, array of (x, y) pairs
        """

        # Stretch each pin to a ratio of the pin utilization bin to make the pin density map more smoother
        stretch_inst_sizes = inst_sizes.clone()
        stretch_inst_sizes[:, 0].clamp_(min=self.bin_size_x * self.pin_stretch_ratio)
        stretch_inst_sizes[:, 1].clamp_(min=self.bin_size_y * self.pin_stretch_ratio)

        func = pin_utilization_cuda.forward if inst_pos.is_cuda else pin_utilization_cpp.forward
        output = func(inst_pos,
                      stretch_inst_sizes,
                      self.inst_pin_weights,
                      self.xl,
                      self.yl,
                      self.xh,
                      self.yh,
                      self.bin_size_x,
                      self.bin_size_y,
                      self.num_bins_x,
                      self.num_bins_y,
                      self.inst_range,
                      self.deterministic_flag)
        # convert demand to utilization in each bin
        output.mul_(1 / (self.bin_size_x * self.bin_size_y * self.unit_pin_capacity))
        return output
