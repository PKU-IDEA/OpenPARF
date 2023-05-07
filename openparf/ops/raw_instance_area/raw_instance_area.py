#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : raw_instance_area.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 07.25.2020
# Last Modified Date: 08.11.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>

from torch import nn
import math

from openparf import configure
from . import raw_instance_area_cpp


# if configure.compile_configurations['CUDA_FOUND'] == "TRUE":
#     from . import raw_instance_area_cuda


class ComputeRawInstanceArea(nn.Module):
    def __init__(self,
                 xl,
                 yl,
                 xh,
                 yh,
                 bin_size_x,
                 bin_size_y):
        """ Operator that computes the raw instance area from utilization map.

        :param xl: minimum x-coordinates of the layout
        :param yl: minimum y-coordinates of the layout
        :param xh: maximum x-coordinates of the layout
        :param yh: maximum y-coordinates of the layout
        :param bin_size_x: the width of bin in the x-axis direction
        :param bin_size_y: the height of bin in the y-axis direction
        """
        super(ComputeRawInstanceArea, self).__init__()
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.bin_size_x = bin_size_x
        self.bin_size_y = bin_size_y
        self.num_bins_x = math.ceil((xh - xl) / bin_size_x)
        self.num_bins_y = math.ceil((yh - yl) / bin_size_y)

    def forward(self, inst_pos, inst_half_sizes, movable_range, utilization_map):
        """

        :param inst_pos: tensor of all cell central positions, shape of (#cells, 2)
        :param inst_half_sizes: size pair (width/2, height/2) of all cell sizes, shape of (#cells, 2)
        :param movable_range: the index pair [lower bound, higher bound) of the movable cells
        :param utilization_map: utilization map, shape of (|num_bins_x|, |num_bins_y|)
        :return: adjusted movable cell sizes, one-dimensional tensor with length #movable cells
        """
        functor = raw_instance_area_cpp
        new_movable_inst_area = functor.forward(
            inst_pos.cpu(),
            inst_half_sizes.cpu(),
            movable_range,
            utilization_map.cpu(),
            self.xl,
            self.yl,
            self.xh,
            self.yh,
            self.num_bins_x,
            self.num_bins_y,
            self.bin_size_x,
            self.bin_size_y,
        ).to(inst_pos.device)
        assert new_movable_inst_area.shape == (movable_range[1] - movable_range[0],)
        return new_movable_inst_area
