#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : rudy.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 07.10.2020
# Last Modified Date: 07.10.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>

import torch
from torch import nn

from openparf import configure
from . import rudy_cpp

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import rudy_cuda


class Rudy(nn.Module):
    def __init__(self,
                 netpin_start,
                 flat_netpin,
                 net_weights,
                 xl,
                 xh,
                 yl,
                 yh,
                 num_bins_x, num_bins_y,
                 unit_horizontal_capacity,
                 unit_vertical_capacity,
                 deterministic_flag,
                 initial_horizontal_utilization_map=None,
                 initial_vertical_utilization_map=None):
        """ Constructor of RUDY/RISA operator.

        :param netpin_start: starting index in netpin map for each net, length of #nets+1, the last entry is #pins
        :param flat_netpin: flat netpin map, length of #pins
        :param net_weights: weight of nets, length of #nets
        :param xl: minimum x-coordinates of the layout
        :param xh: maximum x-coordinates of the layout
        :param yl: minimum y-coordinates of the layout
        :param yh: maximum y-coordinates of the layout
        :param num_bins_x: number of bins in the x-axis direction
        :param num_bins_y: number of bins in the y-axis direction
        :param unit_horizontal_capacity: the number of horizontal routing tracks per unit distance
        :param unit_vertical_capacity: the number of vertical routing tracks per unit distance
        :param initial_horizontal_utilization_map: initial horizontal rudy map, length of num_bins_x * num_bins_y
        :param initial_vertical_utilization_map initial vertical rudy map, length of num_bins_x * num_bins_y
        """
        super(Rudy, self).__init__()
        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.net_weights = net_weights
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        self.unit_horizontal_capacity = unit_horizontal_capacity
        self.unit_vertical_capacity = unit_vertical_capacity
        self.determistic_flag = deterministic_flag
        self.initial_horizontal_utilization_map = initial_horizontal_utilization_map
        self.initial_vertical_utilization_map = initial_vertical_utilization_map

    def forward(self, pin_pos):
        """ Forward function that calculates the routing congestion map

        :param pin_pos: tensor of pin position, length of 2 * #pins, in the form of xyxyxy...
        :return: rudy map, length of num_bins_x * num_bins_y
        """
        horizontal_utilization_map = torch.zeros((self.num_bins_x, self.num_bins_y),
                                                 dtype=pin_pos.dtype,
                                                 device=pin_pos.device)
        vertical_utilization_map = torch.zeros_like(horizontal_utilization_map)
        foo = rudy_cuda.forward if pin_pos.is_cuda else rudy_cpp.forward
        foo(pin_pos,
            self.netpin_start,
            self.flat_netpin,
            self.net_weights,
            self.bin_size_x,
            self.bin_size_y,
            self.xl,
            self.yl,
            self.xh,
            self.yh,
            self.num_bins_x,
            self.num_bins_y,
            self.determistic_flag,
            horizontal_utilization_map,
            vertical_utilization_map)
        # Convert demand to utilization in each bin
        bin_area = self.bin_size_x * self.bin_size_y
        horizontal_utilization_map.mul_(1.0 / (bin_area * self.unit_horizontal_capacity))
        vertical_utilization_map.mul_(1.0 / (bin_area * self.unit_vertical_capacity))
        if self.initial_horizontal_utilization_map is not None:
            horizontal_utilization_map.add_(self.initial_horizontal_utilization_map)
        if self.initial_vertical_utilization_map is not None:
            vertical_utilization_map.add_(self.initial_vertical_utilization_map)

        route_utilization_map = torch.max(horizontal_utilization_map.abs(), vertical_utilization_map.abs())

        # Routing Utilization Overflow
        return route_utilization_map, horizontal_utilization_map, vertical_utilization_map
