#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : wasll.py
# Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
# Date              : 11.17.2023
# Last Modified Date: 11.17.2023
# Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>

import time
import torch
from torch import nn
from torch.autograd import Function
import logging

from . import wasll_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import wasll_cuda

logger = logging.getLogger(__name__)


class WASLLFunction(Function):
    """
    @brief Autograd function for computing the weighted average super long line (SLL).
    It supports both forward and backward computations.
    """

    @staticmethod
    def forward(
        ctx, pos, flat_netpin, netpin_start, pin2net_map, net_weights, net_mask,
        pin_mask, inv_gamma, num_slrX, num_slrY
    ):
        """
        Forward pass of WASLL computation.

        :param ctx: Context object for saving state.
        :param pos: Tensor of pin locations (x and y arrays), not cell locations.
        :param flat_netpin: Flattened netpin mapping.
        :param netpin_start: Starting index in the netpin map for each net.
        :param pin2net_map: Mapping of pins to nets.
        :param net_weights: Weights for each net.
        :param net_mask: Mask to indicate which nets to include in the computation.
        :param pin_mask: Mask to indicate which pins to compute gradients for.
        :param inv_gamma: Inverse of the gamma parameter; higher values approximate SLL more closely.
        :param num_slrX: Number of super logic regions along the X-axis.
        :param num_slrY: Number of super logic regions along the Y-axis.
        :return: Tensor representing the weighted average SLL.
        """
        if pos.is_cuda:
            func = wasll_cuda.forward
        else:
            func = wasll_cpp.forward
        output = func(
            pos, flat_netpin, netpin_start, pin2net_map, net_weights, net_mask,
            inv_gamma, num_slrX, num_slrY
        )
        ctx.pin2net_map = pin2net_map
        ctx.flat_netpin = flat_netpin
        ctx.netpin_start = netpin_start
        ctx.net_weights = net_weights
        ctx.net_mask = net_mask
        ctx.pin_mask = pin_mask
        ctx.inv_gamma = inv_gamma
        ctx.grad_intermediate = output[1]
        ctx.pos = pos
        if pos.is_cuda:
            torch.cuda.synchronize()
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        """
        Backward pass of WASLL computation.

        :param ctx: Context object with saved state from the forward pass.
        :param grad_pos: Gradient of the output with respect to the input positions.
        :return: Gradient of the input with respect to each parameter.
        """
        if grad_pos.is_cuda:
            func = wasll_cuda.backward
        else:
            func = wasll_cpp.backward
        output = func(
            grad_pos, ctx.pos, ctx.grad_intermediate, ctx.flat_netpin,
            ctx.netpin_start, ctx.pin2net_map, ctx.net_weights, ctx.net_mask,
            ctx.inv_gamma
        )
        output.view([-1, 2]).masked_fill_(ctx.pin_mask.view([-1, 1]), 0.0)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        return output, None, None, None, None, None, None, None, None, None


class WASLL(nn.Module):
    """
    @brief Compute weighted average super long line.
    It utilizes a merged algorithm and can operate in both forward and backward modes.
    """

    def __init__(
        self, flat_netpin, netpin_start, pin2net_map, net_weights, net_mask,
        pin_mask, gamma, num_slrX, num_slrY
    ):
        """
        Initialization for the WASLL module.

        @param flat_netpin: Flattened netpin map.
        @param netpin_start: Starting index in the netpin map for each net.
        @param pin2net_map: Mapping of pins to nets.
        @param net_weights: Weights for each net.
        @param net_mask: Mask indicating which nets to compute wirelength for.
        @param pin_mask: Mask indicating which pins to compute gradients for.
        @param gamma: Gamma parameter; lower values approximate HPWL more closely.
        @param num_slrX: Number of super logic regions along the X-axis.
        @param num_slrY: Number of super logic regions along the Y-axis.
        """
        super(WASLL, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.netpin_values = None
        self.pin2net_map = pin2net_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.pin_mask = pin_mask
        self.gamma = gamma
        self.num_slrX = num_slrX
        self.num_slrY = num_slrY

    def forward(self, pos):
        return WASLLFunction.apply(
            pos,
            self.flat_netpin,
            self.netpin_start,
            self.pin2net_map,
            self.net_weights,
            self.net_mask,
            self.pin_mask,
            1.0 / self.gamma,  # do not store inv_gamma as gamma is changing
            self.num_slrX,
            self.num_slrY
        )
