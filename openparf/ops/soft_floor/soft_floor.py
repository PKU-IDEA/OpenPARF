#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : soft_floor.py
# Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
# Date              : 11.17.2023
# Last Modified Date: 11.17.2023
# Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>

import math
import torch
from torch import nn
from torch.autograd import Function

from . import soft_floor_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import soft_floor_cuda


class SoftFloorFunction(Function):
    """
    @brief A PyTorch autograd function for applying a 'soft floor' operation to pin locations.
    This operation adjusts pin locations based on the provided Soft Floor parameters.
    """

    @staticmethod
    def forward(
        ctx, pos, xl, yl, slr_width, slr_height, num_slrX, num_slrY, soft_floor_gamma
    ):
        """
        Forward pass of the Soft Floor function.
        
        @param ctx: Context object for saving state.
        @param pos: Tensor of pin positions.
        @param xl, yl: Coordinates of the lower-left corner.
        @param slr_width, slr_height: Dimensions of the super logic regions.
        @param num_slrX, num_slrY: Number of super logic regions along X and Y axes.
        @param soft_floor_gamma: Gamma parameter for the soft floor calculation.
        @return: Adjusted positions after applying soft floor operation.
        """
        if pos.is_cuda:
            func = soft_floor_cuda.forward
        else:
            func = soft_floor_cpp.forward
        output = func(
            pos, xl, yl, slr_width, slr_height, num_slrX, num_slrY,
            soft_floor_gamma
        )
        ctx.grad_intermediate = output[1]
        return output[0]

    @staticmethod
    def backward(ctx, grad_pin_pos):
        """
        Backward pass of the Soft Floor function.
        
        @param ctx: Context object with saved state from the forward pass.
        @param grad_pin_pos: Gradient of the output with respect to the input positions.
        @return: Gradient of the input with respect to each parameter.
        """
        if grad_pin_pos.is_cuda:
            func = soft_floor_cuda.backward
        else:
            func = soft_floor_cpp.backward
        output = func(grad_pin_pos.contiguous(), ctx.grad_intermediate)
        return output, None, None, None, None, None, None, None


class SoftFloor(nn.Module):
    """
    @brief Implements the Soft Floor function, as described in the LEAPS paper,
    Section IV-3. It softens the position of pins during placement optimization.
    """

    def __init__(
        self, xl, yl, slr_width, slr_height, num_slrX, num_slrY,
        soft_floor_gamma
    ):
        """
        Initializes the Soft Floor module with the given parameters.
        
        @param xl, yl: Coordinates of the lower-left corner of the layout area.
        @param slr_width, slr_height: Dimensions of the super logic regions.
        @param num_slrX, num_slrY: Number of super logic regions along X and Y axes.
        @param soft_floor_gamma: Gamma parameter for the soft floor calculation.
        """
        super(SoftFloor, self).__init__()
        self.xl = xl
        self.yl = yl
        self.slr_width = slr_width
        self.slr_height = slr_height
        self.num_slrX = num_slrX
        self.num_slrY = num_slrY
        self.soft_floor_gamma = soft_floor_gamma

    def forward(self, pos):
        return SoftFloorFunction.apply(
            pos, 
            self.xl, 
            self.yl, 
            self.slr_width, 
            self.slr_height,
            self.num_slrX, 
            self.num_slrY, 
            self.soft_floor_gamma
        )

    def __call__(self, pos):
        return self.forward(pos)
