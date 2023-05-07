##
# @file   pin_pos.py
# @author Xiaohan Gao
# @date   Sep 2019
# @brief  Compute pin pos
#

import math
import torch
from torch import nn
from torch.autograd import Function

from . import pin_pos_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import pin_pos_cuda

import pdb


class PinPosFunction(Function):
    """
    @brief Given cell locations, compute pin locations.
    """
    @staticmethod
    def forward(ctx, pos, pin_offsets, inst_pins, inst_pins_start,
                pin2inst_map):
        if pos.is_cuda:
            func = pin_pos_cuda.forward
        else:
            func = pin_pos_cpp.forward
        output = func(pos, pin_offsets, pin2inst_map)
        ctx.pos = pos
        ctx.inst_pins = inst_pins
        ctx.inst_pins_start = inst_pins_start
        return output

    @staticmethod
    def backward(ctx, grad_pin_pos):
        # grad_pin_pos is not contiguous
        if grad_pin_pos.is_cuda:
            func = pin_pos_cuda.backward
        else:
            func = pin_pos_cpp.backward
        # this output only contains physical instances
        output = func(grad_pin_pos.contiguous(), ctx.pos, ctx.inst_pins,
                      ctx.inst_pins_start)
        #print("wirelength grad")
        #print(output.view([-1, 2])[1078476:1079242].norm(dim=0))
        return output, None, None, None, None


class PinPos(nn.Module):
    """
    @brief Given cell locations, compute pin locations.
    Different from torch.index_add which computes x[index[i]] += t[i],
    the forward function compute x[i] += t[index[i]]
    """
    def __init__(self, pin_offsets, inst_pins, inst_pins_start, pin2inst_map):
        """
        @brief initialization
        @param pin_offset pin offset in x or y direction, only computes one direction
        @param algorithm segment|inst-by-inst
        """
        super(PinPos, self).__init__()
        self.pin_offsets = pin_offsets
        self.inst_pins = inst_pins
        self.inst_pins_start = inst_pins_start
        self.pin2inst_map = pin2inst_map

    def forward(self, pos):
        """
        @brief API
        @param pos cell locations of (x, y) pairs.
        """
        return PinPosFunction.apply(pos, self.pin_offsets, self.inst_pins,
                                    self.inst_pins_start, self.pin2inst_map)
