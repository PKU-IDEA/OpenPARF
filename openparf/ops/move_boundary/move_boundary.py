##
# @file   move_boundary.py
# @author Yibo Lin
# @date   Apr 2020
#

import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import pdb

from . import move_boundary_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import move_boundary_cuda


class MoveBoundaryFunction(Function):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """
    @staticmethod
    def forward(pos, inst_sizes, xl, yl, xh, yh, movable_range, filler_range):
        if pos.is_cuda:
            func = move_boundary_cuda.forward
        else:
            func = move_boundary_cpp.forward
        output = func(pos, inst_sizes, xl, yl, xh, yh, movable_range,
                      filler_range)
        return output


class MoveBoundary(object):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """
    def __init__(self, inst_sizes, xl, yl, xh, yh, movable_range,
                 filler_range):
        super(MoveBoundary, self).__init__()
        self.inst_sizes = inst_sizes
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.movable_range = movable_range
        self.filler_range = filler_range

    def forward(self, pos):
        return MoveBoundaryFunction.forward(pos=pos,
                                            inst_sizes=self.inst_sizes,
                                            xl=self.xl,
                                            yl=self.yl,
                                            xh=self.xh,
                                            yh=self.yh,
                                            movable_range=self.movable_range,
                                            filler_range=self.filler_range)

    def __call__(self, pos):
        return self.forward(pos)
