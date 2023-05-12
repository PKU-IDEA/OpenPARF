#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : stable_div.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.29.2020
# Last Modified Date: 04.29.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>

import torch
from torch import nn
from torch.autograd import Function

from . import stable_div_cpp
from openparf import configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import stable_div_cuda


class StableDivFunction(Function):
    """
    @brief Given x and y, perform x / y.
    For entries where y is 0, x / y = 0;
    """

    @staticmethod
    def forward(ctx, x, y):
        if x.is_cuda:
            func = stable_div_cuda.forward
        else:
            func = stable_div_cpp.forward
        output = func(x, y)
        ctx.x = x
        ctx.y = y
        return output

    @staticmethod
    def backward(ctx, grad):
        # grad_stable_div is not contiguous
        if grad.is_cuda:
            func = stable_div_cuda.backward
        else:
            func = stable_div_cpp.backward
        output = func(grad.contiguous(), ctx.x, ctx.y)
        return output[0], output[1]


class StableDiv(nn.Module):
    """
    @brief Given x and y, perform x / y.
    For entries where y is 0, x / y = 0;
    """

    def forward(self, x, y):
        """
        @brief API
        @param x dividend
        @param y divisor
        """
        return StableDivFunction.apply(x, y)


class StableZeroDiv(nn.Module):
    """
    @brief A stable but slow way to handle zero-division
    """

    def forward(self, x, y):
        assert isinstance(y, torch.Tensor)
        out = torch.zeros_like(y)
        y_nonzero_ids = torch.nonzero(y)
        out[y_nonzero_ids] = 1.0 / y[y_nonzero_ids]
        out.mul_(x)
        return out
