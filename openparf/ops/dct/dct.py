##
# @file   dct.py (originally dct2_fft2.py)
# @author Zixuan Jiang, Jiaqi Gu, modified by Yibo Lin
# @date   Jun 2018
# @brief  Implement 2d dct, 2d idct, idxst(idct(x)), idct(idxst(x)) based on 2d fft
#

import numpy as np
import torch
from torch.autograd import Function
from torch import nn

from .discrete_spectral_transform import getExactExpk as precomputeExpk

from . import dct2_fft2_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import dct2_fft2_cuda


class Dct2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.dct2(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.dct2(x, expkM, expkN, out, buf)
        return out


class Dct2(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(Dct2, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(
                -2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precomputeExpk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(
                -2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precomputeExpk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M,
                                   N // 2 + 1,
                                   2,
                                   dtype=x.dtype,
                                   device=x.device)

        return Dct2Function.apply(x, self.expkM, self.expkN, self.out,
                                  self.buf)


class Idct2Function(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.idct2(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.idct2(x, expkM, expkN, out, buf)
        return out


class Idct2(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(Idct2, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(
                -2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precomputeExpk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(
                -2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precomputeExpk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M,
                                   N // 2 + 1,
                                   2,
                                   dtype=x.dtype,
                                   device=x.device)

        return Idct2Function.apply(x, self.expkM, self.expkN, self.out,
                                   self.buf)


class IdctIdxstFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.idctIdxst(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.idctIdxst(x, expkM, expkN, out, buf)
        return out


class IdctIdxst(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IdctIdxst, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(
                -2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precomputeExpk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(
                -2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precomputeExpk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M,
                                   N // 2 + 1,
                                   2,
                                   dtype=x.dtype,
                                   device=x.device)

        return IdctIdxstFunction.apply(x, self.expkM, self.expkN, self.out,
                                       self.buf)


class IdxstIdctFunction(Function):
    @staticmethod
    def forward(ctx, x, expkM, expkN, out, buf):
        if x.is_cuda:
            dct2_fft2_cuda.idxstIdct(x, expkM, expkN, out, buf)
        else:
            dct2_fft2_cpp.idxstIdct(x, expkM, expkN, out, buf)
        return out


class IdxstIdct(nn.Module):
    def __init__(self, expkM=None, expkN=None):
        super(IdxstIdct, self).__init__()

        self.expkM = expkM
        self.expkN = expkN
        self.out = None
        self.buf = None

    def forward(self, x):
        M = x.size(-2)
        N = x.size(-1)
        if self.expkM is None or self.expkM.size(
                -2) != M or self.expkM.dtype != x.dtype:
            self.expkM = precomputeExpk(M, dtype=x.dtype, device=x.device)
        if self.expkN is None or self.expkN.size(
                -2) != N or self.expkN.dtype != x.dtype:
            self.expkN = precomputeExpk(N, dtype=x.dtype, device=x.device)
        if self.out is None:
            self.out = torch.empty(M, N, dtype=x.dtype, device=x.device)
            self.buf = torch.empty(M,
                                   N // 2 + 1,
                                   2,
                                   dtype=x.dtype,
                                   device=x.device)

        return IdxstIdctFunction.apply(x, self.expkM, self.expkN, self.out,
                                       self.buf)
