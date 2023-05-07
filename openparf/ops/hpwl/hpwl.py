##
# @file   hpwl.py
# @author Yibo Lin
# @date   Apr 2020
# @brief  Compute half-perimeter wirelength
#

import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import pdb

from . import hpwl_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import hpwl_cuda


class HPWLFunction(Function):
    """compute half-perimeter wirelength.
    @param pos pin location (x array, y array), not cell location
    @param flat_netpin flat netpin map, length of #pins
    @param netpin_start starting index in netpin map for each net,
                        length of #nets+1, the last entry is #pins
    @param net_weights weight of nets
    @param net_mask a boolean mask containing whether a net should be computed
    @param pin2net_map pin2net map, second set of options
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, net_weights, net_mask):
        if pos.is_cuda:
            func = hpwl_cuda.forward
        else:
            func = hpwl_cpp.forward
        output = func(pos, flat_netpin, netpin_start, net_weights, net_mask)
        return output.sum(dim=0)


class HPWL(nn.Module):
    """
    @brief Compute half-perimeter wirelength using the net-by-net algorithm.
    Guarantee determinism in parallel.
    """
    def __init__(self, flat_netpin, netpin_start, net_weights, net_mask):
        """
        @brief initialization
        @param flat_netpin flat netpin map, length of #pins
        @param netpin_start starting index in netpin map for each net,
                            length of #nets+1, the last entry is #pins
        @param pin2net_map pin2net map
        @param net_weights weight of nets
        @param net_mask whether to compute wirelength,
                        1 means to compute, 0 means to ignore
        """
        super(HPWL, self).__init__()
        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.net_weights = net_weights
        self.net_mask = net_mask

    def forward(self, pos):
        return HPWLFunction.apply(pos, self.flat_netpin, self.netpin_start,
                                    self.net_weights, self.net_mask)

    def __call__(self, pos):
        return self.forward(pos)
