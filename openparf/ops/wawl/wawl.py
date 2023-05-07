##
# @file   wawl.py
# @author Yibo Lin
# @date   Apr 2020
# @brief  Compute weighted-average wirelength
#

import time
import torch
from torch import nn
from torch.autograd import Function
import logging
import pdb

from . import wawl_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import wawl_cuda

logger = logging.getLogger(__name__)


class WAWLFunction(Function):
    """
    @brief compute weighted average wirelength.
    """
    @staticmethod
    def forward(ctx, pos, flat_netpin, netpin_start, pin2net_map, net_weights,
                net_mask, pin_mask, inv_gamma):
        """
        @param pos pin location (x array, y array), not cell location
        @param pin2net_map pin2net map
        @param net_weights weight of nets
        @param net_mask whether to compute wirelength
        @param pin_mask whether compute gradient for a pin,
                        1 means to fill with zero, 0 means to compute
        @param inv_gamma 1/gamma, the larger, the closer to HPWL
        """
        tt = time.time()
        if pos.is_cuda:
            func = wawl_cuda.forward
        else:
            func = wawl_cpp.forward
        output = func(pos, flat_netpin, netpin_start, pin2net_map, net_weights,
                      net_mask, inv_gamma)
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
        logger.debug("wirelength forward %.3f ms" %
                     ((time.time() - tt) * 1000))
        return output[0]

    @staticmethod
    def backward(ctx, grad_pos):
        tt = time.time()
        if grad_pos.is_cuda:
            func = wawl_cuda.backward
        else:
            func = wawl_cpp.backward
        output = func(grad_pos, ctx.pos, ctx.grad_intermediate,
                      ctx.flat_netpin, ctx.netpin_start, ctx.pin2net_map,
                      ctx.net_weights, ctx.net_mask, ctx.inv_gamma)
        output.view([-1, 2]).masked_fill_(ctx.pin_mask.view([-1, 1]), 0.0)
        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        logger.debug("wirelength backward %.3f ms" %
                     ((time.time() - tt) * 1000))
        return output, None, None, None, None, None, None, None, None


class WAWL(nn.Module):
    """
    @brief Compute weighted average wirelength.
    Use the merged algorithm.
    """
    def __init__(self, flat_netpin, netpin_start, pin2net_map, net_weights,
                 net_mask, pin_mask, gamma):
        """
        @brief initialization
        @param flat_netpin flat netpin map, length of #pins
        @param netpin_start starting index in netpin map for each net,
                            length of #nets+1, the last entry is #pins
        @param pin2net_map pin2net map
        @param net_weights weight of nets
        @param net_mask whether to compute wirelength,
                        1 means to compute, 0 means to ignore
        @param pin_mask whether compute gradient for a pin,
                        1 means to fill with zero, 0 means to compute
        @param gamma the smaller, the closer to HPWL
        """
        super(WAWL, self).__init__()

        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.netpin_values = None
        self.pin2net_map = pin2net_map
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.pin_mask = pin_mask
        self.gamma = gamma

    def forward(self, pos):
        return WAWLFunction.apply(
            pos,
            self.flat_netpin,
            self.netpin_start,
            self.pin2net_map,
            self.net_weights,
            self.net_mask,
            self.pin_mask,
            1.0 / self.gamma  # do not store inv_gamma as gamma is changing
        )
