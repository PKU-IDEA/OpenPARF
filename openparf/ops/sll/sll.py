#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : sll.py
# Author            : Runzhe Tao <rztao@my.swjtu.edu.cn>
# Date              : 11.17.2023
# Last Modified Date: 11.17.2023
# Last Modified By  : Runzhe Tao <rztao@my.swjtu.edu.cn>

import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import logging

from . import sll_cpp
from openparf import configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import sll_cuda

logger = logging.getLogger(__name__)

import pdb


class SLLFunction(Function):
    """
    @brief Autograd function for computing super long line (SLL) counts.
    """

    @staticmethod
    def forward(
        ctx,
        pos,
        flat_netpin,
        netpin_start,
        net_weights,
        net_mask,
        sll_counts_table,
        xl,
        yl,
        slr_width,
        slr_height,
        num_slrX,
        num_slrY,
    ):
        """
        @param ctx Context object for saving state.
        @param pos Tensor representing pin positions, structured as an array 
                of x coordinates and an array of y coordinates.
        @param flat_netpin Flattened net-pin map, length equals total number of pins.
        @param netpin_start Starting index in the net-pin map for each net, 
                            with length equal to the total number of nets plus one.
                            The last entry indicates the total number of pins.
        @param net_weights Weights assigned to each net.
        @param net_mask Boolean mask indicating which nets should be included in the SLL calculation.
                        A value of 1 indicates inclusion, while 0 indicates exclusion.
        @param xl Lower bound coordinate in the x-dimension.
        @param yl Lower bound coordinate in the y-dimension.
        @param slr_width Width of SLR.
        @param slr_height Height of SLR.
        @param num_slrX Number of columns in the SLR topology.
        @param num_slrY Number of rows in the SLR topology.
        """
        func = sll_cuda.forward if pos.is_cuda else sll_cpp.forward
        output = func(
            pos,
            flat_netpin,
            netpin_start,
            net_weights,
            net_mask,
            sll_counts_table,
            xl,
            yl,
            slr_width,
            slr_height,
            num_slrX,
            num_slrY,
        )
        return output.sum()


class SLL(nn.Module):
    """
    @brief Compute super long line (SLL) counts. This module uses a deterministic, 
    net-by-net algorithm to ensure accuracy in parallel computations.
    """

    def __init__(self, flat_netpin, netpin_start, net_weights, net_mask, xl, yl,
                 slr_width, slr_height, num_slrX, num_slrY):
        """
        @param flat_netpin Flattened net-pin map, length equals total number of pins.
        @param netpin_start Starting index in the net-pin map for each net, 
                            with length equal to the total number of nets plus one.
                            The last entry indicates the total number of pins.
        @param net_weights Weights assigned to each net.
        @param net_mask Boolean mask to indicate nets to be included in SLL computation.
                        A value of 1 indicates inclusion, while 0 indicates exclusion.
        @param xl, yl: Coordinates of the lower-left corner of the layout area.
        @param slr_width, slr_height: Dimensions of the super logic regions.
        @param num_slrX, num_slrY: Number of super logic regions along X and Y axes.
        """
        super(SLL, self).__init__()
        self.flat_netpin = flat_netpin
        self.netpin_start = netpin_start
        self.net_weights = net_weights
        self.net_mask = net_mask
        self.xl = xl
        self.yl = yl
        self.slr_width = slr_width
        self.slr_height = slr_height
        self.num_slrX = num_slrX
        self.num_slrY = num_slrY

        # Initialize a lookup table for super long line (SLL) counts.
        # This table is represented as a fixed-size array,
        # where each index corresponds to a unique binary representation of an SLR configuration.
        # The value at each index represents the corresponding SLL counts assuming a
        # distribution of net pins as per that specific SLR configuration.
        #
        # The lookup table simplifies the process of determining SLL counts based on different SLR
        # configurations. For example, in a 1x4 SLR configuration, the binary representation '0b0011'
        # (index 3 in the table) would correspond to a specific SLL count value that indicates the
        # number of SLLs when a net is distributed across the SLRs as if following this configuration.
        #
        # Two specific configurations are defined:
        # - A 1x4 SLR configuration, represented by a binary sequence from '0b0000' to '0b1111',
        #   where each binary value maps to an SLL count.
        # - A 2x2 SLR configuration, similarly represented with a binary sequence mapping.
        #
        # If the SLR configuration does not match any predefined setups (1x4 or 2x2), the lookup
        # table is set to None. An assertion check ensures that the lookup table is properly defined
        # for the given SLR topology.
        # It is important for users to define their own SLL counts table here, particularly
        # tailored to the unique specifications of their Multi-die FPGA architecture. This customization
        # ensures accurate mapping and functionality in line with specific architecture designs.
        if num_slrX == 1 and num_slrY == 4:
            # Binary SLR maps -> SLL counts
            # 0b0000 -> 0, 0b0001 -> 0, 0b0010 -> 0, 0b0011 -> 1
            # 0b0100 -> 0, 0b0101 -> 2, 0b0110 -> 1, 0b0111 -> 2
            # 0b1000 -> 0, 0b1001 -> 3, 0b1010 -> 2, 0b1011 -> 3
            # 0b1100 -> 1, 0b1101 -> 3, 0b1110 -> 2, 0b1111 -> 3
            self.sll_counts_table = torch.tensor(
                [0, 0, 0, 1, 0, 2, 1, 2, 0, 3, 2, 3, 1, 3, 2, 3],
                dtype=torch.int32)
        elif num_slrX == 2 and num_slrY == 2:
            # Binary SLR maps -> SLL counts
            # 0b0000 -> 0, 0b0001 -> 0, 0b0010 -> 0, 0b0011 -> 1
            # 0b0100 -> 0, 0b0101 -> 1, 0b0110 -> 2, 0b0111 -> 2
            # 0b1000 -> 0, 0b1001 -> 2, 0b1010 -> 1, 0b1011 -> 2
            # 0b1100 -> 1, 0b1101 -> 2, 0b1110 -> 2, 0b1111 -> 3
            self.sll_counts_table = torch.tensor(
                [0, 0, 0, 1, 0, 1, 2, 2, 0, 2, 1, 2, 1, 2, 2, 3],
                dtype=torch.int32)
        else:
            self.sll_counts_table = None
            # load numpy data, see `scripts/compute_sll_counts_table.py`
            # self.sll_counts_table = torch.from_numpy(
            #     np.load(
            #         os.path.join(
            #             os.path.dirname(os.path.abspath(__file__)),
            #             "sll_counts_table.npy"))).to(dtype=torch.int32)

            try:
                assert self.sll_counts_table is not None
            except AssertionError:
                logger.debug(
                    f"The SLL count lookup table is not defined for the SLR topology {self.num_slrX} x {self.num_slrY}."
                )

    def forward(self, pos):
        if pos.is_cuda and not self.sll_counts_table.is_cuda:
            self.sll_counts_table = self.sll_counts_table.to(pos.device)

        return SLLFunction.apply(
            pos,
            self.flat_netpin,
            self.netpin_start,
            self.net_weights,
            self.net_mask,
            self.sll_counts_table,
            self.xl,
            self.yl,
            self.slr_width,
            self.slr_height,
            self.num_slrX,
            self.num_slrY,
        )

    def __call__(self, pos):
        return self.forward(pos)
