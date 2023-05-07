#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : ism_dp.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 07.14.2020
# Last Modified Date: 07.14.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>

import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import pdb

from . import ism_dp_cpp


class ISMDetailedPlaceParam(object):
    """
    @brief This class defines parameters needed by DLSolver
    """

    def __init__(self, params):
        self.routeUtilBinW = 1.0   # The bin width for routing estimation
        self.routeUtilBinH = 1.0   # The bin height for routing estimation
        # The bin dimension for routing estimation
        self.routeUtilBinDimX = np.iinfo(np.int32).max
        # The bin dimension for routing estimation
        self.routeUtilBinDimY = np.iinfo(np.int32).max
        self.unitHoriRouteCap = 0.0   # The unit horizontal routing capacity
        self.unitVertRouteCap = 0.0   # The unit vertical routing capacity

        self.pinUtilBinW = 1.0   # The bin width for pin utilization estimation
        self.pinUtilBinH = 1.0   # The bin height for pin utilization estimation
        # The bin dimension for pin utilization estimation
        self.pinUtilBinDimX = np.iinfo(np.int32).max
        # The bin dimension for pin utilization estimation
        self.pinUtilBinDimY = np.iinfo(np.int32).max
        self.unitPinCap = 0.0   # The unit pin capacity
        self.ffPinWeight = 3.0   # The weight of FF pins for pin density optimization
        # Stretch pin to a ratio of the pin utilization bin to smooth the density map
        self.pinStretchRatio = 1.414

        # We fix white spaces at places where routing utilization is higher than this
        self.routeUtilToFixWhiteSpace = 0.8
        # We fix white spaces at places where pin utilization is higher than this
        self.pinUtilToFixWhiteSpace = 0.9

        self.xWirelenWt = params.wirelength_weights[0]
        self.yWirelenWt = params.wirelength_weights[1]

        self.verbose = 1


class ISMDetailedPlace:
    """ 
    @brief Indepented Set Matching based detailed placement algorithm
    """

    def __init__(self,
                 placedb,
                 params,
                 honor_clock_constraints=False
                 ):
        """
        @brief initialization 
        """
        super(ISMDetailedPlace, self).__init__()
        self.placedb = placedb

        self.param = ISMDetailedPlaceParam(params=params)
        self.param.routeUtilBinDimX = placedb.siteMapDim().x()
        self.param.routeUtilBinDimY = placedb.siteMapDim().y()
        self.param.routeUtilBinW = 1.0
        self.param.routeUtilBinH = 1.0
        self.param.pinUtilBinDimX = placedb.siteMapDim().x()
        self.param.pinUtilBinDimY = placedb.siteMapDim().y()
        self.param.pinUtilBinW = 1.0
        self.param.pinUtilBinH = 1.0
        self.param.honorClockConstraints = honor_clock_constraints
        self.clock_available_clock_region = None
        self.fixed_mask = None

    def reset_clock_available_clock_region(self, clock_available_clock_region: torch.Tensor):
        assert clock_available_clock_region.dtype == torch.uint8
        self.clock_available_clock_region = clock_available_clock_region

    def reset_half_column_available_clock_region(self, half_column_available_clock_region: torch.Tensor):
        assert half_column_available_clock_region.dtype == torch.uint8
        self.half_column_available_clock_region = half_column_available_clock_region

    def reset_honor_clock_constraints(self, honor_clock_constraints: bool):
        self.param.honorClockConstraints = honor_clock_constraints

    def forward(self, pos):
        if pos.is_cuda:
            local_pos = pos.cpu()
        else:
            local_pos = pos
        if self.fixed_mask is None:
            num_insts = local_pos.shape[0]
            self.fixed_mask = torch.zeros(num_insts, dtype=torch.uint8, device="cpu", requires_grad=False)
        assert self.fixed_mask.shape[0] == local_pos.shape[0]
        if self.param.honorClockConstraints:
            assert isinstance(self.clock_available_clock_region, torch.Tensor)
            assert self.clock_available_clock_region.dtype == torch.uint8
            assert isinstance(
                self.half_column_available_clock_region, torch.Tensor)
            assert self.half_column_available_clock_region.dtype == torch.uint8
            return ism_dp_cpp.forward(
                self.placedb,
                self.param,
                self.clock_available_clock_region,
                self.half_column_available_clock_region,
                self.fixed_mask,
                local_pos).to(pos.device)
        else:
            dummy = torch.zeros((1))
            return ism_dp_cpp.forward(
                self.placedb,
                self.param,
                dummy,
                dummy,
                self.fixed_mask,
                local_pos).to(pos.device)

    def __call__(self, pos):
        return self.forward(pos)
