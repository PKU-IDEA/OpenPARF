#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : direct_lg.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 05.19.2020
# Last Modified Date: 10.24.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
from typing import Tuple
import pdb

import numpy as np
import torch

from . import direct_lg_cpp


class DirectLegalizeParam(object):
    """
    @brief This class defines parameters needed by DLSolver
    """

    def __init__(self, params):
        # We ignore any nets larger than this value for net sharing score computation
        self.netShareScoreMaxNetDegree = 16
        # We ignore any nets larger than this value for wirelength score computation
        self.wirelenScoreMaxNetDegree = 100
        # The max distance in th pre-clustering phase
        self.preclusteringMaxDist = 4.0
        # The initial site neighbor distance
        self.nbrDistBeg = 1.0
        # The maximum site neighbor distance
        self.nbrDistEnd = 0.0
        # The distance step of adjacent neighbor groups
        self.nbrDistIncr = 1.0
        # The candidate PQ size
        self.candPQSize = 10
        # Minimum number of stable iterations before a top candidate can be committed
        self.minStableIter = 3
        # The minimum number of active instances neighbors for each site
        self.minNeighbors = 10
        # The coefficient for the number of external net count in the candidate score computation
        self.extNetCountWt = 0.3
        # The coefficient for the wirelength improvement in the candidate score computation
        self.wirelenImprovWt = 0.1
        # The extra radius value to search after the first legal location in greedy legalization
        self.greedyExpansion = 5
        # The extra radius value to search after the first legal location in rip-up legalization
        self.ripupExpansion = 1
        # Flow weights must be integer, we up scale them to get better accuracy
        self.slotAssignFlowWeightScale = 1e3
        # The flow weight increase step size for max-weight matching based LUT pairing
        self.slotAssignFlowWeightIncr = 0.5
        # The weight for x-directed wirelength
        self.xWirelenWt = params.wirelength_weights[0]
        # The weight for y-directed wirelength
        self.yWirelenWt = params.wirelength_weights[1]

        # For message printing
        # 0: quiet
        # 1: basic messages
        # 2: extra messages
        self.verbose = 1

        self.CLB_capacity = None
        self.BLE_capacity = None
        self.omp_dynamic_chunk_size = 64

        # Clock Region Attribute
        self.useXarchLgRule = (params.architecture_type == "xarch")


class DirectLegalize(object):
    """
    @brief Legalize instances with college-admission algorithm.
    This algorithm is for LUT and FF.
    """

    def __init__(self, placedb, params):
        """
        @brief initialization
        """
        super(DirectLegalize, self).__init__()
        self.placedb = placedb
        self.param = DirectLegalizeParam(params)
        gpInstStddevTrunc = 2.5
        gpInstStddev = np.sqrt(2.5e-4 * (self.placedb.movableRange()[1] - self.placedb.movableRange()[0])) / (
            2.0 * gpInstStddevTrunc)
        self.param.nbrDistEnd = 1.2 * gpInstStddev * gpInstStddevTrunc
        self.param.CLB_capacity = self.placedb.CLBCapacity()
        self.param.BLE_capacity = self.placedb.BLECapacity()
        self.param.numClockNet = 0
        self.param.numHalfColumn = 0
        self.param.maxClockNetPerHalfColumn = 0

    def forward(self, pos):
        if pos.is_cuda:
            local_pos = pos.cpu()
        else:
            local_pos = pos
        return direct_lg_cpp.forward(
            self.placedb,
            self.param,
            local_pos).to(pos.device)

    def __call__(self, pos):
        return self.forward(pos)


class ClockAwareDirectLegalize(DirectLegalize):
    def __init__(self, placedb, params,
                 inst_to_clock_indexes,
                 max_clock_net_per_half_column=0,
                 honor_fence_region_constraints=False):
        super(ClockAwareDirectLegalize, self).__init__(placedb, params)
        self.honor_fence_region_constraints = honor_fence_region_constraints
        self.inst_to_clock_indexes = inst_to_clock_indexes
        self.clock_available_clock_region = None
        self.param.numClockNet = self.placedb.numClockNets()
        self.param.numHalfColumn = self.placedb.numHalfColumnRegions()
        self.param.maxClockNetPerHalfColumn = max_clock_net_per_half_column

    def reset_honor_fence_region_constraints(self, honor_fence_region_constraints: bool):
        self.honor_fence_region_constraints = honor_fence_region_constraints

    def reset_clock_available_clock_region(self, clock_available_clock_region):
        self.clock_available_clock_region = clock_available_clock_region

    def forward(self, pos):
        if not self.honor_fence_region_constraints:
            return super(ClockAwareDirectLegalize, self).forward(pos)
        assert self.clock_available_clock_region is not None
        local_pos = pos.cpu() if pos.is_cuda else pos
        p, hc_avail_map = direct_lg_cpp.clock_aware_forward(
            self.placedb,
            self.param,
            local_pos,
            self.inst_to_clock_indexes,
            self.clock_available_clock_region)
        return p.to(pos.device), hc_avail_map
