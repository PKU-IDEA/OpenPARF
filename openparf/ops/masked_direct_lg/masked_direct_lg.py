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

from . import masked_direct_lg_cpp


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
    Masked direct legalization for LUT and FF. The diffrence from orginary `direct_lg` opeartor is that,
    we will pass a mask and the masked cells remain intact. Besides, other instances CANNOT move into the
    sites where the masked cells are located.
    """

    def __init__(self, placedb, data_cls, params):
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
        self.chain_lut_ids = data_cls.chain_lut_ids.bs.cpu()

    def forward(self, pos_xyz):
        if pos_xyz.is_cuda:
            local_pos_xyz = pos_xyz.cpu()
        else:
            local_pos_xyz = pos_xyz
        rv = masked_direct_lg_cpp.forward(
            self.placedb,
            self.param,
            self.chain_lut_ids,
            local_pos_xyz)
        with torch.no_grad():
            pos_xyz.data.copy_(rv)

    def __call__(self, pos):
        return self.forward(pos)
