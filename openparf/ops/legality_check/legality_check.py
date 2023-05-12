#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : legality_check.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 07.24.2020
# Last Modified Date: 07.24.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>

import pdb

from . import legality_check_cpp


class LegalityCheck(object):
    """
    @brief Check legality.
    """

    def __init__(self,
                 placedb,
                 data_cls,
                 check_z_flag,
                 max_clk_per_clock_region,
                 max_clk_per_half_column
                 ):
        """
        @brief initialization
        """
        super(LegalityCheck, self).__init__()
        self.placedb = placedb
        self.data_cls = data_cls
        self.chain_cla_ids = data_cls.chain_cla_ids
        self.chain_lut_ids = data_cls.chain_lut_ids
        self.chain_ssr_ids = data_cls.ssr_chain_ids
        self.check_z_flag = check_z_flag
        self.max_clk_per_clock_region = max_clk_per_clock_region
        self.max_clk_per_half_column = max_clk_per_half_column

    def forward(self, pos, arch):
        if pos.is_cuda:
            local_pos = pos.cpu()
        else:
            local_pos = pos
        rv = legality_check_cpp.forward(
            self.placedb,
            self.check_z_flag,
            self.max_clk_per_clock_region,
            self.max_clk_per_half_column,
            local_pos)
        if arch == 'xarch':
            rv |= legality_check_cpp.xarchForward(
                self.placedb,
                self.chain_cla_ids.bs.cpu(),
                self.chain_cla_ids.b_starts.cpu(),
                self.chain_lut_ids.bs.cpu(),
                self.chain_lut_ids.b_starts.cpu(),
                self.chain_ssr_ids.bs.cpu(),
                self.chain_ssr_ids.b_starts.cpu(),
                local_pos,
            )
        return rv

    def __call__(self, pos, arch='ispd'):
        return self.forward(pos, arch)
