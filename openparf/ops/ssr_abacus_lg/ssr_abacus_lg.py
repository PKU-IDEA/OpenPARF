#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : ssr_abacus_lg.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 12.02.2021
# Last Modified Date: 12.02.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import torch
import pdb

from . import ssr_abacus_lg_cpp


class SsrAbacusLegalizer(object):
    def __init__(self, placedb, data_cls, concerned_inst_ids: torch.Tensor, area_type_id):
        assert concerned_inst_ids.dtype == torch.int32
        assert not concerned_inst_ids.is_cuda
        assert concerned_inst_ids.is_contiguous

        self.placedb = placedb
        self.concerned_inst_ids = concerned_inst_ids
        self.area_type_id = area_type_id
        self.chain_ssr_ids = data_cls.ssr_chain_ids

    def __call__(self, pos_xyz: torch.Tensor):
        local_pos_xyz = pos_xyz.cpu() if pos_xyz.is_cuda else pos_xyz
        ssr_abacus_lg_cpp.forward(
            self.placedb, local_pos_xyz, self.concerned_inst_ids, self.area_type_id,
            self.chain_ssr_ids.bs.cpu(),
            self.chain_ssr_ids.b_starts.cpu()
        )
        if local_pos_xyz is not pos_xyz:
            with torch.no_grad():
                pos_xyz.data.copy_(local_pos_xyz)
