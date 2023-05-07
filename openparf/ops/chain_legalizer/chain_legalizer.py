#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : chain_legalizer.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 09.10.2021
# Last Modified Date: 09.10.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import torch
import pdb

from . import chain_legalizer_cpp


class ChainLegalizer(object):
    def __init__(self, placedb, data_cls, concerned_inst_ids: torch.Tensor, area_type_id, search_manh_dist_increment, max_iter):
        assert concerned_inst_ids.dtype == torch.int32
        assert not concerned_inst_ids.is_cuda
        assert concerned_inst_ids.is_contiguous

        self.placedb = placedb
        self.chain_cla_ids = data_cls.chain_cla_ids
        self.chain_lut_ids = data_cls.chain_lut_ids
        self.concerned_inst_ids = concerned_inst_ids
        self.area_type_id = area_type_id
        self.search_manh_dist_increment = search_manh_dist_increment
        self.max_iter = max_iter

    def __call__(self, pos_xyz: torch.Tensor):
        local_pos_xyz = pos_xyz.cpu() if pos_xyz.is_cuda else pos_xyz
        chain_legalizer_cpp.forward(
            self.placedb, local_pos_xyz, self.concerned_inst_ids, self.area_type_id,
            self.search_manh_dist_increment, self.max_iter,
            self.chain_cla_ids.bs.cpu(),
            self.chain_cla_ids.b_starts.cpu(),
            self.chain_lut_ids.bs.cpu(),
            self.chain_lut_ids.b_starts.cpu(),
        )
        if local_pos_xyz is not pos_xyz:
            with torch.no_grad():
                pos_xyz.data.copy_(local_pos_xyz)
