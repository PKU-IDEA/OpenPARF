#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mcf_lg.py
# Author            : Yibai Meng <mengyibai@pku.edu.cn>
# Date              : 05.21.2021
# Last Modified Date: 05.21.2021
# Last Modified By  : Yibai Meng <mengyibai@pku.edu.cn>
import logging

import torch

from . import mcf_lg_cpp
from ...py_config import rule
import pdb

logger = logging.getLogger(__name__)


class MinCostFlowLegalizer:
    def __init__(self, params, placedb, data_cls):
        self.data_cls = data_cls
        self.legalizer = mcf_lg_cpp.MinCostFlowLegalizer(placedb)
        self.honor_fence_region_constraints = False
        self.reset_honor_fence_region_constraints(False)
        if params.honor_half_column_constraints and params.maximum_clock_per_half_column:
            self.legalizer.set_max_clk_per_half_column(
                params.maximum_clock_per_half_column)
        self.inst_ids_groups = []
        data_cls.ssr_area_types = []

        layout = placedb.db().layout()
        for site_type in layout.siteTypeMap():
            resource_id = rule.is_single_site_single_resource_type(site_type, placedb.db())
            if resource_id is None:
                # This site type is not `SSSR`(single site, single resource).
                continue
            # As one model may correspond to different resources,
            # use the average area of different resources.
            inst_ids = placedb.collectInstIds(resource_id).tolist()
            site_boxes = placedb.collectSiteBoxes(resource_id).tolist()
            site_boxes = torch.tensor(
                site_boxes, dtype=data_cls.wl_precond.dtype)
            self.legalizer.add_sssir_instances(torch.tensor(
                inst_ids, dtype=torch.int32),
                data_cls.wl_precond.cpu(),
                site_boxes)
            # Add fixed and movable instances for instances local masks.
            self.inst_ids_groups.append(inst_ids)

            area_types = placedb.resourceAreaTypes(
                layout.resourceMap().resource(resource_id))
            for area_type in area_types:
                filler_bgn = data_cls.filler_range[0] + \
                    data_cls.num_fillers[:area_type].sum()
                filler_end = data_cls.filler_range[0] + \
                    data_cls.num_fillers[:area_type + 1].sum()
                filler_range = (filler_bgn, filler_end)
                # Add filler instances for for instances lock masks.
                self.inst_ids_groups[-1].extend(
                    range(filler_range[0], filler_range[1]))
                # Add area type lock masks.
                data_cls.ssr_area_types.append(area_type)
        logging.info("Single-site resource area types = %s" %
                     data_cls.ssr_area_types)

    def reset_honor_fence_region_constraints(self, v: bool):
        self.honor_fence_region_constraints = v
        self.legalizer.set_honor_clock_constraints(v)

    def reset_clock_available_clock_region(self, clock_available_clock_region):
        self.legalizer.reset_clock_available_clock_region(
            clock_available_clock_region)

    def reset_slr_aware_flag(self, v: bool):
        self.legalizer.set_slr_aware_flag(v)

    def __call__(self, pos):
        if pos.is_cuda:
            local_pos = pos.cpu()
        else:
            local_pos = pos
        with torch.no_grad():
            res = self.legalizer.forward(local_pos)
            pos.data.copy_(res)
            for i in range(len(self.inst_ids_groups)):
                self.data_cls.inst_lock_mask[self.inst_ids_groups[i]] = 1
            self.data_cls.area_type_lock_mask[self.data_cls.ssr_area_types] = 1
