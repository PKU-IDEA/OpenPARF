#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : functor_collections.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 08.11.2020
# Last Modified Date: 08.19.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>

import pdb
from .adjust_inst_area import AdjustInstArea


def build_adjust_instance_area_functor(params, inst_areas, xl, yl, xh, yh, movable_range,
                                       filler_range, fixed_range, inst_pin_weights):
    movable_area_sum = inst_areas[movable_range[0]:movable_range[1]].sum()
    filler_area_sum = inst_areas[filler_range[0]:filler_range[1]].sum()
    fixed_area_sum = inst_areas[fixed_range[0]:fixed_range[1]].sum()
    fixed_insts_num = inst_areas[fixed_range[0]:fixed_range[1]].nonzero(as_tuple=True)[0].size()

    # layout area = movable area + filler area + fixed area + whitespace
    layout_area = (xh - xl) * (yh - yl)
    total_place_area = filler_area_sum + movable_area_sum
    total_whitespace_area = layout_area - total_place_area - fixed_area_sum

    assert total_whitespace_area >= 0

    return AdjustInstArea(
        xl=xl,
        yl=yl,
        xh=xh,
        yh=yh,
        route_bin_size_x=params.routing_bin_size_x,
        route_bin_size_y=params.routing_bin_size_y,
        pin_bin_size_x=params.pin_bin_size_x,
        pin_bin_size_y=params.pin_bin_size_y,
        scaling_hyper_parameter=params.gp_adjust_area_scaling_hyper_parameter,
        total_place_area=total_place_area,
        total_whitespace_area=total_whitespace_area,
        total_fixed_area=fixed_area_sum,
        fixed_insts_num=fixed_insts_num,
        inst_pin_weights=inst_pin_weights,
        unit_pin_capacity=params.unit_pin_capacity)


class FunctorCollections(object):
    """ A collection of functors written in pure Python.

    """

    def __init__(self, params, data_cls, placedb):
        # Derived from elfplace's implementation. Since control set pins(e.g., CK/CR/CE) in
        # FFs can be largely shared, it is almost always an overestimation of using FF's pin count
        # directly. To compensate this effect, we give FFs a pre-defined pin weight. For non-FF
        # instances, we simply use pin counts as the weights.
        inst_pin_start = data_cls.inst_pin_map.b_starts
        inst_pin_weights = (inst_pin_start[1:] - inst_pin_start[:-1]).type(data_cls.inst_sizes.dtype)
        inst_pin_weights.masked_fill_(data_cls.is_inst_ffs.bool(), params.ff_pin_weight)
    
        self.adjust_area_functors = []
        for at_name in params.gp_adjust_area_types:
            at_id = placedb.getAreaTypeIndexFromName(at_name)
            inst_areas = data_cls.inst_areas[:, at_id]
            self.adjust_area_functors.append(
                build_adjust_instance_area_functor(
                    params=params,
                    inst_areas=inst_areas,
                    xl=data_cls.diearea[0],
                    yl=data_cls.diearea[1],
                    xh=data_cls.diearea[2],
                    yh=data_cls.diearea[3],
                    movable_range=data_cls.movable_range,
                    filler_range=data_cls.filler_range,
                    fixed_range=data_cls.fixed_range,
                    inst_pin_weights=inst_pin_weights)
            )
