#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : energy_well.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 09.17.2020
# Last Modified Date: 09.17.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>

import logging
import torch
from torch import nn
from torch.autograd import Function
import pdb

from openparf.py_utils import stopwatch
from . import energy_well_cpp

from openparf import configure

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import energy_well_cuda

logger = logging.getLogger(__name__)


class EnergyWellFunction(Function):
    @staticmethod
    def forward(ctx,
                inst_pos,
                half_inst_sizes,
                inst_areas,
                well_boxes,
                inst_cr_avail_map,
                energy_function_exponents,
                num_crs,
                placedb,
                site2cr_map):
        forward_stopwatch = stopwatch.Stopwatch()
        forward_stopwatch.start()

        num_insts = inst_pos.numel() // 2

        if inst_pos.is_cuda:
            assert site2cr_map is not None
            energy_arr, selected_crs = energy_well_cuda.forward(inst_pos,
                                                                half_inst_sizes,
                                                                well_boxes,
                                                                inst_cr_avail_map,
                                                                energy_function_exponents,
                                                                num_crs,
                                                                placedb, site2cr_map)
            torch.cuda.synchronize()
        else:
            energy_arr, selected_crs = energy_well_cpp.forward(inst_pos,
                                                                half_inst_sizes,
                                                                well_boxes,
                                                                inst_cr_avail_map,
                                                                energy_function_exponents,
                                                                num_crs,
                                                                placedb)

        ctx.save_for_backward(inst_pos,
                              half_inst_sizes,
                              well_boxes,
                              selected_crs,
                              energy_function_exponents,
                              inst_areas)

        elapsed_time_ms = forward_stopwatch.elapsed(stopwatch.Stopwatch.TimeFormat.kMillSecond)
        logger.debug("Energy Well forward: %.3f ms" % elapsed_time_ms)
        return energy_arr

    @staticmethod
    def backward(ctx, grad_well_energy):
        backward_stopwatch = stopwatch.Stopwatch()
        backward_stopwatch.start()

        (inst_pos, half_inst_sizes, well_boxes, selected_crs, energy_function_exponents, inst_areas) = ctx.saved_tensors
        assert grad_well_energy.numel() * 2 == inst_pos.numel()
        assert not torch.any(torch.isnan(grad_well_energy))
        assert inst_pos.dtype == half_inst_sizes.dtype
        assert inst_pos.dtype == well_boxes.dtype
        assert inst_pos.dtype == inst_areas.dtype
        assert inst_pos.dtype == grad_well_energy.dtype
        grad_well_energy = grad_well_energy.contiguous()

        func = energy_well_cuda.backward if inst_pos.is_cuda else energy_well_cpp.backward

        pos_grad = func(
            inst_pos,
            half_inst_sizes,
            well_boxes,
            selected_crs,
            energy_function_exponents,
            grad_well_energy
        )
        if inst_pos.is_cuda:
            torch.cuda.synchronize()
        elapsed_time_ms = backward_stopwatch.elapsed(stopwatch.Stopwatch.TimeFormat.kMillSecond)
        logger.debug("Energy Well backward: %.3f ms" % elapsed_time_ms)
        return pos_grad, None, None, None, None, None, None, None, None


class EnergyWell(nn.Module):
    def __init__(self, well_boxes, energy_function_exponents, inst_areas, inst_sizes,
                 inst_cr_avail_map, num_crs, placedb):
        super(EnergyWell, self).__init__()
        self.inst_cr_avail_map, self.half_inst_sizes, self.inst_areas = None, None, None
        self.well_boxes, self.energy_function_exponents = None, None
        self.num_crs = num_crs
        self.placedb = placedb
        self.reset_box(well_boxes)
        self.reset_instance(inst_areas, inst_sizes, inst_cr_avail_map, energy_function_exponents)
        self.site2cr_map = None

    def reset_box(self, well_boxes):
        self.well_boxes = well_boxes

    def reset_instance(self, inst_areas, inst_sizes, inst_cr_avail_map, energy_function_exponents):
        self.inst_areas = inst_areas
        self.half_inst_sizes = None if inst_sizes is None else inst_sizes.mul(0.5)
        self.inst_cr_avail_map = inst_cr_avail_map
        self.energy_function_exponents = energy_function_exponents

    def forward(self,
                inst_pos):
        """
        :param inst_pos: center of instances, array of (x, y) pairs, shape of (#instance, )
        :return: the well energy of instances
        """
        assert self.well_boxes is not None
        assert self.energy_function_exponents is not None
        assert self.inst_cr_avail_map is not None
        assert self.half_inst_sizes is not None
        assert self.inst_areas is not None

        num_insts = int(inst_pos.shape[0])

        assert self.energy_function_exponents.shape == (num_insts,)
        assert self.half_inst_sizes.shape == (num_insts, 2)
        assert self.inst_areas.shape == (num_insts,)

        if inst_pos.is_cuda and self.site2cr_map is None:
            self.site2cr_map = energy_well_cpp.genSite2CrMap(self.placedb).to(inst_pos.device)

        return EnergyWellFunction.apply(
            inst_pos,
            self.half_inst_sizes,
            self.inst_areas,
            self.well_boxes,
            self.inst_cr_avail_map,
            self.energy_function_exponents,
            self.num_crs,
            self.placedb,
            self.site2cr_map
        )
