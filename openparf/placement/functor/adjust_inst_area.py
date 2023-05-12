#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : adjust_inst_area.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 08.11.2020
# Last Modified Date: 07.18.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

import pdb
import torch
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)
from openparf.ops.stable_div import stable_div
from openparf.ops.raw_instance_area import raw_instance_area
from openparf.py_utils.base import log_dict, DeferredAction


class AdjustInstArea(object):
    def __init__(self,
                 xl,
                 yl,
                 xh,
                 yh,
                 route_bin_size_x, route_bin_size_y,
                 pin_bin_size_x, pin_bin_size_y,
                 scaling_hyper_parameter,
                 total_place_area,
                 total_whitespace_area,
                 total_fixed_area,
                 fixed_insts_num,
                 unit_pin_capacity,
                 inst_pin_weights):
        """ Functor that adjust the final instance areas in terms of routing utilization and pin utilization.

        :param xl: minimum x-coordinates of the layout
        :param yl: minimum y-coordinates of the layout
        :param xh: maximum x-coordinates of the layout
        :param yh: maximum y-coordinates of the layout
        :param route_bin_size_x: the width of bin in the x-axis direction in routing utilization map
        :param route_bin_size_y: the height of bins in the y-axis direction in routing utilization map
        :param pin_bin_size_x: the width of bins in the x-axis direction in pin utilization map
        :param pin_bin_size_y: the height of bins in the y-axis direction in pin utilization map
        :param scaling_hyper_parameter: heuristic parameter for scaling increasing movable area.
        :param total_place_area: movable instances' area + filler instances' area
        :param total_whitespace_area: layout area - (fixed instances' area + movable instances' area + filler
        instances' area)
        :param total_fixed_area: fixed instances' area
        :param fixed_insts_num: number of fixed instances
        :param unit_pin_capacity: number of pins per unit area
        :param inst_pin_weights: pin weights for each instances, shape of (#instance,)
        """
        super(AdjustInstArea, self).__init__()

        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh

        self.layout_area = (xh - xl) * (yh - yl)
        self.total_place_area = total_place_area
        self.total_whitespace_area = total_whitespace_area
        self.total_fixed_area = total_fixed_area
        self.fixed_insts_num = fixed_insts_num

        self.scaling_hyper_parameter = scaling_hyper_parameter
        self.unit_pin_capacity = unit_pin_capacity
        self.inst_pin_weights = inst_pin_weights

        # Stable division that can handle divisor with zeros
        self.stable_div_op = stable_div.StableDiv()

        # operator that computes raw instance area from routing utilization map
        self.compute_movable_inst_area_from_route_functor = (
            raw_instance_area.ComputeRawInstanceArea(
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                bin_size_x=route_bin_size_x,
                bin_size_y=route_bin_size_y
            ))

        # operator that computes raw instance area from pin utilization map
        self.compute_movable_inst_area_from_pin_functor = (
            raw_instance_area.ComputeRawInstanceArea(
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                bin_size_x=pin_bin_size_x,
                bin_size_y=pin_bin_size_y
            ))

    def get_instance_increment(self,
                               inst_pos,
                               inst_sizes,
                               movable_range,
                               filler_range,
                               route_utilization_map,
                               pin_utilization_map,
                               resource_opt_area,
                               logging_prefix=""):
        """

        :param logging_prefix: logging prefix
        :param inst_pos: tensor of all inst central positions, shape of (#insts, 2)
        :param inst_sizes: [in,out] size pair (width, height) of all inst sizes, shape of (#insts, 2)
        :param movable_range: the index pair [lower bound, higher bound) of the movable insts
        :param filler_range: the index pair [lower bound, higher bound) of the fillers
        :param route_utilization_map: routing utilization map, shape of (|route_num_bins_x|, |route_num_bins_y|)
        :param pin_utilization_map: pin utilization map, shape of (|pin_num_bins_x|, |pin_num_bins_y|)
        :param resource_opt_area: resource optimization areas, shape of (#insts,)
        """
        with torch.no_grad():
            assert len(list(inst_sizes.size())) == 2
            num_insts = len(inst_sizes)
            assert inst_sizes.shape == inst_pos.shape
            assert inst_sizes.shape == (num_insts, 2)

            adjust_route_area_flag = route_utilization_map is not None
            adjust_pin_area_flag = pin_utilization_map is not None
            adjust_resource_area_flag = resource_opt_area is not None
            adjust_area_flag = adjust_route_area_flag or adjust_pin_area_flag or adjust_resource_area_flag
            if not adjust_area_flag:
                return False, False, False, False, False, False

            total_insts_num = torch.nonzero(inst_sizes[:, 0]).size(0)

            # compute old areas of movable insts
            movable_inst_sizes = inst_sizes[movable_range[0]:movable_range[1]]
            zero_movable_area_indexes = (movable_inst_sizes[:, 0] == 0).nonzero().squeeze()
            old_movable_area = movable_inst_sizes[:, 0] * movable_inst_sizes[:, 1]
            old_movable_area_sum = old_movable_area.sum()
            movable_insts_num = torch.nonzero(old_movable_area.data).size(0)

            # compute old areas of filler insts
            filler_inst_sizes = inst_sizes[filler_range[0]:filler_range[1]]
            old_filler_area = filler_inst_sizes[:, 0] * filler_inst_sizes[:, 1]
            old_filler_area_sum = old_filler_area.sum()
            filler_insts_num = torch.nonzero(old_filler_area.data).size(0)

            # calculate half inst sizes in advance for the convenience of calculating inst rectangular contour.
            inst_half_sizes = inst_sizes.mul(0.5)

            route_opt_movable_area, pin_opt_movable_area = None, None
            if adjust_route_area_flag is True:
                route_opt_movable_area = self.compute_movable_inst_area_from_route_functor(
                    inst_pos=inst_pos,
                    inst_half_sizes=inst_half_sizes,
                    movable_range=movable_range,
                    utilization_map=route_utilization_map,
                )
            if adjust_pin_area_flag is True:
                pin_opt_movable_area = self.compute_movable_inst_area_from_pin_functor(
                    inst_pos=inst_pos,
                    inst_half_sizes=inst_half_sizes,
                    movable_range=movable_range,
                    utilization_map=pin_utilization_map
                )
                # The Pin density optimized area here is independent to the original instance area.
                #   Derived from elfplace's implementation.
                #   For more Information, see the equation(30) in `elfplace` paper.
                movable_inst_pin_weights = self.inst_pin_weights[movable_range[0]:movable_range[1]]
                pin_opt_movable_area.mul_(movable_inst_pin_weights)
                pin_opt_movable_area = self.stable_div_op(pin_opt_movable_area,
                                                          old_movable_area * self.unit_pin_capacity)

            resource_opt_movable_area = None
            if adjust_resource_area_flag is True:
                resource_opt_movable_area = (resource_opt_area[movable_range[0]:movable_range[1]]).clone()
                resource_opt_movable_area[zero_movable_area_indexes] = 0

            # compute the extra area max(route_opt_area, pin_opt_area) over the base area for each movable node
            new_movable_area = None
            if adjust_pin_area_flag:
                new_movable_area = (pin_opt_movable_area if new_movable_area is None else
                                    torch.max(pin_opt_movable_area, new_movable_area))
            if adjust_resource_area_flag:
                new_movable_area = (resource_opt_movable_area if new_movable_area is None else
                                    torch.max(resource_opt_movable_area, new_movable_area))
            if adjust_route_area_flag:
                new_movable_area = (route_opt_movable_area if new_movable_area is None else
                                    torch.max(route_opt_movable_area, new_movable_area))
            assert new_movable_area is not None
            increasing_movable_area = new_movable_area - old_movable_area
            increasing_movable_area.clamp_(min=0)
            increasing_movable_area_sum = increasing_movable_area.sum()

            # check whether the total area is larger than the max area requirement.
            # If so, scale the extra area to meet the requirement.
            # We assume the total base area is no greater than the max area requirement.
            scaling_factor = (self.stable_div_op(self.total_place_area - old_movable_area_sum,
                                                 increasing_movable_area_sum).item())

            # Reset the new_movable_area as base_area + scaled area increment
            if scaling_factor <= 0:
                new_movable_area = old_movable_area
                increasing_movable_area_sum = torch.zeros_like(old_movable_area_sum)
            elif scaling_factor >= 1:
                new_movable_area = old_movable_area + increasing_movable_area
            else:
                new_movable_area = old_movable_area + increasing_movable_area * scaling_factor
                increasing_movable_area_sum *= scaling_factor

            new_movable_area_sum = old_movable_area_sum + increasing_movable_area_sum
            # in case numerical precision
            assert new_movable_area_sum <= self.total_place_area + 1E-6

            def update_instance_area():
                with torch.no_grad():
                    with DeferredAction() as defer:
                        content_dict = {logging_prefix: ""}
                        defer(log_dict, logger.info, content_dict)
                        # adjust the size of movable instances
                        # each movable node have its own inflation ratio
                        # |movable_inst_ratios| is a tensor with shape of (#movable_insts)
                        movable_inst_ratios = self.stable_div_op(new_movable_area, old_movable_area)
                        # adjust inplace
                        movable_inst_ratios.sqrt_()
                        movable_inst_sizes.mul_(movable_inst_ratios.unsqueeze(1))
                        content_dict["Average inflation ratio for movable instances"] = (
                                "%6.4lf" % ((movable_inst_ratios.sum() / movable_insts_num).item()))
                        content_dict["Maximum inflation ratio for movable instances"] = (
                                "%6.4lf" % (movable_inst_ratios.max().data.item()))

                        # scale the filler instance areas to let the total area be total_place_area
                        # all the filler nodes share the same deflation ratio, |filler_scaling_ratio| is a scalar
                        if new_movable_area_sum + old_filler_area_sum > self.total_place_area:
                            new_filler_area_sum = (self.total_place_area - new_movable_area_sum).clamp_(min=0)
                            filler_scaling_ratio = self.stable_div_op(new_filler_area_sum, old_filler_area_sum)
                            filler_scaling_ratio.sqrt_()
                            filler_inst_sizes.mul_(filler_scaling_ratio)
                            content_dict['Inflation ratio for filler instances'] = "%.4lf" % filler_scaling_ratio
                        else:
                            new_filler_area_sum = old_filler_area_sum
                            content_dict['Inflation ratio for filler instances'] = 1

                        content_dict["Layout Area"] = ", ".join([
                            "%.5E" % self.layout_area,
                            "(%6.2lf%%)" % (self.layout_area / self.layout_area * 100)])
                        content_dict["Movable instances' Area"] = ", ".join([
                            "%.5E" % new_movable_area_sum,
                            "(%6.2lf%%)" % (new_movable_area_sum / self.layout_area * 100).item(),
                            "count = %d" % movable_insts_num])
                        content_dict["Filler instances' Area"] = ", ".join([
                            "%.5E" % new_filler_area_sum,
                            "(%6.2lf%%)" % (new_filler_area_sum / self.layout_area * 100).item(),
                            "count = %d" % filler_insts_num])
                        content_dict["Fixed instances' Area"] = ", ".join([
                            "%.5E" % self.total_fixed_area,
                            "(%6.2lf%%)" % (self.total_fixed_area / self.layout_area * 100).item(),
                            "count = %d" % self.fixed_insts_num])
                        content_dict["Whitespace Area"] = ", ".join([
                            "%.5E" % self.total_whitespace_area,
                            "(%6.2lf%%)" % (self.total_whitespace_area / self.layout_area * 100).item()])
                        content_dict["Place Area(movable+filler)"] = ", ".join([
                            "%.5E" % self.total_place_area,
                            "(%6.2lf%%)" % (self.total_place_area / self.layout_area * 100).item(),
                            "count = %d" % total_insts_num])

            route_movable_area_increment = ((route_opt_movable_area - old_movable_area).clamp_(min=0).sum()
                                            if adjust_route_area_flag else torch.zeros_like(old_movable_area_sum))
            pin_movable_area_increment = ((pin_opt_movable_area - old_movable_area).clamp_(min=0).sum()
                                          if adjust_pin_area_flag else torch.zeros_like(old_movable_area_sum))
            resource_movable_area_increment = ((resource_opt_movable_area - old_movable_area).clamp_(min=0).sum()
                                               if adjust_resource_area_flag else torch.zeros_like(old_movable_area_sum))
            assert increasing_movable_area_sum is not None
            return (increasing_movable_area_sum, route_movable_area_increment, pin_movable_area_increment,
                    resource_movable_area_increment, old_movable_area_sum, update_instance_area)


def check_increment_conditions(final_movable_area_increment,
                               route_movable_area_increment,
                               pin_movable_area_increment,
                               resource_movable_area_increment,
                               old_movable_area_sum,
                               adjust_route_area_flag,
                               adjust_pin_area_flag,
                               adjust_resource_area_flag,
                               pin_area_adjust_stop_ratio,
                               route_area_adjust_stop_ratio,
                               resource_area_adjust_stop_ratio,
                               area_adjust_stop_ratio):
    with torch.no_grad():
        with DeferredAction() as defer:
            content_dict = {}
            defer(log_dict, logger.info, content_dict)
            stable_div_op = stable_div.StableDiv()

            if adjust_route_area_flag:
                route_area_increment_ratio = stable_div_op(route_movable_area_increment, old_movable_area_sum)
                adjust_route_area_flag = route_area_increment_ratio.data.item() > route_area_adjust_stop_ratio
                content_dict['Routing optimization area increment ratio'] = (
                        "%6.4lf%% (threshold %6.4lf%%)" % (route_area_increment_ratio.item() * 100,
                                                           route_area_adjust_stop_ratio * 100))
            else:
                content_dict['Routing optimization area increment ratio'] = "None"

            if adjust_pin_area_flag:
                pin_area_increment_ratio = stable_div_op(pin_movable_area_increment, old_movable_area_sum)
                adjust_pin_area_flag = pin_area_increment_ratio.data.item() > pin_area_adjust_stop_ratio
                content_dict['Pin optimization area increment ratio'] = (
                        "%6.4lf%% (threshold %6.4lf%%)" % (pin_area_increment_ratio.item() * 100,
                                                           pin_area_adjust_stop_ratio * 100))
            else:
                content_dict['Pin optimization area increment ratio'] = "None"

            # compute the adjusted area increasing ratio
            if adjust_resource_area_flag:
                resource_area_increment_ratio = stable_div_op(resource_movable_area_increment, old_movable_area_sum)
                adjust_resource_area_flag = (resource_area_increment_ratio.data.item()
                                             > resource_area_adjust_stop_ratio)
                content_dict['Resource optimization area increment ratio'] = (
                        "%6.4lf%% (threshold %6.4lf%%)" % (resource_area_increment_ratio.item() * 100,
                                                           resource_area_adjust_stop_ratio * 100))
            else:
                content_dict['Resource optimization area increment ratio'] = "None"

            area_increment_ratio = stable_div_op(final_movable_area_increment, old_movable_area_sum)
            content_dict['Total optimization area increment ratio'] = (
                    "%6.4lf%% (threshold %6.4lf%%)" % (area_increment_ratio.item() * 100,
                                                       area_adjust_stop_ratio * 100))
            adjust_area_flag = (area_increment_ratio.data.item() > area_adjust_stop_ratio
                                and (adjust_route_area_flag or adjust_pin_area_flag or adjust_resource_area_flag))

            return adjust_area_flag, adjust_route_area_flag, adjust_pin_area_flag, adjust_resource_area_flag
