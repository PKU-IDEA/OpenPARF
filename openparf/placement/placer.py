#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : placer.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.21.2020
# Last Modified Date: 08.26.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import traceback
import os
import sys
import time
import copy
import gzip
from collections import Counter
import cProfile as profile
import numpy as np
import random
import pathlib

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)
from openparf.py_utils.base import DeferredAction, log_dict
import openparf.py_utils.stopwatch as stopwatch

# from openparf.placement.statistics_viewer import PlacerStatisticsViewer

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import numpy as np
import torch
import torch.nn as nn

from .data_collections import DataCollections
from .op_collections import OpCollections
from .nesterov import NesterovAcceleratedGradientOptimizer
from .metric import OptIter, EvalMetric, array2str, iarray2str
from .place_model import PlaceModel, FenceRegionPlaceModel
from .draw_place import (
    draw_place,
    draw_fence_regions,
    draw_place_with_clock_region_assignments,
)
from .functor import adjust_inst_area, functor_collections

datatypes = {"float32": torch.float32, "float64": torch.float64}

debug_timing_flag = False


class Placer(nn.Module):
    """Top placement engine"""

    def __init__(self, params, placedb):
        self.restore_best_solution_flag = False

        torch.manual_seed(params.random_seed)
        np.random.seed(params.random_seed)
        random.seed(params.random_seed)
        if params.deterministic_flag:
            torch.set_deterministic(True)

        super(Placer, self).__init__()

        self.dtype = datatypes[params.dtype]
        self.device = torch.device("cuda" if params.gpu else "cpu")

        self.params = params
        self.placedb = placedb

        # initialize data collections
        self.data_cls = DataCollections(
            params, placedb, dtype=self.dtype, device=self.device
        )

        # define variables to optimize
        # this is center of cells, not lower left locations
        inst_locs_xy = self.data_cls.inst_locs_xyz.new_zeros(
            [self.data_cls.filler_range[1], 2]
        )
        inst_locs_xy[
            self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
        ].copy_(self.data_cls.inst_locs_xyz[:, 0:2])
        self.pos = nn.ParameterList(
            [nn.Parameter(inst_locs_xy + self.data_cls.inst_sizes_max / 2)]
        )

        self.data_cls.pos = self.pos

        # initialize operator collections
        self.op_cls = OpCollections(params, placedb, self.data_cls)

        # initialize functor collections.
        self.functor_cls = functor_collections.FunctorCollections(
            params=self.params, data_cls=self.data_cls, placedb=self.placedb
        )

        # synthetic adjust area configuration options
        self.gp_adjust_area, self.gp_adjust_route_area = None, None
        self.gp_adjust_pin_area, self.gp_adjust_resource_area = None, None
        self.num_gp_adjust_area = None

        # Clock fence region confining options
        self.num_confine_fence_region = None
        self.confine_fence_region = None
        self.metric_before_clk_assignment = None

        # Backup primordial instances size. We will inflate instance sizes in cell area adjustment stage. However,
        # we have to use primordial instances size in legalization stage.
        self.primordial_inst_sizes = self.data_cls.inst_sizes.clone().detach()
        self.primordial_inst_sizes_max = self.primordial_inst_sizes.max(dim=1)[0]
        self.primordial_inst_areas = (
            self.primordial_inst_sizes_max[..., 0]
            * self.primordial_inst_sizes_max[..., 1]
        )

        # Stopwatch for each stage
        self.adjust_area_stopwatch = stopwatch.IntervalStopwatch()
        self.confine_fence_region_stopwatch = stopwatch.IntervalStopwatch()

        # SSR(e.g. DSP, RAM) legalization
        self.last_ssr_legalize_iter = None

        self.last_area_inflation_iter = None

        self.last_clock_assignment_iter = None

        # save the best solution for clock-aware placement
        self.best_pos_before_ck_ssir_lg = None
        self.best_sol_metric = None

        # Placer model and optimizer
        self.model = None
        self.optimizer = None
        self.optimizer_initial_state = None

        self.visualization_writer = None

        self.eta_update_counter = None
        self.timing_optimization_counter = 0

    def reset_optimizer(self, opt_iter):
        # reset density and overflow operator to update stretched sizes
        self.op_cls.density_op.reset()
        self.op_cls.overflow_op.reset()

        # evaluate overflow
        overflow = self.op_cls.normalized_overflow_op(self.data_cls.pos[0])

        # update gamma
        self.update_gamma(opt_iter, overflow)

        # reset multiplier(lambda)
        self.reset_lambdas(self.params.reset_lambda_param)
        # Alternative way to reset the lambda
        # self.initialize_lambdas()

        # Reset step size
        self.reset_step_size()

        # load state to restart the optimizer
        self.optimizer.load_state_dict(self.optimizer_initial_state)

        # learning rate must be reset after loading the optimizer state
        self.initialize_learning_rate(self.model, self.optimizer, 0.1)

        if self.params.gp_timing_adjustment:
            # reset timing adjustment
            self.last_timing_adjustment_threshold = (
                self.params.gp_timing_adjustment_overflow_threshold
            )
            self.data_cls.net_weights.data.copy_(
                torch.tensor(
                    self.placedb.netWeights(),
                    dtype=self.data_cls.net_weights.dtype,
                    device=self.data_cls.net_weights.device,
                )
            )
            self.num_gp_timing_adjustment = 0

    @staticmethod
    def get_utilization_map_overflow(utilization_map: torch.Tensor):
        overflow_map = utilization_map.sub(1.0).clamp_(min=0.0)
        return overflow_map.sum() / utilization_map.sum()

    def _gp_adjust_area_condition(self, current_metric):
        if not self.params.gp_adjust_area:
            return False
        if self.num_gp_adjust_area >= self.params.gp_max_adjust_area_iters:
            self.gp_adjust_area = False

        if not self.gp_adjust_area:
            return False

        # Check the instance area adjustment conditions
        max_overflow = None
        for at_name in self.params.gp_adjust_area_types:
            at_id = self.placedb.getAreaTypeIndexFromName(at_name)
            overflow = current_metric.overflow[at_id].item()
            max_overflow = (
                overflow if max_overflow is None else max(overflow, max_overflow)
            )
        assert max_overflow is not None

        if max_overflow > self.params.gp_adjust_area_overflow_threshold:
            return False
        return True

    def _gp_adjust_area(self, current_metric, opt_iter):
        """adjust cell(instance) area

        :param current_metric: current metric
        :return: boolean, whether satisfy the condition of adjusting instance area.
        """
        assert len(self.params.gp_adjust_area_types) > 0

        at_names = self.params.gp_adjust_area_types
        at_ids = []
        at_overflows = []
        for at_name in at_names:
            at_id = self.placedb.getAreaTypeIndexFromName(at_name)
            at_overflow = current_metric.overflow[at_id].item()
            at_ids.append(at_id)
            at_overflows.append(at_overflow)

        current_stage_idx = int(self.num_gp_adjust_area)
        self.num_gp_adjust_area += 1
        preface_string = (
            "\n[BEGIN    ] >>>>>>>>> Global Placement Area Adjustment Stage #%02d >>>>>>>>>"
            % (current_stage_idx)
        )
        postscript_string = (
            "\n[      END] <<<<<<<<< Global Placement Area Adjustment Stage #%02d <<<<<<<<<"
            % (current_stage_idx)
        )

        with DeferredAction() as defer:
            logger.info(preface_string)
            defer(logger.info, postscript_string)
            info_dict = {}
            info_dict["[Before Area Adjustment]"] = ""
            info_dict["Overall Adjustment Flag"] = self.gp_adjust_area
            info_dict["Routing Utilization Adjustment Flag"] = self.gp_adjust_route_area
            info_dict["Pin Utilization Adjustment Flag"] = self.gp_adjust_pin_area
            info_dict[
                "Resource Utilization Adjustment Flag"
            ] = self.gp_adjust_resource_area
            for i, at_id in enumerate(at_ids):
                at_name = at_names[i]
                at_overflow = at_overflows[i]
                info_dict[at_name + " Overflow"] = "%g" % at_overflow
            info_dict["Maximum Overflow"] = "%g" % max(at_overflows)
            info_dict[
                "Instance area adjustment's overflow threshold"
            ] = self.params.gp_adjust_area_overflow_threshold
            log_dict(logger.info, info_dict)
            with self.adjust_area_stopwatch:
                # tensor of instance locations, shape of (#cells, 2)
                inst_pos = self.data_cls.pos[0]
                # tensor of pin position, shape of (#pins, 2)
                pin_pos = self.op_cls.pin_pos_op(inst_pos)

                overflow_log_dict = {}

                clamped_route_utilization_map, clamped_pin_utilization_map = None, None
                resource_opt_area = None

                # resource utilization are adjusting strategy
                self.adjust_area_stopwatch.lap()
                if self.gp_adjust_resource_area:
                    resource_opt_area = self.op_cls.resource_area_op(inst_pos)
                resource_opt_area_elapsed_time_ms = (
                    self.adjust_area_stopwatch.lap(
                        stopwatch.Stopwatch.TimeFormat.kMicroSecond
                    )
                    / 1000.0
                )

                # route utilization aware adjusting strategy
                self.adjust_area_stopwatch.lap()
                if self.gp_adjust_route_area:
                    # The maximum/minimum instance area adjustment rate for routability optimization
                    route_opt_adjust_exponent = (
                        self.params.gp_adjust_area_route_opt_adjust_exponent
                    )
                    max_route_opt_adjust_rate = (
                        self.params.gp_adjust_area_max_route_opt_adjust_rate
                    )
                    min_route_opt_adjust_rate = 1.0 / max_route_opt_adjust_rate
                    # Whether turn on nn
                    if self.params.congestion_prediction_flag:
                        (
                            route_utilization_map,
                            horizontal_utilization_map,
                            vertical_utilization_map,
                        ) = self.op_cls.congestion_prediction_op(pin_pos)
                    else:
                        (
                            route_utilization_map,
                            horizontal_utilization_map,
                            vertical_utilization_map,
                        ) = self.op_cls.rudy_op(pin_pos)
                    clamped_route_utilization_map = route_utilization_map.pow_(
                        route_opt_adjust_exponent
                    ).clamp_(
                        min=min_route_opt_adjust_rate, max=max_route_opt_adjust_rate
                    )
                    horizontal_overflow = self.get_utilization_map_overflow(
                        horizontal_utilization_map
                    )
                    vertical_overflow = self.get_utilization_map_overflow(
                        vertical_utilization_map
                    )
                    overflow_log_dict[
                        "X-axis RUDY utilization overflow"
                    ] = "%6.2lf%%" % (horizontal_overflow.item() * 100)
                    overflow_log_dict[
                        "Y-axis RUDY utilization overflow"
                    ] = "%6.2lf%%" % (vertical_overflow.item() * 100)
                else:
                    overflow_log_dict["X-axis RUDY utilization overflow"] = "None"
                    overflow_log_dict["Y-axis RUDY utilization overflow"] = "None"
                route_utilization_map_elapsed_time_ms = (
                    self.adjust_area_stopwatch.lap(
                        stopwatch.Stopwatch.TimeFormat.kMicroSecond
                    )
                    / 1000.0
                )

                # pin utilization aware adjusting strategy
                self.adjust_area_stopwatch.lap()

                if self.gp_adjust_pin_area:
                    pin_utilization_map = self.op_cls.pin_utilization_op(
                        inst_sizes=self.data_cls.inst_sizes_max, inst_pos=inst_pos
                    )
                    pin_overflow = self.get_utilization_map_overflow(
                        pin_utilization_map
                    )
                    overflow_log_dict["Pin utilization overflow"] = "%6.2lf%%" % (
                        pin_overflow.item() * 100
                    )
                    # The maximum/minimum instance area adjustment rate for pin utilization optimization
                    max_pin_opt_adjust_rate = (
                        self.params.gp_adjust_area_max_pin_opt_adjust_rate
                    )
                    min_pin_opt_adjust_rate = 1.0 / max_pin_opt_adjust_rate
                    clamped_pin_utilization_map = pin_utilization_map.clamp_(
                        min=min_pin_opt_adjust_rate, max=max_pin_opt_adjust_rate
                    )
                else:
                    overflow_log_dict["Pin utilization overflow"] = "None"
                pin_utilization_map_elapsed_time_ms = (
                    self.adjust_area_stopwatch.lap(
                        stopwatch.Stopwatch.TimeFormat.kMicroSecond
                    )
                    / 1000.0
                )

                log_dict(logger.info, overflow_log_dict)

                update_closures = {}
                increment_infos = {}
                elapsed_times = {}
                for i, at_id in enumerate(at_ids):
                    self.adjust_area_stopwatch.lap()
                    at_name = at_names[i]
                    adjust_area_functor = self.functor_cls.adjust_area_functors[i]
                    (
                        final_movable_area_increment,
                        route_movable_area_increment,
                        pin_movable_area_increment,
                        resource_movable_area_increment,
                        old_movable_area_sum,
                        update_closure,
                    ) = adjust_area_functor.get_instance_increment(
                        inst_pos=inst_pos,
                        inst_sizes=self.data_cls.inst_sizes[:, at_id, :],
                        movable_range=self.data_cls.movable_range,
                        filler_range=self.data_cls.filler_range,
                        route_utilization_map=clamped_route_utilization_map,
                        pin_utilization_map=clamped_pin_utilization_map,
                        resource_opt_area=resource_opt_area,
                        logging_prefix="[" + at_name + " area type]",
                    )
                    update_closures[at_name] = update_closure
                    increment_infos[at_name] = {
                        "final_movable_area_increment": final_movable_area_increment,
                        "route_movable_area_increment": route_movable_area_increment,
                        "pin_movable_area_increment": pin_movable_area_increment,
                        "resource_movable_area_increment": resource_movable_area_increment,
                        "old_movable_area_sum": old_movable_area_sum,
                    }
                    elapsed_times[at_name] = (
                        self.adjust_area_stopwatch.lap(
                            stopwatch.Stopwatch.TimeFormat.kMicroSecond
                        )
                        / 1000.0
                    )

                increment_counter = Counter()
                for increment_info in increment_infos.values():
                    increment_counter.update(increment_info)
                args_dict = dict(increment_counter)

                # Add adjustment stop ratios into parameter dictionary
                args_dict[
                    "pin_area_adjust_stop_ratio"
                ] = self.params.gp_adjust_pin_area_stop_ratio
                args_dict[
                    "route_area_adjust_stop_ratio"
                ] = self.params.gp_adjust_route_area_stop_ratio
                args_dict[
                    "resource_area_adjust_stop_ratio"
                ] = self.params.gp_adjust_resource_area_stop_ratio
                args_dict[
                    "area_adjust_stop_ratio"
                ] = self.params.gp_adjust_area_stop_ratio

                # Check whether we should carry out area inflation for LUT and FF simultaneously
                (
                    self.gp_adjust_area,
                    self.gp_adjust_route_area,
                    self.gp_adjust_pin_area,
                    self.gp_adjust_resource_area,
                ) = adjust_inst_area.check_increment_conditions(
                    adjust_route_area_flag=self.gp_adjust_route_area,
                    adjust_pin_area_flag=self.gp_adjust_pin_area,
                    adjust_resource_area_flag=self.gp_adjust_resource_area,
                    **args_dict,
                )

                # Realize the adjusted area
                if self.gp_adjust_area:
                    for update_closure in update_closures.values():
                        update_closure()
                else:
                    logger.info("Do adjust instance area.")

                self.data_cls.inst_sizes_max.data.copy_(
                    self.data_cls.inst_sizes.max(dim=1)[0]
                )
                self.data_cls.inst_areas.data.copy_(
                    self.data_cls.inst_sizes[..., 0] * self.data_cls.inst_sizes[..., 1]
                )
                movable_inst_sizes_max = self.data_cls.inst_sizes_max[
                    self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
                ]
                self.data_cls.total_movable_areas.data.copy_(
                    self.data_cls.inst_areas[
                        self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
                    ].sum(dim=0)
                )

                self.reset_optimizer(opt_iter)
            log_dict(
                logger.info,
                {
                    "Gross Global Placement Area Adjustment Elapsed Time(ms)": self.adjust_area_stopwatch.internals[
                        -1
                    ].get_gross_elapsed_time(
                        stopwatch.Stopwatch.TimeFormat.kMicroSecond
                    )
                    / 1000.0,
                },
            )
            log_dict(
                logger.info,
                {
                    "[After Area Adjustment]": "",
                    "Overall adjustment flag": self.gp_adjust_area,
                    "Routing utilization adjustment flag": self.gp_adjust_route_area,
                    "Pin utilization adjustment flag": self.gp_adjust_pin_area,
                    "Resource Utilization adjustment flag": self.gp_adjust_resource_area,
                },
            )

    def timing_adjustment_condition(self, current_metric, opt_iter):
        if not self.params.gp_timing_adjustment:
            return False

        is_last_loop = True
        if (
            torch.any(
                self.data_cls.total_movable_areas[self.data_cls.ssr_area_types] > 0
            )
            and self.last_ssr_legalize_iter == 0
        ):
            is_last_loop = False

        if self.gp_adjust_area:
            is_last_loop = False

        if not is_last_loop:
            return False

        if torch.any(
            self.data_cls.total_movable_areas[self.data_cls.ssr_area_types] > 0
        ):
            # if self.last_ssr_legalize_iter == 0:
            #     return False

            if (
                self.last_ssr_legalize_iter != 0
                and self.last_ssr_legalize_iter + 100 > opt_iter.iteration
            ):
                return False

        if (
            self.last_area_inflation_iter is not None
            and self.last_area_inflation_iter + 100 > opt_iter.iteration
        ):
            return False

        if (
            self.last_timing_adjustment_iter is not None
            and self.last_timing_adjustment_iter
            + self.params.gp_timing_adjustment_interval
            > opt_iter.iteration
        ):
            return False

        # if self.gp_adjust_area:
        #     return False

        if self.num_gp_timing_adjustment >= self.params.gp_timing_adjustment_max_iters:
            self.gp_timing_adjustment = False

        if not self.gp_timing_adjustment:
            return False

        max_overflow = None
        for at_name in self.params.gp_timing_adjustment_area_types:
            at_id = self.placedb.getAreaTypeIndexFromName(at_name)
            overflow = current_metric.overflow[at_id].item()
            max_overflow = (
                overflow if max_overflow is None else max(overflow, max_overflow)
            )
        assert max_overflow is not None

        if max_overflow > self.last_timing_adjustment_threshold:
            return False

        return True

    def timing_adjustment(self, current_metric, opt_iter):
        target_clk_period_ps = self.params.target_clock_period_ns * 1000
        pos = self.data_cls.pos[0]
        (
            max_dly,
            wns,
            tns,
            pin_covered,
            pin_setups,
            pin_required,
            arc_slacks,
            net_covered,
            net_avg_slacks,
            net_min_slacks,
            critical_path_on_net,
            critical_pins,
            critical_insts,
            arc_kappas,
            arc_deltas,
            net_max_multipliers,
            arc_delays,
        ) = self.op_cls.xarch_timing_op(
            pos, self.data_cls.net_weights, target_clk_period_ps
        )
        with torch.no_grad():
            if self.params.gp_timing_adjustment_scheme == "net-weighting":
                updated_net_weights = torch.exp(-net_min_slacks / max_dly)
                updated_net_weights.clamp_min_(1)
                updated_net_weights[
                    (net_covered == 0)
                ] = 1  # don't adjust nets that are not covered
                self.data_cls.net_weights *= updated_net_weights
            elif self.params.gp_timing_adjustment_scheme == "min-max":
                updated_net_weights = net_max_multipliers
                updated_net_weights[
                    net_covered == 0
                ] = 1  # don't adjust nets that are not covered
                self.data_cls.net_weights *= updated_net_weights
            else:
                assert False, "Unknown timing adjustment scheme"

        one_update_ratio = (updated_net_weights == 1).float().mean()
        nonone_update_mean = updated_net_weights[(updated_net_weights != 1)].mean()
        logger.info(
            "[Timing Adjustment] #{} at Iter {} (threshold = {:.03f}): max_dly={:.03f} ns, wns = {:.03f} ns, tns = {:.03f} ns, updated_net_weights(min/avg/max/one-ratio/nonone-mean): {:.03f} / {:.03f} / {:.03f} / {:.02f}% / {:.03f}".format(
                self.num_gp_timing_adjustment,
                opt_iter.iteration,
                self.last_timing_adjustment_threshold,
                max_dly / 1e3,
                wns / 1e3,
                tns / 1e3,
                updated_net_weights.min().item(),
                updated_net_weights.mean().item(),
                updated_net_weights.max().item(),
                one_update_ratio.item() * 100,
                nonone_update_mean.item(),
            )
        )

        self.last_timing_adjustment_iter = opt_iter
        self.last_timing_adjustment_threshold *= (
            self.params.gp_timing_adjustment_threshold_decay
        )
        self.num_gp_timing_adjustment += 1
        # self.reset_optimizer(opt_iter)
        return

    def timing_analysis(self, pos, opt_iter):
        target_clk_period_ps = self.params.target_clock_period_ns * 1000
        (
            max_dly,
            wns,
            tns,
            pin_covered,
            pin_setups,
            pin_required,
            arc_slacks,
            net_covered,
            net_avg_slacks,
            net_min_slacks,
            critical_path_on_net,
            critial_pins,
            critical_insts,
            arc_kappas,
            arc_deltas,
            net_max_multipliers,
            arc_delays,
        ) = self.op_cls.xarch_timing_op(
            pos, self.data_cls.net_weights, target_clk_period_ps
        )

        if debug_timing_flag:
            if not os.path.isdir(self.params.result_dir):
                pathlib.Path(self.params.result_dir).mkdir(parents=True, exist_ok=True)
            with open(
                os.path.join(
                    self.params.result_dir,
                    f"critical_pos-{self.params.benchmark_name}.txt",
                ),
                "w",
            ) as f:
                for i in range(len(critical_insts)):
                    xy = pos[critical_insts[i]]
                    f.write(f"{round(xy[0].item(), 2)} {round(xy[1].item(), 2)}\n")

        return max_dly, wns, tns

    def _confine_clock_region_condition(self, current_metric):
        if not self.params.confine_clock_region_flag:
            return False

        # Carry out clock region assignment after the area inflation are done.
        if self.gp_adjust_area is True:
            return False

        if self.num_confine_fence_region >= self.params.confine_fence_region_max_iters:
            self.confine_fence_region = False

        if not self.confine_fence_region:
            return False

        # Get area type index for LUT and FF
        lut_area_type_id = self.placedb.getAreaTypeIndexFromName("LUT")
        ff_area_type_id = self.placedb.getAreaTypeIndexFromName("FF")
        # Check the instance area adjustment conditions
        lut_overflow = current_metric.overflow[lut_area_type_id].item()
        ff_overflow = current_metric.overflow[ff_area_type_id].item()
        lut_ff_max_overflow = max(lut_overflow, ff_overflow)
        if lut_ff_max_overflow > self.params.confine_fence_region_overflow_threshold:
            return False
        else:
            return True

    def _confine_clock_region(self, opt_iter, metrics):
        with DeferredAction() as defer:
            self.confine_fence_region_stopwatch.start()

            current_stage_idx = int(self.num_confine_fence_region)

            self.num_confine_fence_region += 1
            preface_string = (
                "[BEGIN    ] >>>>>>>>> CLock Region Confinement Stage #%02d >>>>>>>>>"
                % (current_stage_idx)
            )
            postscript_string = (
                "[      END] <<<<<<<<< Fence Region Confinement Stage #%02d <<<<<<<<<"
                % (current_stage_idx)
            )
            logger.info(preface_string)
            defer(logger.info, postscript_string)
            if self.params.dump_before_clock_refinement_flag:
                logger.info("Dumping before clock region assignment...")
                self.dump(
                    "{}/{}.{:02d}.before_cnp.pklz".format(
                        self.params.result_dir,
                        self.params.design_name(),
                        current_stage_idx,
                    )
                )

            assert self.data_cls.movable_range[1] == self.data_cls.fixed_range[0]
            movable_insts_num = (
                self.data_cls.movable_range[1] - self.data_cls.movable_range[0]
            )

            pos = self.data_cls.pos[0]
            with torch.no_grad():
                # move inside boundary
                self.op_cls.move_boundary_op(pos)

            lb_corner_pos = pos - self.data_cls.inst_sizes_max / 2

            physical_pos = pos[
                self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
            ]
            if self.params.count_ck_cr:
                cr_ck_counts = self.op_cls.cr_ck_counter_op(physical_pos)
                logger.info(
                    "CR-CK count before clock region assignment: {}".format(
                        iarray2str(cr_ck_counts)
                    )
                )

            # TODO(Jing Mai, magic3007@pku.edu.cn): Leverage dedicated ways to update the exponents
            self.data_cls.fence_region_cost_parameters.energy_function_exponent = (
                2 * torch.ones(movable_insts_num, dtype=self.dtype, device=self.device)
            )

            # Fence region cost only make senses to movable instances.
            movable_inst_sizes_max = self.data_cls.inst_sizes_max[
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ]
            movable_inst_areas = (
                movable_inst_sizes_max[..., 0] * movable_inst_sizes_max[..., 1]
            )

            movable_and_fixed_inst_sizes_max = self.data_cls.inst_sizes_max[
                self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
            ]
            movable_and_fixed_inst_areas = (
                movable_and_fixed_inst_sizes_max[..., 0]
                * movable_and_fixed_inst_sizes_max[..., 1]
            )

            cnp_algo = self.params.clock_network_planner_algorithm
            if cnp_algo == "utplacefx":
                # TODO: add logic to decide when to assign clock region
                (
                    movable_and_fixed_inst_to_clock_region,
                    self.data_cls.clock_available_clock_region,
                    movable_and_fixed_inst_cr_avail_map,
                    movable_and_fixed_avail_crs,
                ) = self.op_cls.clock_network_planner_op.forward(
                    pos=lb_corner_pos[
                        self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
                    ]
                )
                self.data_cls.movable_inst_to_clock_region = (
                    movable_and_fixed_inst_to_clock_region[:movable_insts_num]
                )
                self.data_cls.movable_inst_cr_avail_map = (
                    movable_and_fixed_inst_cr_avail_map[:movable_insts_num]
                )
                self.movable_inst_avail_crs = movable_and_fixed_avail_crs[
                    :movable_insts_num
                ]
                assert self.data_cls.movable_inst_to_clock_region.shape == (
                    movable_insts_num,
                )
                assert movable_inst_areas.shape == (movable_insts_num,)
            elif cnp_algo == "utplace2":
                (
                    movable_and_fixed_inst_to_clock_region,
                    self.data_cls.clock_available_clock_region,
                    movable_inst_cr_avail_map,
                    movable_and_fixed_avail_crs,
                ) = self.op_cls.utplace2_cnp_op(
                    pos=pos[
                        self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
                    ],
                    areas=self.primordial_inst_areas[
                        self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
                    ],
                )
                self.data_cls.movable_inst_to_clock_region = (
                    movable_and_fixed_inst_to_clock_region[:movable_insts_num]
                )
                self.data_cls.movable_inst_cr_avail_map = movable_inst_cr_avail_map[
                    self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
                ]
                self.movable_inst_avail_crs = movable_and_fixed_avail_crs[
                    :movable_insts_num
                ]

            # area overflow check for each clock region.
            # inst_areas = self.data_cls.inst_sizes_max[..., 0] * self.data_cls.inst_sizes_max[..., 1]
            # cr_num_x, cr_num_y = self.placedb.clockRegionMapSize()
            # buffer = torch.zeros(cr_num_x * cr_num_y, dtype=torch.float64)
            # buffer.index_add_(dim=0,
            #                   index=torch.LongTensor(movable_and_fixed_inst_to_clock_region),
            #                   tensor=inst_areas
            #                   )
            assert (
                self.data_cls.clock_available_clock_region.shape[0]
                == self.data_cls.num_clocks
            )
            # TODO: only reset energy cost

            # Reset the fence region cost function
            self.op_cls.fence_region_op.reset_instance(
                inst_areas=movable_inst_areas,
                inst_sizes=movable_inst_sizes_max,
                inst_cr_avail_map=self.data_cls.movable_inst_cr_avail_map.to(
                    self.device
                ),
                energy_function_exponents=self.data_cls.fence_region_cost_parameters.energy_function_exponent.to(
                    self.device
                ),
            )

            # Reset the fence region checker operator
            self.op_cls.fence_region_checker_op.reset(
                fence_region_boxes=self.data_cls.fence_region_boxes,
                inst_sizes=movable_inst_sizes_max,
                inst_avail_crs=self.movable_inst_avail_crs,
            )

            # Reset the SSSR(single-site-single-resource) legalization operator
            self.op_cls.ssr_legalize_op.reset_honor_fence_region_constraints(True)
            self.op_cls.ssr_legalize_op.reset_clock_available_clock_region(
                self.data_cls.clock_available_clock_region
            )

            # Reset the direct legalization operator
            self.op_cls.direct_lg_op.reset_honor_fence_region_constraints(True)
            self.op_cls.direct_lg_op.reset_clock_available_clock_region(
                self.data_cls.clock_available_clock_region
            )

            self.op_cls.ism_dp_op.reset_honor_clock_constraints(True)
            self.op_cls.ism_dp_op.reset_clock_available_clock_region(
                self.data_cls.clock_available_clock_region
            )

            movable_pos = pos[
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ]
            displacement_arr = self.op_cls.fence_region_checker_op(inst_pos=movable_pos)
            num_total_movable_insts = (
                self.data_cls.movable_range[1] - self.data_cls.movable_range[0]
            )
            logger.info(
                "clock illegal instances: {}/{}, dist-max: {}, dist-non-zero-avg: {}".format(
                    (displacement_arr > 0).sum().item(),
                    num_total_movable_insts,
                    displacement_arr.max().item(),
                    displacement_arr[displacement_arr > 0].mean().item(),
                )
            )
            if self.params.count_ck_cr:
                physical_pos = pos[
                    self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
                ]
                cr_ck_counts = self.op_cls.cr_ck_counter_op(physical_pos)
                logger.info("CR-CK count: {}".format(iarray2str(cr_ck_counts)))

            if self.params.clock_network_planner_enable_moving:
                logger.info(
                    "Moving movable instances to the center of target clock regions..."
                )
                # TODO(Jing Mai, jingmai@pku.edu.cn): only move the illegal instances.
                pos.data[
                    self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
                ].copy_(
                    self.op_cls.inst_cr_mover_op(
                        self.data_cls.movable_inst_to_clock_region
                    )
                )
                displacement_arr = self.op_cls.fence_region_checker_op(
                    inst_pos=movable_pos
                )
                logger.info(
                    "clock illegal instances after moving instances: {}/{}, dist-max: {}".format(
                        (displacement_arr > 0).sum().item(),
                        num_total_movable_insts,
                        displacement_arr.max().item(),
                    )
                )
                if self.params.count_ck_cr:
                    cr_ck_counts = self.op_cls.cr_ck_counter_op(physical_pos)
                    logger.info(
                        "CR-CK count after moving instances: {}".format(
                            iarray2str(cr_ck_counts)
                        )
                    )
            else:
                logger.info(
                    "Do not move the movable instances to the center of target clock regions..."
                )
            # Create a new fence region placement model and optimizer
            self.model = FenceRegionPlaceModel(
                self.params, self.placedb, self.data_cls, self.op_cls
            ).to(self.device)

            self.optimizer = NesterovAcceleratedGradientOptimizer(
                self.parameters(),
                lr=0,
                obj_and_grad_fn=self.model.obj_and_grad_fn,
                constraint_fn=self.op_cls.move_boundary_op,
            )
            self.reset_fence_region_cost_parameters()
            # self.update_gamma(opt_iter, metrics[-1].overflow)
            # self.reset_lambdas(self.params.reset_lambda_param)
            # self.reset_step_size()
            # self.optimizer_initial_state = copy.deepcopy(self.optimizer.state_dict())
            # self.initialize_learning_rate(self.model, self.optimizer, 0.1)

            # logger.info("Dumping after clock region assignment...")
            # self.dump("{}/{}.{:02d}.after_cnp.pklz".format(
            #     self.params.result_dir, self.params.design_name(), current_stage_idx))
            self.confine_fence_region_stopwatch.stop()
            log_dict(
                logger.info,
                {
                    "Gross Fence region Confinement Elapsed Time(ms)": self.confine_fence_region_stopwatch.internals[
                        -1
                    ].get_gross_elapsed_time(
                        stopwatch.Stopwatch.TimeFormat.kMicroSecond
                    )
                    / 1000.0
                },
            )

    def _ssir_legalization_condition(self, current_metric):
        if self.last_ssr_legalize_iter != 0:
            return False

        if self.restore_best_solution_flag is True:
            return True

        if (
            self.params.confine_clock_region_flag
            and self.last_clock_assignment_iter is None
        ):
            return False

        # case I: not self.params.gp_adjust_area
        # case II: self.params.gp_adjust_area and self.num_gp_adjust_area and (not self.gp_adjust_area)
        # the second long condition ensures we do not perform SSR legalization right after inflation
        if not self.params.gp_adjust_area:
            return True

        ssir_legalization_threshold = self.params.ssir_legalization_threshold

        if (
            self.num_gp_adjust_area
            and (not self.gp_adjust_area)
            and torch.all(current_metric.overflow < ssir_legalization_threshold)
            and torch.any(
                self.data_cls.total_movable_areas[self.data_cls.ssr_area_types] > 0
            )
            and torch.all(
                current_metric.overflow[self.data_cls.ssr_area_types]
                < self.params.ssr_stop_overflow
            )
        ):
            return True

        return False

    def io_legalization_condition(self, opt_iter, metrics):
        try:
            if (
                not self.params.io_legalization_flag
                or len(metrics) < 100
                or self.num_io_legalization >= 1
            ):
                return False
            latest_metrics = metrics[-1]
            assert latest_metrics.at_avg_grad_norms is not None
            at_avg_grad_norms = latest_metrics.at_avg_grad_norms
            io_at_ids = [
                self.placedb.getAreaTypeIndexFromName(x)
                for x in self.params.io_at_names
            ]
            lut_ff_like_at_ids = [
                self.placedb.getAreaTypeIndexFromName(x)
                for x in self.params.lut_ff_like_at_names
            ]
            if len(io_at_ids) == 0:
                raise ValueError("Params `io_at_names` is incorrect.")
            if len(lut_ff_like_at_ids) == 0:
                raise ValueError("Params `lut_ff_like_at_names` is incorrect.")
            max_io_grad = max(np.array(at_avg_grad_norms)[io_at_ids])
            max_lut_ff_like_grad = max(np.array(at_avg_grad_norms)[lut_ff_like_at_ids])
            if max_io_grad > 1e2 * max_lut_ff_like_grad or len(metrics) > 300:
                return True
            else:
                return False
        except ValueError as e:
            logger.warning("Value EError: {0}".format(e))
            return False

    def io_legalization(self, opt_iter):
        logger.info("Start IO Legalization...")
        self.num_io_legalization += 1
        # self.plot(os.path.join(self.params.plot_dir, "iter%s_before_io_rough_legalizaion.bmp" % ('{:04}'.format(opt_iter.iteration))),
        #             opt_iter, plot_target_at_names=self.params.plot_target_at_names, filler_flag=True)
        pos = self.data_cls.pos[0]
        self.data_cls.io_pos_xyz = self.op_cls.io_legalization_op(pos)
        # lock IO instances(movable, fixed & filler)
        io_at_names = self.params.io_at_names
        io_at_ids = [self.placedb.getAreaTypeIndexFromName(x) for x in io_at_names]
        for io_at_id in io_at_ids:
            self.data_cls.inst_lock_mask[
                self.data_cls.area_type_inst_groups[io_at_id]
            ] = 1
            self.data_cls.area_type_lock_mask[io_at_id] = 1
        # reset optimizer
        self.reset_optimizer(opt_iter)
        # self.plot(os.path.join(self.params.plot_dir, "iter%s_after_io_rough_legalizaion.bmp" % ('{:04}'.format(opt_iter.iteration))),
        #             opt_iter, plot_target_at_names=self.params.plot_target_at_names, filler_flag=True)
        self.num_io_legalization += 1

    def _check_divergence(self, metrics):
        """
        Only work for clock-aware placement.
        1) HPWL is larger then 1.5 * HPWL right before clock region assignment.
        2) The overflow of DSP and RAM is larger then 2 * ssir_legalization_threshold, and is increasing in the past 50
            while the overflow of LUT & FF is smaller than confine_fence_region_overflow_threshold.
        :param metrics:
        :return:
        """
        assert self.last_clock_assignment_iter is not None

        if len(metrics) == 0:
            return False

        cur_metric = metrics[-1]

        def calc_weighted_hpwl(hpwl):
            return (
                hpwl[0] * self.params.wirelength_weights[0]
                + hpwl[1] * self.params.wirelength_weights[1]
            )

        if calc_weighted_hpwl(cur_metric.hpwl) > 1.5 * calc_weighted_hpwl(
            self.metric_before_clk_assignment.hpwl
        ):
            logger.warning(
                "Possible divergence detected: the HPWL(%.5E) increases too much compared with that right "
                "before clock region assignment(%.5E)."
                % (
                    calc_weighted_hpwl(cur_metric.hpwl).item(),
                    calc_weighted_hpwl(self.metric_before_clk_assignment.hpwl).item(),
                )
            )
            return True

        # Get area type index for LUT and FF
        lut_area_type_id = self.placedb.getAreaTypeIndexFromName("LUT")
        ff_area_type_id = self.placedb.getAreaTypeIndexFromName("FF")
        # Check the instance area adjustment conditions
        lut_overflow = cur_metric.overflow[lut_area_type_id].item()
        ff_overflow = cur_metric.overflow[ff_area_type_id].item()
        lut_ff_max_overflow = max(lut_overflow, ff_overflow)

        sliding_windows_len = 50

        past_ssir_overflow = torch.stack(
            list(
                map(
                    lambda x: x.overflow[self.data_cls.ssr_area_types],
                    metrics[
                        max(
                            self.last_clock_assignment_iter,
                            len(metrics) - sliding_windows_len,
                        ) :
                    ],
                )
            )
        )
        mean_past_ssir_overflow = past_ssir_overflow.mean(dim=0)
        # diff_past_ssir_overflow = past_ssir_overflow[1:] - past_ssir_overflow[:-1]

        if (
            lut_ff_max_overflow < self.params.confine_fence_region_overflow_threshold
            and torch.all(
                past_ssir_overflow > self.params.ssir_legalization_threshold * 2
            )
            and torch.all(
                cur_metric.overflow[self.data_cls.ssr_area_types]
                > mean_past_ssir_overflow
            )
        ):
            return True

        return False

    def _save_best_solution(self, metrics, cur_metric):
        """
        Save the best solution. Only work for clock-aware placement.
        save the solution satisfying the following conditions with minimum HPWL
        1) Do not diverge.
        2) overflow of LUT & FF <= stop_overflow
        3) overflow of DSP & RAM <= stop_overflow * 2
        4) both overflow and HPWL are better current best solution.
        :param cur_metric:
        :return:
        """

        if self._check_divergence(metrics):
            return False

        # Get area type index for LUT and FF
        lut_area_type_id = self.placedb.getAreaTypeIndexFromName("LUT")
        ff_area_type_id = self.placedb.getAreaTypeIndexFromName("FF")
        # Check the instance area adjustment conditions
        lut_overflow = cur_metric.overflow[lut_area_type_id].item()
        ff_overflow = cur_metric.overflow[ff_area_type_id].item()
        lut_ff_max_overflow = max(lut_overflow, ff_overflow)

        if lut_ff_max_overflow > self.params.stop_overflow * 1.1:
            return False

        if torch.any(
            cur_metric.overflow[self.data_cls.ssr_area_types]
            > self.params.stop_overflow * 2
        ):
            return False

        def calc_weighted_hpwl(hpwl):
            return (
                hpwl[0] * self.params.wirelength_weights[0]
                + hpwl[1] * self.params.wirelength_weights[1]
            )

        if self.best_pos_before_ck_ssir_lg is None or (
            lut_overflow
            < max(
                self.best_sol_metric.overflow[lut_area_type_id].item(),
                self.params.stop_overflow,
            )
            and ff_overflow
            < max(
                self.best_sol_metric.overflow[ff_area_type_id].item(),
                self.params.stop_overflow,
            )
            and calc_weighted_hpwl(self.best_sol_metric.overflow)
            > calc_weighted_hpwl(cur_metric.hpwl)
        ):
            logger.info(
                "Save best solution at iter %04d" % cur_metric.opt_iter.iteration
            )
            with torch.no_grad():
                self.best_pos_before_ck_ssir_lg = self.data_cls.pos[0].data.clone()
            self.best_sol_metric = cur_metric
            return True

        return False

    def timing_weighting_condition(self, opt_iter, metrics):
        if not self.params.timing_optimization_flag:
            return False

        if len(metrics) == 0:
            return False

        if (
            self.timing_optimization_counter
            >= self.params.maximum_timing_optimization_time
        ):
            return False

        cur_metric = metrics[-1]

        # Get area type index for LUT and FF
        lut_area_type_id = self.placedb.getAreaTypeIndexFromName("LUT")
        ff_area_type_id = self.placedb.getAreaTypeIndexFromName("FF")
        # Check the instance area adjustment conditions
        lut_overflow = cur_metric.overflow[lut_area_type_id].item()
        ff_overflow = cur_metric.overflow[ff_area_type_id].item()
        lut_ff_max_overflow = max(lut_overflow, ff_overflow)

        if self.timing_optimization_counter == 0 and lut_ff_max_overflow < 0.7:
            return True

        if self.timing_optimization_counter == 1 and lut_ff_max_overflow < 0.5:
            return True

        if self.timing_optimization_counter > 1 and lut_ff_max_overflow < 0.3:
            return True

        return False

    def report_timing(self, opt_iter, pin_slacks=None):
        logger.info("Start Reporting Timing...")
        tt = time.time()
        if pin_slacks is None:
            pos = self.data_cls.pos[0]
            with torch.no_grad():
                pin_delays, ignore_pin_masks = self.op_cls.estimate_delay_op(pos)
                (
                    pin_arrivals,
                    pin_requires,
                    pin_slacks,
                    ignore_net_masks,
                ) = self.op_cls.static_timing_op(pin_delays)
        worest_negative_slacks = torch.min(pin_slacks)
        total_negative_slacks = torch.sum(pin_slacks[pin_slacks < 0])
        logger.info(
            "Iter {} Timing Statistics: WNS: {:.6e}ns TNS: {:.6e}ns".format(
                opt_iter.iteration,
                worest_negative_slacks / 1e3,
                total_negative_slacks / 1e3,
            )
        )
        logger.info(
            "Reporting Timing end. Elapsed Time: {:2f}s".format(time.time() - tt)
        )

    def timing_weighting(self, opt_iter):
        logger.info("Start Timing Weighting...")
        self.timing_optimization_counter += 1
        pos = self.data_cls.pos[0]
        num_nets = self.placedb.numNets()
        with torch.no_grad():
            pin_delays, ignore_pin_masks = self.op_cls.estimate_delay_op(pos)
            (
                pin_arrivals,
                pin_requires,
                pin_slacks,
                ignore_net_masks,
            ) = self.op_cls.static_timing_op(pin_delays)
            pin_criticalitys = torch.pow(
                self.params.timing_criticality_alpha,
                (pin_slacks.max() - pin_slacks) / (pin_slacks.max() - pin_slacks.min()),
            )
            net_criticalitys = torch.scatter_add(
                input=pin_criticalitys.new_zeros((num_nets,)),
                dim=0,
                index=self.data_cls.net_pin_map.b2as.long(),
                src=pin_criticalitys,
            )
            count = torch.scatter_add(
                input=torch.zeros_like(net_criticalitys),
                dim=0,
                index=self.data_cls.net_pin_map.b2as.long(),
                src=torch.ones_like(pin_criticalitys),
            )
            net_criticalitys /= torch.clamp(count, min=1)
            net_criticalitys[ignore_net_masks] = net_criticalitys[
                ignore_net_masks == 0
            ].mean()
            self.report_timing(opt_iter=opt_iter, pin_slacks=pin_slacks)
            self.data_cls.net_weights *= net_criticalitys
            logger.info(
                "net_criticalitys(min/avg/max): {:f}/{:f}/{:f}".format(
                    net_criticalitys.min().item(),
                    net_criticalitys.mean().item(),
                    net_criticalitys.max().item(),
                )
            )
        # self.reset_optimizer(opt_iter)

    def forward(self):
        """@brief Top API to solve placement"""

        # if self.params.visualization_flag:
        #     dataset = os.path.basename(os.path.normpath(self.params.result_dir))
        #     experiment = dataset if self.params.experiment is None else self.params.experiment
        #     self.visualization_writer = PlacerStatisticsViewer(
        #         dataset=dataset,
        #         experiment=experiment,
        #         repo=self.params.repo)
        #     self.visualization_writer.setParams(vars(self.params))

        # create model
        self.model = PlaceModel(
            self.params, self.placedb, self.data_cls, self.op_cls
        ).to(self.device)

        def constraint_fn(pos):
            self.op_cls.move_boundary_op(pos)
            if self.params.align_carry_chain_flag:
                self.op_cls.chain_alignment_op(pos)

        # define optimizer
        self.optimizer = NesterovAcceleratedGradientOptimizer(
            self.parameters(),
            lr=0,
            obj_and_grad_fn=self.model.obj_and_grad_fn,
            constraint_fn=constraint_fn,
        )

        # defining evaluation ops
        eval_ops = {
            # "wirelength": self.op_cls.wirelength_op,
            # "density"   : self.op_cls.density_op,
            # "objective": self.model.obj_fn,
            "hpwl": self.op_cls.hpwl_op,
            "overflow": self.op_cls.normalized_overflow_op,
        }

        # initial iteration
        # TODO: what is the meaning of 0,0,0,0
        opt_iter = OptIter(0, 0, 0, 0, 0)

        # initialize optimization parameters
        metrics = []

        # initialize synthetic adjust area configuration options
        self.gp_adjust_area = self.params.gp_adjust_area
        self.gp_adjust_route_area = (
            self.params.gp_adjust_area and self.params.gp_adjust_route_area
        )
        self.gp_adjust_pin_area = (
            self.params.gp_adjust_area and self.params.gp_adjust_pin_area
        )
        self.gp_adjust_resource_area = (
            self.params.gp_adjust_area and self.params.gp_adjust_resource_area
        )
        self.gp_timing_adjustment = self.params.gp_timing_adjustment
        self.latest_timing_analysis_result = None
        self.num_gp_adjust_area = 0
        self.num_io_legalization = 0
        self.num_gp_timing_adjustment = 0

        # Initialize clock fence region options
        self.confine_fence_region = self.params.confine_clock_region_flag
        self.num_confine_fence_region = 0
        self.metric_before_clk_assignment = None

        self.data_cls.io_pos_xyz = None
        # global placement
        if self.params.global_place_flag:
            self.last_ssr_legalize_iter = 0
            self.last_area_inflation_iter = None
            self.last_clock_assignment_iter = None
            self.last_timing_adjustment_iter = None
            self.last_timing_adjustment_threshold = (
                self.params.gp_timing_adjustment_overflow_threshold
            )
            self.eta_update_counter = 0

            tt = time.time()
            if self.params.load_global_place_init_file:
                logger.info(
                    "load global placement initial file from {}".format(
                        self.params.load_global_place_init_file
                    )
                )
                self.load_gp(self.params.load_global_place_init_file)
                # self.metric_before_clk_assignment = self.initialize_params(eval_ops, opt_iter, set_random_pos=False)
                # self._confine_clock_region(opt_iter, metrics)
                # cur_metric = self.initialize_params(eval_ops, opt_iter, set_random_pos=False)
                # eval_ops['fence_region'] = self.op_cls.fence_region_op
            else:
                cur_metric = self.initialize_params(eval_ops, opt_iter)

            self.plot(
                os.path.join(
                    self.params.plot_dir,
                    "iter%s_initial.bmp" % ("{:04}".format(opt_iter.iteration)),
                ),
                opt_iter,
                plot_target_at_names=self.params.plot_target_at_names,
                filler_flag=False,
            )
            # logger.info("<initial metric>: " + str(cur_metric))
            # metrics.append(cur_metric)
            # the state must be saved before setting learning rate
            self.optimizer_initial_state = copy.deepcopy(self.optimizer.state_dict())
            self.initialize_learning_rate(self.model, self.optimizer, 0.1)

            # if self.params.load_global_place_init_file:
            #     # set this flag to trigger SSIR legalization routine.
            #     self.last_clock_assignment_iter = cur_metric.opt_iter.iteration
            #     self.gp_adjust_area = False
            #     self.num_gp_adjust_area = 1

            self.restore_best_solution_flag = False

            while opt_iter.iteration < 20 or not self.stop_condition(metrics):
                if self.last_clock_assignment_iter is None:
                    for opt_iter.iter_gamma in range(self.params.gamma_iters):
                        for opt_iter.iter_lambda in range(self.params.lambda_iters):
                            for opt_iter.iter_sub in range(self.params.sub_iters):
                                cur_metric = self.one_step(
                                    self.optimizer, eval_ops, opt_iter
                                )
                                if (
                                    self.params.plot_flag
                                    and opt_iter.iteration
                                    % self.params.plot_iteration_frequency
                                    == 0
                                ):
                                    self.plot(
                                        os.path.join(
                                            self.params.plot_dir,
                                            "iter%s.bmp"
                                            % ("{:04}".format(opt_iter.iteration)),
                                        ),
                                        opt_iter,
                                        plot_target_at_names=self.params.plot_target_at_names,
                                        filler_flag=True,
                                    )

                                if (
                                    self.params.gp_timing_analysis_flag
                                    and opt_iter.iteration
                                    % self.params.gp_timing_analysis_iters
                                    == 0
                                ):
                                    max_dly, wns, tns = self.timing_analysis(
                                        self.data_cls.pos[0], opt_iter
                                    )
                                    self.latest_timing_analysis_result = (
                                        max_dly,
                                        wns,
                                        tns,
                                    )
                                    logger.info(
                                        "[Timing Analysis] at Iter {}: max_dly={:.03f} ns, wns={:.03f} ns, tns={:.03f} ns".format(
                                            opt_iter.iteration,
                                            max_dly / 1e3,
                                            wns / 1e3,
                                            tns / 1e3,
                                        )
                                    )

                                opt_iter.iteration += 1
                                metrics.append(cur_metric)
                        self.update_lambdas(opt_iter)
                    # adjust instance areas
                    if self._gp_adjust_area_condition(metrics[-1]) is True:
                        self.plot(
                            os.path.join(
                                self.params.plot_dir,
                                "iter%s_before_area_adjustment_%d.bmp"
                                % (
                                    "{:04}".format(opt_iter.iteration),
                                    self.num_gp_adjust_area,
                                ),
                            ),
                            opt_iter,
                            plot_target_at_names=self.params.plot_target_at_names,
                            filler_flag=True,
                        )
                        self._gp_adjust_area(metrics[-1], opt_iter)
                        self.last_area_inflation_iter = cur_metric.opt_iter.iteration
                        continue

                    if self.timing_adjustment_condition(metrics[-1], opt_iter) is True:
                        self.timing_adjustment(metrics[-1], opt_iter)
                        self.last_timing_adjustment_iter = cur_metric.opt_iter.iteration

                    # clock region constraints
                    if self._confine_clock_region_condition(metrics[-1]) is True:
                        self.metric_before_clk_assignment = metrics[-1]
                        self.reset_optimizer(opt_iter)
                        self._confine_clock_region(opt_iter, metrics)
                        self.last_clock_assignment_iter = cur_metric.opt_iter.iteration
                        eval_ops["fence_region"] = self.op_cls.fence_region_op
                        # new_metric = self.initialize_params(
                        #     eval_ops, opt_iter, set_random_pos=False
                        # )
                        # logger.info("<metric after clock assignment>" + str(new_metric))
                        # metrics.append(new_metric)
                        self.optimizer_initial_state = copy.deepcopy(
                            self.optimizer.state_dict()
                        )
                        self.initialize_learning_rate(self.model, self.optimizer, 0.1)
                        self.plot(
                            os.path.join(
                                self.params.plot_dir,
                                "iter%s_after_ck_assignment.bmp"
                                % ("{:04}".format(opt_iter.iteration)),
                            ),
                            opt_iter,
                            plot_target_at_names=self.params.plot_target_at_names,
                            filler_flag=True,
                        )
                        self.best_pos_before_ck_ssir_lg = None
                        self.best_sol_metric = None
                        continue
                    self.update_gamma(opt_iter, metrics[-1].overflow)
                else:
                    for opt_iter.iter_eta in range(self.params.eta_iters):
                        for opt_iter.iter_gamma in range(self.params.gamma_ck_iters):
                            for opt_iter.iter_lambda in range(self.params.lambda_iters):
                                for opt_iter.iter_sub in range(self.params.sub_iters):
                                    cur_metric = self.one_step(
                                        self.optimizer, eval_ops, opt_iter
                                    )
                                    if (
                                        self.params.plot_flag
                                        and opt_iter.iteration
                                        % self.params.plot_iteration_frequency
                                        == 0
                                    ):
                                        self.plot(
                                            os.path.join(
                                                self.params.plot_dir,
                                                "iter%s.bmp"
                                                % ("{:04}".format(opt_iter.iteration)),
                                            ),
                                            opt_iter,
                                            filler_flag=True,
                                            plot_target_at_names=self.params.plot_target_at_names,
                                        )
                                    opt_iter.iteration += 1
                                    metrics.append(cur_metric)
                                    if self.restore_best_solution_flag is False:
                                        self._save_best_solution(metrics, cur_metric)
                            self.update_lambdas(opt_iter)
                        self.update_gamma(opt_iter, metrics[-1].overflow)
                    # self._update_eta()
                    # if self._check_divergence(metrics) is True:
                    #     assert self.best_pos_before_ck_ssir_lg is not None, "Can not find good enough solution"
                    #     logger.info("Roll back to the best solution")
                    #     with torch.no_grad():
                    #         self.data_cls.pos[0].data.copy_(self.best_pos_before_ck_ssir_lg)
                    #     self.restore_best_solution_flag = True

                # lookahead legalization for IOs
                if self.io_legalization_condition(opt_iter, metrics):
                    self.io_legalization(opt_iter)
                    continue
                # if self.timing_weighting_condition(opt_iter, metrics):
                #     self.timing_weighting(opt_iter)
                #     continue

                # lookahead legalization for single-site resources like DSP and RAM
                if self._ssir_legalization_condition(metrics[-1]):
                    if (
                        self.params.io_legalization_flag
                        and self.num_io_legalization == 0
                    ):
                        self.io_legalization(opt_iter)
                    if self.params.dump_before_ssir_legalization_flag:
                        logger.info("Dumping before SSIR legalization...")
                        self.dump(
                            "{}/{}.before_ssir.pklz".format(
                                self.params.result_dir, self.params.design_name()
                            )
                        )
                    logger.info("Legalize single-site resources")
                    self.op_cls.ssr_legalize_op.reset_honor_fence_region_constraints(
                        self.params.confine_clock_region_flag
                    )
                    if self.op_cls.ssr_abacus_legalize_op:
                        self.op_cls.ssr_abacus_legalize_op(self.data_cls.pos[0])
                    self.op_cls.ssr_legalize_op(self.data_cls.pos[0])
                    # lock the legalized instances for ssr_legalize_lock_iters
                    self.last_ssr_legalize_iter = opt_iter.iteration
                    self.reset_optimizer(opt_iter)

                    # need to apply the change of gradient to the optimizer
                    # with torch.no_grad():
                    #    for group in optimizer.param_groups:
                    #        for i in range(len(group['g_k'])):
                    #            group['g_k'][i].masked_fill_(self.data_cls.inst_lock_mask.view([-1, 1]), 0)
                    #            group['g_k_1'][i].masked_fill_(self.data_cls.inst_lock_mask.view([-1, 1]), 0)
                    #            for area_type, lock in enumerate(self.data_cls.area_type_lock_mask):
                    #                inst_ids = self.data_cls.area_type_inst_groups[area_type]
                    #                if lock and len(inst_ids):
                    #                    group['u_k'][i][inst_ids] = self.pos[0][inst_ids]
                    #                    group['v_k_1'][i][inst_ids] = self.pos[0][inst_ids]
                    #                    group['v_kp1'][i][inst_ids] = self.pos[0][inst_ids]
                    #            # I found masked_scatter cannot assign the data correctly
                    #            # group['u_k'][i].data.masked_scatter_(self.data_cls.inst_lock_mask.view([-1, 1]), self.pos[0])
                    #            ##group['v_k'][i].data.masked_scatter_(self.data_cls.inst_lock_mask.view([-1, 1]), self.pos[0])
                    #            # group['v_k_1'][i].data.masked_scatter_(self.data_cls.inst_lock_mask.view([-1, 1]), self.pos[0])
                    #            # group['v_kp1'][i].data.masked_scatter_(self.data_cls.inst_lock_mask.view([-1, 1]), self.pos[0])
                    if self.params.plot_flag:
                        self.plot(
                            os.path.join(
                                self.params.plot_dir,
                                "iter%s_ssir_lg.bmp"
                                % ("{:04}".format(opt_iter.iteration)),
                            ),
                            opt_iter,
                            filler_flag=False,
                            plot_target_at_names=self.params.plot_target_at_names,
                        )

            if self.params.gp_timing_analysis_flag:
                max_dly, wns, tns = self.timing_analysis(self.data_cls.pos[0], opt_iter)
                logger.info(
                    "[Timing Analysis] at Iter {}: max_dly={:.03f} ns, wns={:.03f} ns, tns={:.03f} ns".format(
                        opt_iter.iteration, max_dly / 1e3, wns / 1e3, tns / 1e3
                    )
                )

            if self.params.report_timing_flag:
                logger.info("Timing After Global Placement...")
                self.report_timing(opt_iter=opt_iter)

            # restore primordial instances size for legalization
            self.data_cls.inst_sizes.data.copy_(self.primordial_inst_sizes)
            # update data collections
            self.data_cls.inst_sizes_max.data.copy_(
                self.data_cls.inst_sizes.max(dim=1)[0]
            )
            self.data_cls.inst_areas.data.copy_(
                self.data_cls.inst_sizes[..., 0] * self.data_cls.inst_sizes[..., 1]
            )
            self.data_cls.total_movable_areas.data.copy_(
                self.data_cls.inst_areas[
                    self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
                ].sum(dim=0)
            )
            # reset density and overflow operator to update stretched sizes
            self.op_cls.density_op.reset()
            self.op_cls.overflow_op.reset()

            physical_pos = self.data_cls.pos[0][
                self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
            ]
            # self.dump("%s/%s.after_gp.pklz" % (self.params.result_dir, self.params.design_name()))
            if self.params.confine_clock_region_flag and self.params.count_ck_cr:
                cr_ck_counts = self.op_cls.cr_ck_counter_op(physical_pos)
                logger.info(
                    "CR-CK count after global placement: {}".format(
                        iarray2str(cr_ck_counts)
                    )
                )

            logger.info("global placement takes %.3f seconds" % (time.time() - tt))

        if (
            self.params.global_place_flag
            and self.params.dump_global_place_solution_flag
        ):
            self.dump(
                "%s/%s.gp.pklz" % (self.params.result_dir, self.params.design_name())
            )

        if (
            not self.params.global_place_flag
            and self.params.load_global_place_solution_file
        ):
            self.load(self.params.load_global_place_solution_file)

        # plot last iteration
        if self.params.plot_flag:
            self.plot(
                os.path.join(
                    self.params.plot_dir,
                    "iter%s.bmp" % ("{:04}".format(opt_iter.iteration)),
                ),
                opt_iter,
                filler_flag=False,
                plot_target_at_names=self.params.plot_target_at_names,
            )
            if (
                self.params.plot_fence_region_flag
                and self.data_cls.fence_region_boxes is not None
                and self.data_cls.movable_inst_to_clock_region is not None
            ):
                self.plot_cr(
                    os.path.join(
                        self.params.plot_dir,
                        "iter%s_cr.bmp" % ("{:04}".format(opt_iter.iteration)),
                    )
                )
        # This block is for testing loading gp results and doing cr assignment on it
        # if self.params.cr_gen_after_gp:
        #     movable_insts_num = self.data_cls.movable_range[1] - self.data_cls.movable_range[0]
        #     movable_inst_sizes_max = self.data_cls.inst_sizes_max[
        #                              self.data_cls.movable_range[0]:self.data_cls.movable_range[1]]
        #     movable_inst_areas = movable_inst_sizes_max[..., 0] * movable_inst_sizes_max[..., 1]
        #
        #     movable_and_fixed_inst_sizes_max = self.data_cls.inst_sizes_max[
        #                                        self.data_cls.movable_range[0]:self.data_cls.fixed_range[1]]
        #     movable_and_fixed_inst_areas = (
        #                 movable_and_fixed_inst_sizes_max[..., 0] * movable_and_fixed_inst_sizes_max[..., 1])
        #     pos = self.data_cls.pos[0]
        #     lb_corner_pos = pos - self.data_cls.inst_sizes_max / 2
        #
        #     cnp_algo = self.params.clock_network_planner_algorithm
        #     if cnp_algo == "utplacefx":
        #         # TODO: add logic to decide when to assign clock region
        #         (movable_and_fixed_inst_to_clock_region,
        #          self.data_cls.clock_available_clock_region,
        #          movable_and_fixed_inst_cr_avail_map,
        #          movable_and_fixed_avail_crs) = self.op_cls.clock_network_planner_op.forward(
        #             pos=lb_corner_pos[self.data_cls.movable_range[0]:self.data_cls.fixed_range[1]]
        #         )
        #         self.data_cls.movable_inst_to_clock_region = movable_and_fixed_inst_to_clock_region[:movable_insts_num]
        #         self.data_cls.movable_inst_cr_avail_map = movable_and_fixed_inst_cr_avail_map[:movable_insts_num]
        #         self.movable_inst_avail_crs = movable_and_fixed_avail_crs[:movable_insts_num]
        #         assert self.data_cls.movable_inst_to_clock_region.shape == (movable_insts_num,)
        #         assert movable_inst_areas.shape == (movable_insts_num,)
        #     elif cnp_algo == "utplace2":
        #         (movable_and_fixed_inst_to_clock_region,
        #          self.data_cls.clock_available_clock_region, movable_inst_cr_avail_map,
        #          movable_and_fixed_avail_crs) = self.op_cls.utplace2_cnp_op(
        #             pos=pos[self.data_cls.movable_range[0]:self.data_cls.fixed_range[1]],
        #             areas=self.primordial_inst_areas[self.data_cls.movable_range[0]:self.data_cls.fixed_range[1]])
        #         self.data_cls.movable_inst_to_clock_region = movable_and_fixed_inst_to_clock_region[:movable_insts_num]
        #         self.data_cls.movable_inst_cr_avail_map = movable_inst_cr_avail_map[
        #                                                   self.data_cls.movable_range[0]:self.data_cls.movable_range[1]]
        #         self.movable_inst_avail_crs = movable_and_fixed_avail_crs[:movable_insts_num]
        #     else:
        #         logger.fatal("Unsupported clock network planner algorithm: ", cnp_algo)
        #
        #     self.data_cls.movable_inst_to_clock_region = movable_and_fixed_inst_to_clock_region[:movable_insts_num]
        #     self.op_cls.ssr_legalize_op.reset_fence_region_boxes(self.data_cls.clock_available_clock_region)
        #     self.op_cls.ssr_legalize_op.reset_honor_fence_region_constraints(True)
        #     self.op_cls.ssr_legalize_op(self.data_cls.pos[0])

        legality_check_done = False
        # legalization placement
        pos = self.data_cls.pos[0]
        physical_pos = pos[
            self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
        ]
        if self.params.confine_clock_region_flag and self.params.count_ck_cr:
            cr_ck_count = self.op_cls.cr_ck_counter_op(physical_pos)
            logger.info(
                "CR-CK Count before direct legalization: {}".format(
                    iarray2str(cr_ck_count)
                )
            )
        if self.params.legalize_flag:
            tt = time.time()
            if self.params.carry_chain_legalization_flag:
                assert self.data_cls.io_pos_xyz is not None
                pos_xyz = self.data_cls.io_pos_xyz.to(self.device).to(self.dtype)
                with torch.no_grad():
                    pos_xyz[:, :2].data.copy_(pos[: self.data_cls.movable_range[1]])
                logger.info("Start Carry Chain Legalization...")
                self.op_cls.chain_legalization_op(pos_xyz)
                self.op_cls.masked_direct_lg_op(pos_xyz)
            else:
                if self.params.confine_clock_region_flag:
                    self.op_cls.direct_lg_op.reset_honor_fence_region_constraints(
                        self.params.confine_clock_region_flag
                    )
                    self.op_cls.direct_lg_op.reset_clock_available_clock_region(
                        self.data_cls.clock_available_clock_region
                    )
                # legalize LUTs and FFs
                if self.params.confine_clock_region_flag:
                    (
                        pos_xyz,
                        self.data_cls.half_column_available_clock_region,
                    ) = self.op_cls.direct_lg_op(pos)
                else:
                    pos_xyz = self.op_cls.direct_lg_op(pos)

            # apply solution
            loc_xyz = pos_xyz[
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ]
            self.data_cls.pos[0][
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ].data.copy_(loc_xyz[:, :2])
            self.data_cls.inst_locs_xyz[
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ].data.copy_(loc_xyz)

            # evaluate
            opt_iter.iteration += 1
            cur_metric = EvalMetric(self.params, copy.deepcopy(opt_iter))
            cur_metric.evaluate(self.data_cls, eval_ops, self.data_cls.pos[0])
            metrics.append(cur_metric)
            logger.info(cur_metric)
            if self.visualization_writer is not None:
                self.visualization_writer.recordMetric(cur_metric)
            if self.params.count_ck_cr:
                physical_pos = pos[
                    self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
                ]
                cr_ck_count = self.op_cls.cr_ck_counter_op(physical_pos)
                logger.info(
                    "CR-CK Count after direct legalization: {}".format(
                        iarray2str(cr_ck_count)
                    )
                )
            if not legality_check_done:
                if self.params.architecture_name == "xarch":
                    legal = self.op_cls.legality_check_op(
                        self.data_cls.inst_locs_xyz, arch="xarch"
                    )
                else:
                    legal = self.op_cls.legality_check_op(self.data_cls.inst_locs_xyz)
                if not legal:
                    logger.warning("Placement is not LEGAL")
                legality_check_done = True
            if self.params.gp_timing_analysis_flag or debug_timing_flag:
                max_dly, wns, tns = self.timing_analysis(self.data_cls.pos[0], opt_iter)
                logger.info(
                    "[Timing Analysis] after legalization: max_dly={:.03f} ns, wns={:.03f} ns, tns={:.03f} ns".format(
                        max_dly / 1e3, wns / 1e3, tns / 1e3
                    )
                )
            logger.info("legalization takes %.3f seconds" % (time.time() - tt))

        # plot legalization iteration
        if self.params.plot_flag:
            self.plot(
                os.path.join(
                    self.params.plot_dir,
                    "iter%s.lg.bmp" % ("{:04}".format(opt_iter.iteration)),
                ),
                opt_iter,
                filler_flag=False,
                plot_target_at_names=self.params.plot_target_at_names,
            )

        if self.params.legalize_flag and self.params.dump_legalize_solution_flag:
            self.dump(
                "%s/%s.lg.pklz" % (self.params.result_dir, self.params.design_name())
            )

        if not self.params.legalize_flag and self.params.load_legalize_solution_file:
            self.load(self.params.load_legalize_solution_file)
            legality_check_done = False

        # detailed placement
        if self.params.detailed_place_flag:
            tt = time.time()
            if not legality_check_done:
                legal = self.op_cls.legality_check_op(self.data_cls.inst_locs_xyz)
                if not legal:
                    logger.warning("Placement is not LEGAL")
                legality_check_done = True
            if self.params.confine_clock_region_flag:
                self.op_cls.ism_dp_op.reset_honor_clock_constraints(
                    self.params.confine_clock_region_flag
                )
                self.op_cls.ism_dp_op.reset_clock_available_clock_region(
                    self.data_cls.clock_available_clock_region
                )
                self.op_cls.ism_dp_op.reset_half_column_available_clock_region(
                    self.data_cls.half_column_available_clock_region
                )
            if self.params.io_legalization_flag:
                chain_at_name = self.params.carry_chain_at_name
                chain_at_id = self.placedb.getAreaTypeIndexFromName(chain_at_name)
                inst_ids = self.data_cls.area_type_inst_groups[chain_at_id]
                inst_ids = inst_ids[
                    torch.logical_and(
                        self.data_cls.movable_range[0] <= inst_ids,
                        inst_ids < self.data_cls.movable_range[1],
                    )
                ]
                fixed_mask = torch.zeros(
                    self.data_cls.inst_locs_xyz.shape[0],
                    dtype=torch.uint8,
                    device="cpu",
                    requires_grad=False,
                )
                fixed_mask[inst_ids] = 1
                self.op_cls.ism_dp_op.fixed_mask = fixed_mask
            loc_xyz = self.op_cls.ism_dp_op(self.data_cls.inst_locs_xyz)
            # apply solution
            pos_xyz = loc_xyz[
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ]
            self.data_cls.pos[0][
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ].data.copy_(pos_xyz[:, :2])
            # convert to lower left
            self.data_cls.inst_locs_xyz[
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ].data.copy_(pos_xyz)
            # evaluate
            opt_iter.iteration += 1
            cur_metric = EvalMetric(self.params, copy.deepcopy(opt_iter))
            cur_metric.evaluate(self.data_cls, eval_ops, self.data_cls.pos[0])
            metrics.append(cur_metric)
            legality_check_done = False
            if not legality_check_done:
                legal = self.op_cls.legality_check_op(self.data_cls.inst_locs_xyz)
                if not legal:
                    logger.warning("Placement is not LEGAL")
                legality_check_done = True
            if self.params.gp_timing_analysis_flag:
                max_dly, wns, tns = self.timing_analysis(self.data_cls.pos[0], opt_iter)
                logger.info(
                    "[Timing Analysis] after detailed placement: max_dly={:.03f} ns, wns={:.03f} ns, tns={:.03f} ns".format(
                        max_dly / 1e3, wns / 1e3, tns / 1e3
                    )
                )
            logger.info("detailed placement takes %.3f seconds" % ((time.time() - tt)))
            if self.params.report_timing_flag:
                logger.info("Timing After Detailed Placement...")
                self.report_timing(opt_iter=opt_iter)

        # return all metrics
        return metrics

    def initialize_params(self, eval_ops, opt_iter, set_random_pos=True):
        """@brief initialize nonlinear placement parameters"""
        pos = self.data_cls.pos[0]

        # random initial placement
        if set_random_pos:
            self.op_cls.random_pos_op(pos)

        self.init_pos = pos.data.clone()

        # evaluate overflow
        overflow = self.op_cls.normalized_overflow_op(pos)
        # update gamma
        self.update_gamma(opt_iter, overflow)

        # initialize density weights
        self.initialize_lambdas()

        # initialize step size
        self.initialize_step_size()

        cur_metric = EvalMetric(self.params, copy.deepcopy(opt_iter))
        cur_metric.evaluate(self.data_cls, eval_ops, pos)

        return cur_metric

    def debug_at_grad(self, pos):
        _, grad, _ = self.model.obj_and_grad_fn(self.op_cls.move_boundary_op(pos))
        for at_type in range(self.data_cls.num_area_types):
            at_grad = grad[self.data_cls.area_type_inst_groups[at_type]]
            if at_grad.size()[0] > 0:
                avg_at_grad_norm = at_grad.norm(dim=1).mean()
                logger.info("at_type: %d, avg-norm: %g", at_type, avg_at_grad_norm)

    def one_step(self, optimizer, eval_ops, opt_iter):
        """@brief forward one step"""
        pos = self.data_cls.pos[0]
        cur_metric = EvalMetric(self.params, copy.deepcopy(opt_iter))
        cur_metric.gamma = self.data_cls.gamma.gamma.data
        cur_metric.lambdas = self.data_cls.multiplier.lambdas.data
        cur_metric.step_size = self.data_cls.multiplier.t.item()
        if (
            self.num_confine_fence_region is not None
            and self.num_confine_fence_region > 0
        ):
            cur_metric.eta = self.data_cls.fence_region_cost_parameters.eta
        # move any out-of-bound cell back to placement region
        self.op_cls.move_boundary_op(pos)

        optimizer.zero_grad()

        # one descent step
        tt = time.time()
        optimizer.step(cur_metric=cur_metric)
        logger.debug("optimizer step %.3f ms" % ((time.time() - tt) * 1000))

        cur_metric.evaluate(self.data_cls, eval_ops, pos)
        # nesterov has already computed the objective of the next step
        cur_metric.objective = optimizer.param_groups[0]["obj_k_1"][0].data.clone()

        if (
            self.params.count_ck_cr
            and self.num_confine_fence_region is not None
            and self.num_confine_fence_region > 0
        ):
            assert self.data_cls.movable_range[1] == self.data_cls.fixed_range[0]
            movable_pos = pos[
                self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
            ]
            displacement_arr = self.op_cls.fence_region_checker_op(inst_pos=movable_pos)
            cur_metric.ck_illegal_insts_num = (displacement_arr > 0).sum().item()
            cur_metric.cr_max_displacement = displacement_arr.max().item()
            cur_metric.movable_insts_num = (
                self.data_cls.movable_range[1] - self.data_cls.movable_range[0]
            )

            physical_pos = pos[
                self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
            ]
            # cur_metric.cr_ck_count = self.op_cls.cr_ck_counter_op(physical_pos)

            # cur_metric.cr_ck_count = None
        # actually reports the metric before step
        logger.info(cur_metric)
        if self.visualization_writer is not None:
            self.visualization_writer.recordMetric(cur_metric)
        return cur_metric

    def update_gamma(self, opt_iter, overflow):
        """
        @brief update gamma in wirelength model
        @param iteration optimization step
        @param overflow evaluated in current step
        """
        with torch.no_grad():
            # length of #area types
            gamma = self.data_cls.gamma.base * torch.pow(
                10, self.data_cls.gamma.k * overflow + self.data_cls.gamma.b
            )
            self.data_cls.gamma.gamma.data.fill_(
                (gamma * self.data_cls.gamma.weights).sum()
                / self.data_cls.gamma.weights.sum()
            )

    def _compute_relative_potential_energy(self, phi):
        """Compute the `"hat"` potential energy (phi).
            `"hat"` follows the notation in `elfPlace` paper and means tne relative values compared with initial states.
        :param phi: Potential energy vector.
        :return: Relative potential energy compared with initial potential energy.
        """
        phi_hat = self.op_cls.stable_zero_div_op(phi, self.data_cls.phi_0)
        return phi_hat

    def _compute_update_lambdas_method1(
        self, phi, density_term_grad, wirelength_grad, lambda_param
    ):
        with torch.no_grad():
            # Compute relative potential energy
            phi_hat = self._compute_relative_potential_energy(phi)

            # Get density gradient for each areas type
            density_term_grad_1l_norm_array = torch.zeros_like(phi, requires_grad=False)
            for area_type in range(self.data_cls.num_area_types):
                inst_ids = self.data_cls.area_type_inst_groups[area_type]
                if len(inst_ids):
                    density_term_grad_1l_norm_array[
                        area_type
                    ] += density_term_grad.view([-1, 2])[inst_ids].norm(p=1)

            # Compute the relative sub-gradient of density multiplier vector(lambda) w.r.t. instance position.
            # See `Equation(20)` in `elfPlace` paper.
            subgrad_hat = phi_hat + 0.5 * self.params.lambda_beta * phi_hat.pow(2)

            # See `Equation(27)` in `elfPlace` paper.
            wirelength_grad_l1_norm = wirelength_grad.norm(p=1)
            density_grad_dot_subgrad_hat = torch.dot(
                density_term_grad_1l_norm_array, subgrad_hat
            )
            coefficient = self.op_cls.stable_div_op(
                lambda_param * wirelength_grad_l1_norm, density_grad_dot_subgrad_hat
            )
            new_lambdas = subgrad_hat.mul(coefficient)
            return new_lambdas

    def _compute_update_lambdas_method2(
        self, phi, density_term_grad, wirelength_grad, lambda_param
    ):
        with torch.no_grad():
            # Compute relative potential energy
            phi_hat = self._compute_relative_potential_energy(phi)

            wl_grad_norm_arr = phi.new_zeros(self.placedb.numAreaTypes())
            density_grad_norm_arr = phi.new_zeros(self.placedb.numAreaTypes())
            for area_type in range(self.data_cls.num_area_types):
                inst_ids = self.data_cls.area_type_inst_groups[area_type]
                if len(inst_ids):
                    wl_grad_norm_arr[area_type] = wirelength_grad.view([-1, 2])[
                        inst_ids
                    ].norm(p=1)
                    density_grad_norm_arr[area_type] = density_term_grad.view([-1, 2])[
                        inst_ids
                    ].norm(p=1)

            # Compute the relative sub-gradient of density multiplier vector(lambda) w.r.t. instance position.
            # See `Equation(20)` in `elfPlace` paper.
            subgrad_hat = phi_hat + 0.5 * self.params.lambda_beta * phi_hat.pow(2)

            coefficient = self.op_cls.stable_div_op(
                lambda_param * wl_grad_norm_arr, density_grad_norm_arr
            )
            new_lambdas = subgrad_hat.mul(coefficient)
            return new_lambdas

    def initialize_lambdas(self):
        """Compute initial density weight (lambda)"""
        pos = self.data_cls.pos[0]

        # Compute wirelength
        wirelength = self.op_cls.wirelength_op(pos)

        # Gradient of wirelength w.r.t. instance position.
        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        # Backup initial wirelength
        self.data_cls.wl_0 = wirelength.data.clone()

        # Compute potential energy vector
        phi = self.op_cls.density_op(pos)

        # Backup initial potential energy
        self.data_cls.phi_0 = phi.data.clone()

        # According to Equation (12) elfplace's paper
        # this setting will make 1/2 * cs * phi^2 = phi, when phi = 10^-3 * phi_0
        self.data_cls.multiplier.cs = self.op_cls.stable_zero_div_op(
            self.params.lambda_beta, self.data_cls.phi_0.data
        )

        # compute density term
        density = (phi + 0.5 * self.data_cls.multiplier.cs * phi.pow(2)).sum()

        # Get density gradient
        if pos.grad is not None:
            pos.grad.zero_()
        density.backward()
        density_grad = pos.grad.clone()

        # Backup initial density term
        self.data_cls.density_0 = density.data.clone()

        self.data_cls.multiplier.lambdas = self._compute_update_lambdas_method1(
            phi=phi,
            density_term_grad=density_grad,
            wirelength_grad=wirelength_grad,
            lambda_param=self.params.lambda_eta,
        )

        logger.info(
            "initial lambdas = %s" % (array2str(self.data_cls.multiplier.lambdas))
        )

    def reset_lambdas(self, lambda_param):
        """Reset the multipliers vector right after area adjustment.
        Redirect lambda to its current normalized sub-gradient with the scale determined by the gradient norm ratio
        between wire-length and electric density. Derived the `Equation(27)` in `elfPlace` paper.
        """
        pos = self.data_cls.pos[0]

        # Compute wirelength
        wirelength = self.op_cls.wirelength_op(pos)

        # Gradient of wirelength w.r.t. instance position.
        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        # Compute potential energy vector
        phi = self.op_cls.density_op(pos)

        # Compute density term in overall loss function
        density_term = (phi + 0.5 * self.data_cls.multiplier.cs * phi.pow(2)).sum()

        # Get density gradient
        if pos.grad is not None:
            pos.grad.zero_()
        density_term.backward()
        density_term_grad = pos.grad.clone()

        self.data_cls.multiplier.lambdas = self._compute_update_lambdas_method1(
            phi=phi,
            density_term_grad=density_term_grad,
            wirelength_grad=wirelength_grad,
            lambda_param=lambda_param,
        )

        logger.info(
            "Reset lambdas = %s" % (array2str(self.data_cls.multiplier.lambdas))
        )

    def reset_fence_region_cost_parameters(self):
        pos = self.data_cls.pos[0]

        # Compute wirelength
        wirelength = self.op_cls.wirelength_op(pos)

        # Gradient of wirelength w.r.t. instance position.
        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        # Compute potential energy vector
        phi = self.op_cls.density_op(pos)

        # Compute density term in overall loss function
        cs = self.data_cls.multiplier.cs
        density = (phi + 0.5 * cs * phi.pow(2)).sum()

        # Get density gradient
        if pos.grad is not None:
            pos.grad.zero_()
        density.backward()
        density_grad = pos.grad.clone()

        # Compute fence region cost. Note that fence region cost only makes senses to movable instances.
        movable_pos = pos[
            self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
        ]
        fence_region_cost = self.op_cls.fence_region_op(movable_pos).sum()

        # Gradient of fence region cost w.r.t. instance positions.
        if pos.grad is not None:
            pos.grad.zero_()
        fence_region_cost.backward()
        fence_region_cost_grad = pos.grad.clone()
        fence_region_cost_grad[0 : self.data_cls.movable_range[0]] = 0
        fence_region_cost_grad[self.data_cls.movable_range[1] :] = 0

        # Update density penalty multipliers
        # self.data_cls.multiplier.lambdas = self._compute_update_lambdas_method1(
        #     phi=phi,
        #     density_term_grad=density_grad,
        #     wirelength_grad=wirelength_grad,
        #     lambda_param=self.params.lambda_eta,
        # )

        # Update clock region cost multipliers
        self.data_cls.fence_region_cost_parameters.eta = self._compute_initial_eta(
            wirelength_grad=wirelength_grad,
            density_grad=density_grad,
            fence_region_cost_grad=fence_region_cost_grad,
            eta_scale=self.params.confine_fence_region_eta_scale,
            eta_offset=self.params.confine_fence_region_eta_offset,
        )

    def _reset_eta(self):
        pos = self.data_cls.pos[0]

        # Compute wirelength
        wirelength = self.op_cls.wirelength_op(pos)

        # Gradient of wirelength w.r.t. instance position.
        if pos.grad is not None:
            pos.grad.zero_()
        wirelength.backward()
        wirelength_grad = pos.grad.clone()

        # Compute potential energy vector
        phi = self.op_cls.density_op(pos)

        # Compute density term in overall loss function
        cs = self.data_cls.multiplier.cs
        density = (phi + 0.5 * cs * phi.pow(2)).sum()

        # Get density gradient
        if pos.grad is not None:
            pos.grad.zero_()
        density.backward()
        density_grad = pos.grad.clone()

        # Compute fence region cost. Note that fence region cost only makes senses to movable instances.
        movable_pos = pos[
            self.data_cls.movable_range[0] : self.data_cls.movable_range[1]
        ]
        fence_region_cost = self.op_cls.fence_region_op(movable_pos)
        fence_region_cost = fence_region_cost.sum()

        if pos.grad is not None:
            pos.grad.zero_()
        fence_region_cost.backward()
        pos.grad[0 : self.data_cls.movable_range[0]] = 0
        pos.grad[self.data_cls.movable_range[1] :] = 0
        fence_region_cost_grad = pos.grad.clone()

        # Update clock region cost multipliers
        self.data_cls.fence_region_cost_parameters.eta = self._compute_initial_eta(
            wirelength_grad=wirelength_grad,
            density_grad=density_grad,
            fence_region_cost_grad=fence_region_cost_grad,
            eta_scale=self.params.confine_fence_region_eta_scale,
            eta_offset=self.params.confine_fence_region_eta_offset,
        )

    def _update_eta(self):
        with torch.no_grad():
            temp = self.op_cls.stable_div_op(
                self.data_cls.fence_region_cost.norm(p=1), self.data_cls.wirelength
            )
            rate = (
                0
                if temp == 0
                else torch.log(self.params.eta_update_scale * temp).clamp(min=0)
            )
            rate = 1 / (1 + rate)
            rate = rate * (
                self.params.eta_update_low
                + (self.params.eta_update_high - self.params.eta_update_low)
            )
            self.data_cls.fence_region_cost_parameters.eta *= rate

    def _compute_initial_eta(
        self,
        wirelength_grad,
        density_grad,
        fence_region_cost_grad,
        eta_scale,
        eta_offset,
    ):
        wirelength_grad_l2_norm = wirelength_grad.norm(p=2)
        density_grad_l2_norm = density_grad.norm(p=2)
        fence_region_cost_grad_l2_norm = fence_region_cost_grad.norm(p=2)
        return self.op_cls.stable_div_op(
            eta_scale * wirelength_grad_l2_norm,
            fence_region_cost_grad_l2_norm + eta_offset,
        )

    def initialize_step_size(self):
        self.data_cls.multiplier.t = (
            self.params.lambda_alpha_low - 1
        ) * self.data_cls.multiplier.lambdas.norm(p=2)
        logger.info("Initial step size = %g" % self.data_cls.multiplier.t.item())

    def reset_step_size(self):
        """Reset the step size after area adjustment.
        Derived the `Equation(28)` in `elfPlace` paper.
        """
        self.data_cls.multiplier.t = (
            self.params.lambda_alpha_high - 1
        ) * self.data_cls.multiplier.lambdas.norm(p=2)
        logger.info("Reset step size = %g" % self.data_cls.multiplier.t.item())

    def update_lambdas(self, opt_iter):
        """Update density weight (lambda) with sub-gradient method"""
        with torch.no_grad():
            phi = self.data_cls.phi
            phi_hat = self._compute_relative_potential_energy(phi)
            subgrad = phi_hat + 0.5 * self.params.lambda_beta * phi_hat.pow(2)
            subgrad_normalized = subgrad / subgrad.norm(p=2)
            subgrad_normalized.masked_fill_(self.data_cls.area_type_lock_mask, 0)
            # print("subgrad_normalized1 = ", subgrad_normalized)

            # Equation (21) in elfplace's paper
            # a heuristic to avoid too fast increase in lambdas
            # subgrad_normalized.clamp_(max=subgrad_normalized[subgrad_normalized.nonzero()].min() * 2)
            # print("subgrad_normalized2 = ", subgrad_normalized)
            self.data_cls.multiplier.lambdas += (
                self.data_cls.multiplier.t * subgrad_normalized
            )

            if self.last_clock_assignment_iter is not None:
                lambdas_upper_bounds = self.op_cls.stable_zero_div_op(
                    1.0, self.data_cls.multiplier.gd_gw_norm_ratio
                )
                self.data_cls.multiplier.lambdas = torch.min(
                    self.data_cls.multiplier.lambdas, lambdas_upper_bounds * 1e3
                )

            # Equation (22) in elfplace's paper
            # rate = torch.log(self.params.lambda_beta * phi_hat.norm(p=2) + 1)
            rate = torch.log(self.params.lambda_beta * phi_hat.norm(p=2)).clamp_(min=0)
            rate = rate / (1 + rate)
            rate = (
                rate * (self.params.lambda_alpha_high - self.params.lambda_alpha_low)
                + self.params.lambda_alpha_low
            )
            # rate = self.params.lambda_alpha_low
            self.data_cls.multiplier.t *= rate

    def initialize_learning_rate(self, model, optimizer, lr):
        """
        @brief Estimate initial learning rate by moving a small step.
        Computed as | x_k - x_k_1 |_2 / | g_k - g_k_1 |_2.
        @param x_k current solution
        @param lr small step
        """
        x_k = self.data_cls.pos[0]
        obj_k, g_k, grad_dicts = model.obj_and_grad_fn(x_k)
        x_k_1 = torch.autograd.Variable(x_k - lr * g_k, requires_grad=True)
        obj_k_1, g_k_1, grad_dicts = model.obj_and_grad_fn(x_k_1)

        learning_rate = (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)
        # update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate.data

    def stop_condition(self, metrics):
        """Stop condition"""
        # maximum iteration reached
        if (
            len(metrics) > 1
            and metrics[-1].opt_iter.iteration >= self.params.max_global_place_iters
        ):
            return True

            # do not stop if HPWL is still improving
        if len(metrics) > 1 and metrics[-1].hpwl.sum() < metrics[-2].hpwl.sum():
            return False

        # do not stop if some area types have not reached stop overflow
        if len(metrics):
            io_at_ids = (
                set()
                if not self.params.io_legalization_flag
                else set(
                    [
                        self.placedb.getAreaTypeIndexFromName(x)
                        for x in self.params.io_at_names
                    ]
                )
            )
            for area_type, ov in enumerate(metrics[-1].overflow):
                # do not concern the overflow of IOs
                if area_type in io_at_ids:
                    continue
                if (
                    len(self.data_cls.area_type_inst_groups[area_type]) > 10
                    and ov > self.params.stop_overflow
                ):
                    return False

        # do not stop if DSP/RAM have not been legalized for a long enough time
        if (
            self.last_ssr_legalize_iter + self.params.ssr_legalize_lock_iters
            >= metrics[-1].opt_iter.iteration
        ):
            return False

        # if (
        #     self.last_timing_adjustment_iter is not None
        #     and self.last_timing_adjustment_iter
        #     + self.params.timing_adjustment_lock_iters
        #     >= metrics[-1].opt_iter.iteration
        # ):
        #     return False

        return True

    def __call__(self):
        """@brief Alias of top API to solve placement"""
        if self.params.profile:
            profile.runctx("self.forward()", globals(), locals())
        else:
            self.forward()

    def plot_cr(self, filename):
        draw_place.draw_place_with_clock_region_assignments(
            self.data_cls.pos[0].data.cpu().numpy(),
            self.data_cls.inst_sizes_max.cpu().numpy(),
            self.data_cls.area_type_inst_groups,
            self.data_cls.diearea,
            self.data_cls.movable_range,
            self.data_cls.fence_region_boxes,
            self.data_cls.movable_inst_to_clock_region,
            filename,
            target_area_types=None,
        )

    def plot(self, filename, opt_iter, plot_target_at_names, filler_flag=True):
        """@brief Draw placement"""
        target_at_names = [
            self.placedb.getAreaTypeIndexFromName(x) for x in plot_target_at_names
        ]
        draw_place(
            self.data_cls.pos[0].data.cpu().numpy(),
            self.data_cls.inst_sizes_max.cpu().numpy(),
            self.data_cls.area_type_inst_groups,
            self.data_cls.diearea,
            self.data_cls.movable_range,
            self.data_cls.fixed_range,
            self.data_cls.filler_range if filler_flag else None,
            filename,
            opt_iter.iteration,
            target_area_types=target_at_names,
        )


        assert self.data_cls.movable_range[1] == self.data_cls.fixed_range[0]
        pos = self.data_cls.pos[0].detach()
        movable_and_fixed_pos = pos[
            self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
        ]
        movable_and_fixed_inst_sizes = self.data_cls.inst_sizes_max[
            self.data_cls.movable_range[0] : self.data_cls.fixed_range[1]
        ]
        if (
            self.params.plot_fence_region_flag
            and self.data_cls.fence_region_boxes is not None
            and self.data_cls.movable_inst_to_clock_region is not None
        ):
            draw_fence_regions(
                pos=movable_and_fixed_pos.cpu().numpy(),
                inst_sizes=movable_and_fixed_inst_sizes.cpu().numpy(),
                die_area=self.data_cls.diearea,
                movable_range=self.data_cls.movable_range,
                fixed_range=self.data_cls.fixed_range,
                inst_to_regions=self.data_cls.movable_inst_to_clock_region.cpu().numpy(),
                fence_region_boxes=self.data_cls.fence_region_boxes,
                filename=filename,
                target_fence_regions_indexes=np.arange(self.data_cls.num_clocks),
                iteration=opt_iter.iteration,
            )

    def dump(self, filename):
        """@brief Dump placement data"""
        logger.debug("write to %s" % filename)
        # TODO: fix
        inst_to_clock_indexes_bk, cr_map_bk = None, None
        try:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            if self.data_cls.cr_map:
                inst_to_clock_indexes_bk = self.data_cls.inst_to_clock_indexes
                cr_map_bk = self.data_cls.cr_map
            self.data_cls.inst_to_clock_indexes = None
            self.data_cls.cr_map = None
            self.data_cls.site_bboxes = None
            self.data_cls.chain_info_vec = None
            self.data_cls.ssr_chain_info_vec = None
            with gzip.open(filename, "wb") as f:
                pickle.dump(
                    {
                        "data_cls": self.data_cls,
                        "density_op": self.op_cls.density_op,
                        "overflow_op": self.op_cls.overflow_op,
                    },
                    f,
                )
        except Exception as e:
            logger.warning("Error occurs when dump data collection: {}".format(e))
        finally:
            if inst_to_clock_indexes_bk is not None:
                self.data_cls.inst_to_clock_indexes = inst_to_clock_indexes_bk
                self.data_cls.cr_map = cr_map_bk
            self.data_cls.site_bboxes = self.placedb.collectSiteBoxes()

    def load_gp(self, filename):
        """Load the intermediate state of global placement, including the instance positions

        :param filename:
        :return:
        """
        logger.debug("read from %s" % filename)
        try:
            with gzip.open(filename, "rb") as f:
                data = pickle.load(f)
                data_cls = data["data_cls"]
                self.data_cls.pos[0].data.copy_(data_cls.pos[0])
                self.data_cls.inst_locs_xyz.data.copy_(data_cls.inst_locs_xyz)
                self.data_cls.inst_sizes.data.copy_(data_cls.inst_sizes)
                self.data_cls.inst_sizes_max.data.copy_(data_cls.inst_sizes_max)
                self.data_cls.inst_areas.data.copy_(data_cls.inst_areas)
                self.data_cls.total_movable_areas.data.copy_(
                    data_cls.total_movable_areas
                )
                self.data_cls.gamma.gamma.data.copy_(data_cls.gamma.gamma)
                self.data_cls.wl_0 = data_cls.wl_0
                self.data_cls.phi_0 = data_cls.phi_0
                self.data_cls.density_0 = data_cls.density_0
                self.data_cls.multiplier = data_cls.multiplier
                self.data_cls.io_pos_xyz = data_cls.io_pos_xyz
                self.op_cls.density_op = data["density_op"]
                self.op_cls.overflow_op = data["overflow_op"]
                if self.data_cls.inst_sizes.is_cuda:
                    self.data_cls.gamma.gamma = self.data_cls.gamma.gamma.cuda()
                    self.data_cls.gamma.base = self.data_cls.gamma.base.cuda()
                    self.data_cls.gamma.weights = self.data_cls.gamma.weights.cuda()
                    for t in {self.op_cls.density_op, self.op_cls.overflow_op}:
                        for key, value in vars(t).items():
                            if isinstance(value, torch.Tensor):
                                t.__dict__[key] = value.cuda()
                            if isinstance(value, list):
                                for idx, sl in enumerate(value):
                                    if isinstance(sl, torch.Tensor):
                                        t.__dict__[key][idx] = sl.cuda()
                    self.op_cls.density_op.reset()
                    self.op_cls.overflow_op.reset()

        except Exception as e:
            logger.warning("Error occurs when reading initial solution: {}".format(e))

    def load(self, filename):
        """@brief Load placement data"""
        logger.debug("read from %s" % filename)
        try:
            with gzip.open(filename, "rb") as f:
                data = pickle.load(f)
                data_cls = data["data_cls"]
                if hasattr(data_cls, "movable_inst_to_clock_region"):
                    self.data_cls.movable_inst_to_clock_region = (
                        data_cls.movable_inst_to_clock_region
                    )
                if hasattr(data_cls, "clock_available_clock_region"):
                    self.data_cls.clock_available_clock_region = (
                        data_cls.clock_available_clock_region
                    )
                if hasattr(data_cls, "half_column_available_clock_region"):
                    self.data_cls.half_column_available_clock_region = (
                        data_cls.half_column_available_clock_region
                    )
                self.data_cls.pos[0].data.copy_(data_cls.pos[0])
                self.data_cls.inst_locs_xyz.data.copy_(data_cls.inst_locs_xyz)
                self.data_cls.io_pos_xyz = data_cls.io_pos_xyz
        except Exception as e:
            logger.warning("Error occurs when reading initial solution: {}".format(e))

    def dump_pl(self, filename, left_corner=False):
        """
        When left_corner is set to true, the bottom left corner of each instance is dumped.
        Otherwise,  the center is dumped.
        """
        logger.debug("Reading placement results from %s" % filename)

        with open(filename, "w") as fp:
            for i in range(
                self.data_cls.movable_range[0], self.data_cls.movable_range[1]
            ):
                xy = self.data_cls.pos[0][i]
                if left_corner:
                    r = xy - self.data_cls.inst_sizes_max[i] / 2
                else:
                    r = xy
                fp.write(
                    self.placedb.instName(i)
                    + " "
                    + str(r[0].item())
                    + " "
                    + str(r[1].item())
                    + "\n"
                )

    def load_pl(self, filename, left_corner=False):
        """
        When left_corner is set to true, it is assumed the
        file read stores the bottom left corner of each instance is dumped.
        Otherwise, the file stores the center of each instance.
        """

        logger.debug("Dumping placement results to %s" % filename)
        pos = self.data_cls.pos[0]
        with open(filename, "r") as fp:
            for l in fp:
                d = l.split()
                name = d[0]
                id_ = self.placedb.nameToInst(name)
                x = float(d[1])
                y = float(d[2])
                pos[id_][0] = x
                pos[id_][1] = y
        if left_corner:
            pos += self.data_cls.inst_sizes_max / 2

    def load_clock_available_clock_region(self, filename):
        logger.debug("Loading clock available clock region from %s" % filename)
        self.data_cls.clock_available_clock_region = torch.zeros(
            (
                self.data_cls.num_clocks,
                self.data_cls.clock_region_size[0] * self.data_cls.clock_region_size[1],
            ),
            dtype=torch.uint8,
        )
        with open(filename, "r") as fp:
            for l in fp:
                d = l.split()
                name = d[0]
                clk_net_id = self.placedb.nameToNet(name)
                clk_id = self.placedb.netIdToClockId(clk_net_id)
                assert clk_id >= 0 and clk_id < self.data_cls.num_clocks
                cr_id = 0
                for i in d[1:]:
                    assert i in ["0", "1"]
                    self.data_cls.clock_available_clock_region[clk_id][cr_id] = int(i)
                    cr_id += 1
                assert cr_id == self.data_cls.clock_available_clock_region.size()[1]

    def load_inst_clock(self, filename):
        logger.debug("Loading inst clock from %s" % filename)
        test_inst_to_clk = [-1 for i in range(self.placedb.numInsts())]
        with open(filename, "r") as fp:
            for l in fp:
                d = l.split()
                inst_name = d[0]
                inst_id = self.placedb.nameToInst(inst_name)
                if len(d) > 1:
                    for clk_name in d[1:]:
                        clk_net_id = self.placedb.nameToNet(clk_name)
                        clk_id = self.placedb.netIdToClockId(clk_net_id)
                        assert clk_id >= 0 and clk_id < self.data_cls.num_clocks
                        if test_inst_to_clk[inst_id] == -1:
                            test_inst_to_clk[inst_id] = set([clk_id])
                        else:
                            test_inst_to_clk[inst_id].add(clk_id)
        for i in range(self.placedb.numInsts()):
            for c in self.data_cls.inst_to_clock_indexes[i]:
                if not c in test_inst_to_clk[i]:
                    print("n", i, c)

    def load_clock_region_assignment(self, filename):
        logger.debug("Loading clock region assignment from %s" % filename)
        self.data_cls.movable_inst_to_clock_region = torch.zeros(
            self.data_cls.movable_range[1] - self.data_cls.movable_range[0],
            dtype=torch.int32,
        )
        with open(filename, "r") as fp:
            for l in fp:
                d = l.split()
                name = d[0]
                id_ = self.placedb.nameToInst(name)
                cr_x = float(d[1])
                cr_y = float(d[2])
                self.data_cls.movable_inst_to_clock_region[id_] = (
                    cr_x * self.data_cls.clock_region_size[1] + cr_y
                )

    def check_cr_assignment(self, loc_xyz):
        for i in range(self.data_cls.movable_range[0], self.data_cls.fixed_range[1]):
            if self.placedb.isInstClockSource(i):
                continue
            x = loc_xyz[i][0]
            y = loc_xyz[i][1]
            cr = self.placedb.xyToCrIndex(x, y)
            hc_id = self.placedb.xyToHcIndex(x, y)
            for c in self.data_cls.inst_to_clock_indexes[i]:
                if self.data_cls.clock_available_clock_region[c][cr] == 0:
                    logger.debug(
                        "Clock illegal instance %s at %i %i (cr %i)",
                        self.placedb.instName(i),
                        x,
                        y,
                        cr,
                    )
                if self.data_cls.half_column_available_clock_region[c][hc_id] == 0:
                    logger.debug(
                        "HC illegal instance %s at %i %i (hc %i)",
                        self.placedb.instName(i),
                        x,
                        y,
                        hc_id,
                    )

    def apply(self):
        """@brief update database"""
        self.placedb.apply(self.data_cls.inst_locs_xyz.data.view(-1).cpu().tolist())

    def write(self, filename):
        """@brief write to file"""
        self.apply()
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.placedb.writeBookshelfPl(filename)
