#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : data_collections.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.21.2020
# Last Modified Date: 10.20.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>

import logging
import numpy as np
import torch
from .metric import array2str, array2d2str
from ..custom_data.chain_info import chain_info_cpp
from ..custom_data.ssr_chain_info import ssr_chain_info_cpp
import pdb


class A2MultiBMap(object):
    """ A map recording nested vectors with element
    A maps to multile B, while each element B maps
    to a single A. It correspondes to vector<vector<B>>
    in C++. This data structure stores bidirectional
    mapping betweeen A and B.
    """

    def __init__(self, bs, b_starts, b2as):
        # two arrays for mapping from a to multiple bs
        # flat array of bs
        self.bs = bs
        # starting index of each row
        self.b_starts = b_starts
        # one array for mapping from b to a
        self.b2as = b2as

    def __repr__(self) -> str:
        return f"A2MultiBMap(\n bs={self.bs},\n b_starts={self.b_starts}, \n b2as={self.b2as})"


class Nested2DVector(object):
    def __init__(self, bs, b_starts):
        self.bs = bs
        self.b_starts = b_starts


class LagMultiplier(object):
    """ A class to wrap data needed to compute multipliers
    """

    def __init__(self):
        """ Subgradient descent to update lambda
            lambda_{k+1} = lambda_k + t_k * lambda_sub_k / |lambda_sub_k|
            lambda_sub = (phi + 0.5 * cs * phi^2) / phi_0
        """
        self.lambdas = None
        self.t = None
        self.cs = None
        self.gd_gw_gs_norm_ratio = None
        # params for multi-die
        self.psi = 0.0  
        self.psi_m = 0.0
        self.psi_v = 0.0
        self.psi_beta1 = 0.9
        self.psi_beta2 = 0.999
        self.psi_epsilon = 1e-8

    def __str__(self):
        """ convert to string
        """
        content = "t %g, cs %g\n" % (self.t, self.cs)
        content += "lambdas %s" % (array2str(self.lambdas))
        return content

    def __repr__(self):
        return self.__str__()


class FenceRegionCostParams(object):
    def __init__(self):
        self.eta = None
        self.energy_function_exponent = None

    def __str__(self):
        return "eta: %g, energy_function_exponent(mean): %g" % (self.eta, self.energy_function_exponent.mean().item())

    def __repr__(self):
        return self.__str__()

class SoftFloorGamma(object):
    """ A class to wrap gamma related data for the soft_floor function
    """

    def __init__(self):
        # base gamma
        self.base = None
        self.k0 = None
        self.b0 = None
        self.k1 = None
        self.b1 = None
        # weights for different area types; use wl_precond as weight.
        self.weights = None
        # real gamma value
        self.gamma = None

    def __str__(self):
        """ convert to string
        """
        content = "soft_floor_gamma:  %g\n" % (self.gamma)
        return content

    def __repr__(self):
        return self.__str__()
    
class WirelengthGamma(object):
    """ A class to wrap gamma related data
    """

    def __init__(self):
        # base gamma
        self.base = None
        self.k = None
        self.b = None
        # weights for different area types
        self.weights = None
        # real gamma value
        self.gamma = None

    def __str__(self):
        """ convert to string
        """
        content = "base %g, k %g, b %g, gamma %g\n" % (self.base, self.k,
                                                       self.b, self.gamma)
        content += "weights %s" % (array2str(self.weights))
        return content

    def __repr__(self):
        return self.__str__()


torch2numpy_type_map = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.uint8: np.uint8,
    torch.int64: np.int64
}

numpy2torch_type_map = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.uint8: torch.uint8,
    np.int64: torch.int64
}


class DataCollections(object):
    """ A collection of all data tensors
    """

    def __init__(self, params, placedb, dtype, device):
        if dtype in torch2numpy_type_map:  # a torch type
            ttype = dtype
            ntype = torch2numpy_type_map[dtype]
        else:  # a numpy type
            ttype = numpy2torch_type_map[dtype]
            ntype = dtype
        with torch.no_grad():
            self.diearea = placedb.diearea().tolist()
            self.total_area = placedb.diearea().area()
            # Assume the area types are sorted in a way that
            # unused area types are at the end.
            # This variable represents the area types with valid instances.
            # Later instances with invalid area types are ignored
            # when computing the density map
            self.num_area_types = placedb.numAreaTypes()
            # for area_type in range(placedb.numAreaTypes()):
            #    if placedb.totalMovableInstAreas()[area_type] > 0:
            #        assert self.num_area_types == area_type, \
            #                "area type must be sorted in a way that zero movable area should appear at the end"
            #        self.num_area_types = area_type + 1
            total_movable_areas = np.array(
                placedb.totalMovableInstAreas().tolist(), dtype=ntype)
            self.total_movable_areas = torch.from_numpy(
                total_movable_areas).to(device)
            # 1 for area type with movable types, 0 for without
            # record the area types we really care
            self.area_type_mask = (self.total_movable_areas > 0)
            total_fixed_areas = np.array(placedb.totalFixedInstAreas(),
                                         dtype=ntype)
            self.total_fixed_areas = torch.from_numpy(
                total_fixed_areas).to(device)
            self.movable_range = placedb.movableRange()
            self.fixed_range = placedb.fixedRange()
            self.filler_range = None
            # bin maps
            self.bin_map_dims = np.array(placedb.binMapDims().tolist(),
                                         dtype=np.int32)
            self.bin_map_sizes = np.array(placedb.binMapSizes().tolist(),
                                          dtype=ntype)
            self.initial_density_maps = [None] * placedb.numAreaTypes()
            self.total_filler_areas = np.zeros_like(total_movable_areas)
            self.filler_sizes = np.zeros([placedb.numAreaTypes(), 2],
                                         dtype=ntype)
            self.num_fillers = np.zeros(placedb.numAreaTypes(), dtype=np.int32)
            self.sll_flag = False
            self.sll_start_overflow = 0.9
            inst_sizes = np.array(placedb.instSizes().tolist(), dtype=ntype).reshape(
                [placedb.numInsts(), -1, 2])
            self.area_type_inst_groups = [
                np.array(x, dtype=np.int32) for x in placedb.areaTypeInstGroups().tolist()]
            self.num_insts = np.array(
                [len(x) for x in self.area_type_inst_groups], dtype=np.int32)
            # inst_area_types = np.array(placedb.instAreaTypes().tolist(), dtype=np.uint8)
            for area_type in range(placedb.numAreaTypes()):
                if total_movable_areas[area_type] > 0:
                    bin_capacity_map = np.array(
                        placedb.binCapacityMap(area_type).tolist(),
                        dtype=ntype)
                    bin_size = placedb.binMapSize(area_type)
                    bin_area = bin_size.product()
                    initial_density_map = bin_area - bin_capacity_map
                    assert bin_capacity_map.sum(
                    ) == self.total_area - initial_density_map.sum()
                    self.initial_density_maps[area_type] = torch.from_numpy(
                        initial_density_map.reshape(
                            self.bin_map_dims[area_type])).to(device)
                    total_placeable_area = self.total_area - initial_density_map.sum(
                    ) - total_fixed_areas[area_type]
                    self.total_filler_areas[area_type] = max(
                        total_placeable_area -
                        total_movable_areas[area_type], 0.0)
                    # indices of instances for this area type
                    inst_ids = self.area_type_inst_groups[area_type]
                    # define a hint filler size
                    filler_sizes = np.median(inst_sizes[inst_ids, area_type],
                                             axis=0)
                    # HARD CODED FF filler: make it consistent with elfplace
                    if area_type == 1:
                        filler_sizes = self.filler_sizes[0]
                    self.num_fillers[area_type] = int(
                        self.total_filler_areas[area_type] /
                        (filler_sizes[0] * filler_sizes[1]))
                    # compute filler area from #fillers
                    filler_area = self.total_filler_areas[
                        area_type] / self.num_fillers[area_type]
                    aspect_ratio = filler_sizes[1] / filler_sizes[0]
                    # actual filler size
                    self.filler_sizes[area_type][0] = np.sqrt(filler_area /
                                                              aspect_ratio)
                    self.filler_sizes[area_type][1] = np.sqrt(filler_area *
                                                              aspect_ratio)
            self.filler_range = (self.fixed_range[1],
                                 self.fixed_range[1] + self.num_fillers.sum())
            # centers for movable and fixed instances
            self.inst_locs_xyz = torch.tensor(placedb.instLocs().tolist(),
                                              dtype=ttype,
                                              device=device)
            self.inst_sizes = torch.zeros([self.filler_range[1], placedb.numAreaTypes(), 2],
                                          dtype=ttype,
                                          device=device)
            # self.inst_area_types = torch.zeros(self.filler_range[1],
            #                                   dtype=torch.uint8,
            #                                   device=device)
            self.inst_sizes[self.movable_range[0]:self.fixed_range[1]].copy_(
                torch.tensor(placedb.instSizes().tolist(),
                             dtype=ttype,
                             device=device).view(placedb.numInsts(), placedb.numAreaTypes(), 2))
            # self.inst_area_types[self.movable_range[0]:self.
            #                     fixed_range[1]].copy_(
            #                         torch.tensor(placedb.instAreaTypes(),
            #                                      dtype=torch.uint8,
            #                                      device=device))

            filler_bgn = self.filler_range[0]
            for area_type in range(placedb.numAreaTypes()):
                filler_end = filler_bgn + self.num_fillers[area_type]
                self.inst_sizes[filler_bgn:filler_end].zero_()
                self.inst_sizes[filler_bgn:filler_end, area_type] = torch.from_numpy(
                    self.filler_sizes[area_type])
                # self.inst_area_types[filler_bgn:filler_end] = area_type
                self.area_type_inst_groups[area_type] = np.concatenate(
                    [self.area_type_inst_groups[area_type], np.arange(filler_bgn, filler_end, dtype=np.int32)])
                filler_bgn = filler_end
            for area_type in range(placedb.numAreaTypes()):
                self.area_type_inst_groups[area_type] = torch.from_numpy(
                    self.area_type_inst_groups[area_type]).long().to(device)
            # a long type variable for torch functions
            # self.inst_area_types_long = self.inst_area_types.long()
            self.inst_areas = self.inst_sizes[..., 0] * self.inst_sizes[..., 1]
            # maximum instance sizes across all area types
            self.inst_sizes_max, _ = self.inst_sizes.max(dim=1)

            self.is_inst_luts = torch.tensor(placedb.isInstLUTs().tolist(),
                                             dtype=torch.uint8,
                                             device=device)
            self.is_inst_ffs = torch.tensor(placedb.isInstFFs().tolist(),
                                            dtype=torch.uint8,
                                            device=device)

            # Find the unique clock net for each instance.
            # If None, use `-1` to indicate the such instance is not connected to any clock net/signal.
            # Note that the primordial net indexes take all of nets into consideration, such as clock signal,
            # control signal(set/reset), control signal(clock enable), cascade signal and other signals.
            # To be efficient, `The clock net indexes` are relabeled and only consider clock nets.
            # NOTE: this is NOT a tensor! check placedb documentation!
            self.num_clocks = placedb.numClockNets()
            self.inst_to_clock_indexes = placedb.instToClocksCP()
            crMap = placedb.db().layout().clockRegionMap()
            self.cr_map = crMap
            self.clock_region_size = (crMap.width(), crMap.height())
            self.num_clock_regions = self.clock_region_size[0] * \
                self.clock_region_size[1]
            self.fence_region_boxes = []
            for i in range(self.clock_region_size[0]):
                for j in range(self.clock_region_size[1]):
                    bbox = crMap.at(i, j).bbox()
                    # Note that the indexes  in `bbox` is based on grid system with grid size (1, 1).
                    # Therefore, the geometry bounding box is [xl, yl, xh + 1, yh + 1]
                    self.fence_region_boxes.append(
                        [bbox.xl(), bbox.yl(), bbox.xh() + 1, bbox.yh() + 1])
            self.fence_region_boxes = torch.Tensor(
                self.fence_region_boxes).to(dtype).to(device)
            self.movable_inst_to_clock_region = None
            self.movable_inst_to_super_logic_region = None
            self.movable_inst_cnp_sll_optim_mask = None
            self.clock_available_clock_region = None
            self.movable_inst_cr_avail_map = None
            # Create a tensor that maps all instances, namely movable, fixed and filler instances, to their clock
            # clock net/signals. Although the filler instances are not connected to any clock net/signals virtually,
            # we define such tensor for the convenience of passing parameters to operators.
            # So as aforementioned, we use `-1` to indicate that filler instances is not connected to any clock
            # net/signal.
            assert self.movable_range[1] == self.fixed_range[0]

            self.inst_model_ids = torch.tensor(
                placedb.getInstModelIds().tolist(),
                dtype=torch.int32,
                device=device)

            # self.inst_resource_categories = torch.tensor(placedb.instResourceCategories().tolist(),
            #        dtype=torch.uint8,
            #        device=device)
            self.resource_categories = torch.tensor(placedb.resourceCategories().tolist(),
                                                    dtype=torch.uint8,
                                                    device=device)

            self.pin_offsets = torch.tensor(placedb.pinOffsets().tolist(),
                                            dtype=ttype,
                                            device=device)
            # no longer need pin offsets
            self.pin_offsets.zero_()
            self.pin_signal_directs = torch.tensor(placedb.pinSignalDirects().tolist(),
                                                   dtype=torch.uint8,
                                                   device=device)
            self.pin_signal_types = torch.tensor(placedb.pinSignalTypes().tolist(),
                                                 dtype=torch.uint8,
                                                 device=device)
            self.net_weights = torch.tensor(placedb.netWeights(),
                                            dtype=ttype,
                                            device=device)
            # bidirectional mapping between nodes and pins
            self.inst_pin_map = A2MultiBMap(
                torch.tensor(placedb.instPins().data().tolist(),
                             dtype=torch.int32,
                             device=device),
                torch.tensor(placedb.instPins().indexBeginData().tolist(),
                             dtype=torch.int32,
                             device=device),
                torch.tensor(placedb.pin2Inst().tolist(),
                             dtype=torch.int32,
                             device=device))
            # bidirectional mapping between nets and pins
            self.net_pin_map = A2MultiBMap(
                torch.tensor(placedb.netPins().data().tolist(),
                             dtype=torch.int32,
                             device=device),
                torch.tensor(placedb.netPins().indexBeginData().tolist(),
                             dtype=torch.int32,
                             device=device),
                torch.tensor(placedb.pin2Net().tolist(),
                             dtype=torch.int32,
                             device=device))

            self.net_mask = torch.ones(placedb.numNets(),
                                       dtype=torch.uint8,
                                       device=device)
            # net degree
            self.net_degrees = self.net_pin_map.b_starts[1:] - \
                self.net_pin_map.b_starts[:-1]
            assert torch.any(self.net_degrees <= 1) == False
            # ignore nets with large degrees
            self.net_mask_ignore_large = (
                self.net_degrees < 1000).type(torch.uint8).to(device)
            self.pin_mask = ((self.inst_pin_map.b2as >= self.fixed_range[0])
                             & (self.inst_pin_map.b2as < self.fixed_range[1]))
            self.inst_lock_mask = torch.zeros(len(self.inst_areas),
                                              dtype=torch.uint8,
                                              device=device)
            self.area_type_lock_mask = torch.zeros(self.num_area_types,
                                                   dtype=torch.uint8,
                                                   device=device)
            # CK/SR, CE infomation of instances
            # elfplace: src/GlobalPlacer.cpp
            # Also see ops/direct_lg/src/dl_solver.cpp
            # A #(numinsts, 2) tensor 0 is cksr id, 1 is ce id
            # TODO: why the fcuk it takes 5 second to do this step..
            # self.ff_ctrlsets, self.ff_ctrlsets_cksr_size, self.ff_ctrlsets_ce_size = self.compute_ff_ctrlsets(placedb, device)

            # site related
            self.site_bboxes = torch.tensor(
                placedb.collectFlattenSiteBoxes(), dtype=ttype, device=device).view([-1, 4])
            self.site_map_dim = (placedb.siteMapDim().width(),
                                 placedb.siteMapDim().height())
            self.site_lut_capacities = torch.zeros(
                len(self.site_bboxes), dtype=torch.int32, device=device)
            self.site_ff_capacities = torch.zeros(
                len(self.site_bboxes), dtype=torch.int32, device=device)
            for resource in range(placedb.numResources()):
                if placedb.isResourceLUT(resource):
                    self.site_lut_capacities += torch.tensor(placedb.collectSiteCapacities(resource).tolist(),
                                                             dtype=torch.int32, device=device)
                if placedb.isResourceFF(resource):
                    self.site_ff_capacities += torch.tensor(placedb.collectSiteCapacities(resource).tolist(),
                                                            dtype=torch.int32, device=device)
            # target density
            self.target_density = torch.tensor([params.target_density] *
                                               placedb.numAreaTypes())

            # compute wirelength preconditioning
            self.compute_wl_precond(params, placedb, ttype, device)

            # gamma to compute wirelength
            self.compute_base_gamma(params, placedb, ttype, device)

            # helpers to compute lambdas
            # according to elfplace
            self.multiplier = LagMultiplier()

            # helpers to computer fence region parameters, namely `eta`
            self.fence_region_cost_parameters = FenceRegionCostParams()

            # some temporary storage
            self.wirelength = None
            self.wasll = None
            self.density = None
            self.phi = None
            self.fence_region_cost = None

            # initial wirelength term
            self.wl_0 = None
            # initial energy
            self.phi_0 = None
            # initial density term
            self.density_0 = None
            # initial wirelength gradient norm
            self.wl_0_grad_norm = None
            # initial density gradient norm
            self.density_0_grad_norm = None

            # placeholder for single-site resource area types, e.g., DSP, RAM
            # will be initialized when building the SSR legalization op
            self.ssr_area_types = None

            # placeholder variables to optimize
            self.pos = None

            # carry chain alignment database
            if params.carry_chain_module_name:
                self.chain_info_vec = chain_info_cpp.MakeChainInfoVecFromPlaceDB(
                    placedb)
                cla_ids, lut_ids = chain_info_cpp.MakeNestedNewIdx(
                    self.chain_info_vec)
                self.chain_cla_ids = Nested2DVector(bs=torch.tensor(cla_ids.data().tolist(), dtype=torch.int32, device=device),
                                                    b_starts=torch.tensor(cla_ids.indexBeginData().tolist(), dtype=torch.int32, device=device))
                self.chain_lut_ids = Nested2DVector(bs=torch.tensor(lut_ids.data().tolist(), dtype=torch.int32, device=device),
                                                    b_starts=torch.tensor(lut_ids.indexBeginData().tolist(), dtype=torch.int32, device=device))
            else:
                self.chain_info_vec, self.chain_cla_ids, self.chain_lut_ids = None, None, None

            # ssr chain alignment database
            if params.abacus_lg_resource_name:
                self.ssr_chain_info_vec = ssr_chain_info_cpp.MakeChainInfoVecFromPlaceDB(
                    placedb
                )
                ssr_chain_ids = ssr_chain_info_cpp.MakeNestedNewIdx(
                    self.ssr_chain_info_vec
                )
                self.ssr_chain_ids = Nested2DVector(bs=torch.tensor(ssr_chain_ids.data().tolist(), dtype=torch.int32, device=device),
                                                    b_starts=torch.tensor(ssr_chain_ids.indexBeginData().tolist(), dtype=torch.int32, device=device))
            else:
                self.ssr_chain_info_vec, self.ssr_chain_ids = None, None

            self.num_slrX = placedb.numSlrX() if params.slr_aware_flag else 1 
            self.num_slrY = placedb.numSlrY() if params.slr_aware_flag else 1

            logging.info("Layout = (%f, %f) (%f, %f)" % (self.diearea[0], self.diearea[1],
                                                         self.diearea[2], self.diearea[3]))
            logging.info("total movable areas %s" %
                         (total_movable_areas.tolist()))
            logging.info("total fixed areas %s" %
                         (total_fixed_areas.tolist()))
            logging.info("total filler areas %s" %
                         (self.total_filler_areas.tolist()))
            logging.info("filler sizes %s" % (array2d2str(self.filler_sizes)))
            logging.info("#instances %s" % (self.num_insts.tolist()))
            logging.info("#fillers %s" % (self.num_fillers.tolist()))
            logging.info("movable range (%d, %d)" %
                         (self.movable_range[0], self.movable_range[1]))
            logging.info("fixed range (%d, %d)" %
                         (self.fixed_range[0], self.fixed_range[1]))
            logging.info("filler range (%d, %d)" %
                         (self.filler_range[0], self.filler_range[1]))
            logging.info("bin map dimensions %s" %
                         (self.bin_map_dims.tolist()))
            logging.info("bin sizes %s" % (array2d2str(self.bin_map_sizes)))
            logging.info("#clock nets: %d" % (self.num_clocks))

    @property
    def total_insts(self):
        """ Total number of instances including fillers
        """
        return self.filler_range[1]

    def compute_wl_precond(self, params, placedb, dtype, device):
        # length of #pins
        wl_precond_of_pins = torch.gather(input=self.net_weights /
                                          (self.net_degrees - 1).type(dtype),
                                          index=self.net_pin_map.b2as.long(),
                                          dim=0)
        # length of #insts
        # NOTE(Jing Mai): `torch_scatter_add` is nondeterministic when called on a CUDA tensor,
        # so we compute on CPU and then move to target devicendex_.
        self.wl_precond = torch.zeros(self.total_insts,
                                      dtype=dtype).scatter_add_(
            src=wl_precond_of_pins.cpu(),
            index=self.inst_pin_map.b2as.long().cpu(),
            dim=0).to(device)

        # self.wl_precond.clamp_(max=10)

    def compute_base_gamma(self, params, placedb, dtype, device):
        """
        @brief compute base gamma
        @param params parameters
        @param placedb placement database
        """
        self.gamma = WirelengthGamma()
        self.soft_floor_gamma = SoftFloorGamma()
        # truncate to valid area types
        self.gamma.base = 0.5 * params.base_gamma * torch.from_numpy(
            self.bin_map_sizes.sum(axis=1))
        self.gamma.base = self.gamma.base.to(device)
        self.soft_floor_gamma.base = 0.5 * 6 * torch.from_numpy(self.bin_map_sizes.sum(axis=1))
        self.soft_floor_gamma.base = self.soft_floor_gamma.base.to(device)
        # According to elfplace's implementation
        # Compute coeffcient for wirelength gamma updating
        # The basic idea is that we want to achieve
        #   gamma =  10 * base_gamma, if overflow = 1.0
        #   gamma = 0.1 * base_gamma, if overflow = target_overflow
        # We use function f(ovfl) = 10^(k * ovfl + b) to achieve the two above two points
        # So we want
        #   k + b = 1
        #   k * target_overflow + b = -1
        # Then we have
        #   k = 2.0 / (1 - target_overflow)
        #   b = 1.0 - k
        self.gamma.k = 2.0 / (1.0 - params.stop_overflow)
        self.gamma.b = 1.0 - self.gamma.k
        self.soft_floor_gamma.k0 = -2.59 / (self.sll_start_overflow - params.stop_overflow)
        self.soft_floor_gamma.b0 = -self.sll_start_overflow * self.soft_floor_gamma.k0

        # weights of gamma for different area types
        # truncate to valid area types
        self.gamma.weights = torch.zeros(placedb.numAreaTypes(), dtype=dtype)
        for area_type in range(placedb.numAreaTypes()):
            self.gamma.weights[area_type] = self.wl_precond[self.area_type_inst_groups[area_type]].sum(
            )
        self.gamma.weights = self.gamma.weights.to(device)
        self.soft_floor_gamma.weights = self.gamma.weights.clone()
        # self.gamma.weights = torch.zeros(placedb.numAreaTypes(),
        #                                 dtype=dtype,
        #                                 device=device).scatter_add_(
        #                                     src=self.wl_precond,
        #                                     index=self.inst_area_types_long,
        #                                     dim=0)[:self.num_area_types]
        self.gamma.gamma = torch.tensor(0, dtype=dtype, device=device)
        self.soft_floor_gamma.gamma = torch.tensor(5.0, dtype=dtype, device=device)
