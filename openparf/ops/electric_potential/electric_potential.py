##
# @file   electric_potential.py
# @author Yibo Lin
# @date   Apr 2020
# @brief  electric potential according to e-place (http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf)
#

import os
import sys
import math
import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
import logging

from ..dct import dct
from ..dct.discrete_spectral_transform import getExactExpk as precomputeExpk

from ..density_map.density_map import DensityMapFunction, DensityOverflow

from . import electric_potential_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import electric_potential_cuda

import pdb

logger = logging.getLogger(__name__)


class ElectricOverflow(DensityOverflow):
    def __init__(self,
                 inst_sizes,
                 # inst_area_types,
                 initial_density_maps,
                 bin_map_dims,
                 area_type_mask,
                 xl,
                 yl,
                 xh,
                 yh,
                 movable_range,
                 filler_range,
                 fixed_range,
                 target_density,
                 smooth_flag,
                 deterministic_flag,
                 movable_macro_mask=None):
        self.movable_macro_mask = movable_macro_mask
        super(ElectricOverflow,
              self).__init__(inst_sizes=inst_sizes,
                             # inst_area_types=inst_area_types,
                             initial_density_maps=initial_density_maps,
                             bin_map_dims=bin_map_dims,
                             area_type_mask=area_type_mask,
                             xl=xl,
                             yl=yl,
                             xh=xh,
                             yh=yh,
                             movable_range=movable_range,
                             filler_range=filler_range,
                             fixed_range=fixed_range,
                             stretch_flag=True,
                             smooth_flag=smooth_flag,
                             deterministic_flag=deterministic_flag,
                             target_density=target_density)

    def reset(self):
        super(ElectricOverflow, self).reset()

        # detect movable macros and scale down the density to avoid halos
        # the definition of movable macros should be different according to algorithms
        self.num_movable_macros = 0
        if self.movable_macro_mask is not None:
            self.num_movable_macros = self.movable_macro_mask.sum().data.item()
            # masked_area_types = torch.masked_select(self.inst_area_types,
            #                                        self.movable_macro_mask)
            #masked_target_densities = self.target_density[masked_area_types]
            # self.inst_weights.masked_scatter_(self.movable_macro_mask,
            #                                  masked_target_densities)
            self.inst_weights[torch.nonzero(self.movable_macro_mask, as_tuple=True)].mul_(self.target_density)


class ElectrostaticSystem(object):
    """Compute electrostatic system given density maps
    """

    def __init__(self, bin_map_dims, bin_sizes, dtype, device, fast_mode,
                 xy_ratio):
        self.fast_mode = fast_mode
        # wirelength of x and y directions may have different weights
        # assume this is wx / wy
        self.xy_ratio = xy_ratio
        num_area_types = len(bin_map_dims)
        self.inv_wu2_plus_wv2 = [None] * num_area_types
        self.wu_by_wu2_plus_wv2_half = [None] * num_area_types
        self.wv_by_wu2_plus_wv2_half = [None] * num_area_types

        # dct2, idct2, idct_idxst, idxst_idct functions
        self.dct2 = [None] * num_area_types
        self.idct2 = [None] * num_area_types
        self.idct_idxst = [None] * num_area_types
        self.idxst_idct = [None] * num_area_types

        for area_type in range(num_area_types):
            # expk
            M = bin_map_dims[area_type][0].item()
            N = bin_map_dims[area_type][1].item()
            exact_expkM = precomputeExpk(M, dtype=dtype, device=device)
            exact_expkN = precomputeExpk(N, dtype=dtype, device=device)

            # init dct2, idct2, idct_idxst, idxst_idct with expkM and expkN
            self.dct2[area_type] = dct.Dct2(exact_expkM, exact_expkN)
            self.idct2[area_type] = dct.Idct2(exact_expkM, exact_expkN)
            self.idct_idxst[area_type] = dct.IdctIdxst(exact_expkM,
                                                       exact_expkN)
            self.idxst_idct[area_type] = dct.IdxstIdct(exact_expkM,
                                                       exact_expkN)

            # wu and wv
            wu = torch.arange(M, dtype=dtype,
                              device=device).mul_(2 * np.pi / M).view([M, 1])
            # scale wv because the aspect ratio of a bin may not be 1
            # it is equivalent to scale by bin width / bin height
            wv = torch.arange(N, dtype=dtype,
                              device=device).mul_(2 * np.pi / N).view([
                                  1, N
                              ]).mul_(bin_sizes[area_type][0] /
                                      bin_sizes[area_type][1] * self.xy_ratio)
            wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
            wu2_plus_wv2[0,
                         0] = 1.0  # avoid zero-division, it will be zeroed out
            self.inv_wu2_plus_wv2[area_type] = 1.0 / wu2_plus_wv2
            self.inv_wu2_plus_wv2[area_type][0, 0] = 0.0
            self.wu_by_wu2_plus_wv2_half[area_type] = wu.mul(
                self.inv_wu2_plus_wv2[area_type]).mul_(1. / 2)
            self.wv_by_wu2_plus_wv2_half[area_type] = wv.mul(
                self.inv_wu2_plus_wv2[area_type]).mul_(1. / 2)

    def forward(self, density_maps):
        """Compute potential, field, energy given density map;
        The energy here is actually total potential
        """
        num_area_types = len(density_maps)
        field_map_xs = [None] * num_area_types
        field_map_ys = [None] * num_area_types
        potential_maps = [None] * num_area_types
        energy = density_maps[0].new_zeros(num_area_types)
        for area_type in range(num_area_types):
            # compute auv
            auv = self.dct2[area_type].forward(density_maps[area_type])

            # compute field xi
            auv_by_wu2_plus_wv2_wu = auv.mul(
                self.wu_by_wu2_plus_wv2_half[area_type])
            auv_by_wu2_plus_wv2_wv = auv.mul(
                self.wv_by_wu2_plus_wv2_half[area_type])

            field_map_xs[area_type] = self.idxst_idct[area_type].forward(
                auv_by_wu2_plus_wv2_wu)
            field_map_ys[area_type] = self.idct_idxst[area_type].forward(
                auv_by_wu2_plus_wv2_wv)

            # energy = \sum q*phi
            # it takes around 80% of the computation time
            # so I will not always evaluate it
            if self.fast_mode:  # dummy for invoking backward propagation
                pass
            else:
                # compute potential phi
                # auv / (wu**2 + wv**2)
                # I changed auv to save memory
                auv_by_wu2_plus_wv2 = auv.mul_(
                    self.inv_wu2_plus_wv2[area_type])
                potential_maps[area_type] = self.idct2[area_type].forward(
                    auv_by_wu2_plus_wv2)
                # compute energy
                energy[area_type] = density_maps[area_type].mul(
                    potential_maps[area_type]).sum()
                #energy[area_type] = potential_maps[area_type].sum()
                # if area_type == 2:
                #    print("energy[2] = ", energy[area_type].item())

        return potential_maps, field_map_xs, field_map_ys, energy


class ElectricPotentialFunction(Function):
    """
    @brief compute electric potential according to e-place.
    """
    @staticmethod
    def forward(ctx, pos, inst_sizes, inst_weights,  # inst_area_types,
                bin_map_dims, bin_map_areas, area_type_mask,
                xl, yl, xh, yh, movable_range,
                filler_range, smooth_flag, deterministic_flag, density_maps, electrostatic_system):
        """Compute electric potential and energy given
        """
        density_maps = DensityMapFunction.forward(
            pos, inst_sizes, inst_weights,  # inst_area_types,
            bin_map_dims, area_type_mask, xl,
            yl, xh, yh, movable_range, filler_range, smooth_flag, deterministic_flag, density_maps)

        # convert from area to density
        num_area_types = len(density_maps)
        for area_type in range(num_area_types):
            density_maps[area_type].mul_(1.0 / bin_map_areas[area_type])

        potential_maps, field_map_xs, field_map_ys, energy = electrostatic_system.forward(
            density_maps)

        ctx.pos = pos
        ctx.inst_sizes = inst_sizes
        ctx.inst_weights = inst_weights
        #ctx.inst_area_types = inst_area_types
        ctx.bin_map_dims = bin_map_dims
        ctx.xl = xl
        ctx.yl = yl
        ctx.xh = xh
        ctx.yh = yh
        ctx.movable_range = movable_range
        ctx.filler_range = filler_range
        ctx.field_map_xs = field_map_xs
        ctx.field_map_ys = field_map_ys

        if pos.is_cuda:
            torch.cuda.synchronize()
        return energy

    @staticmethod
    def backward(ctx, grad_pos):
        """Compute electric force/gradient given field maps
        """
        if grad_pos.is_cuda:
            func = electric_potential_cuda.backward
        else:
            func = electric_potential_cpp.backward
        output = -func(grad_pos, ctx.pos, ctx.inst_sizes, ctx.inst_weights,
                       # ctx.inst_area_types,
                       ctx.field_map_xs, ctx.field_map_ys,
                       ctx.bin_map_dims, ctx.xl, ctx.yl, ctx.xh, ctx.yh,
                       ctx.movable_range, ctx.filler_range, len(ctx.bin_map_dims))

        if grad_pos.is_cuda:
            torch.cuda.synchronize()
        return (output, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)


class ElectricPotential(ElectricOverflow):
    """
    @brief Compute electric potential according to e-place
    """

    def __init__(self,
                 inst_sizes,
                 # inst_area_types,
                 initial_density_maps,
                 bin_map_dims,
                 area_type_mask,
                 xl,
                 yl,
                 xh,
                 yh,
                 movable_range,
                 filler_range,
                 fixed_range,
                 target_density,
                 smooth_flag,
                 deterministic_flag,
                 movable_macro_mask=None,
                 fast_mode=True,
                 xy_ratio=1):
        """
        @brief initialization
        Be aware that all scalars must be python type instead of tensors.
        Otherwise, GPU version can be weirdly slow.
        @param inst_size_x cell width array consisting of movable cells, fixed cells, and filler cells in order
        @param inst_size_y cell height array consisting of movable cells, fixed cells, and filler cells in order
        @param movable_macro_mask some large movable macros need to be scaled to avoid halos
        @param bin_center_x bin center x locations
        @param bin_center_y bin center y locations
        @param target_density target density
        @param xl left boundary
        @param yl bottom boundary
        @param xh right boundary
        @param yh top boundary
        @param bin_size_x bin width
        @param bin_size_y bin height
        @param num_movable_insts number of movable cells
        @param num_terminals number of fixed cells
        @param num_filler_insts number of filler cells
        @param padding bin padding to boundary of placement region
        @param deterministic_flag control whether to use deterministic routine
        @param smooth_flag whether smooth density map
        @param fast_mode if true, only gradient is computed, while objective computation is skipped
        """
        self.fast_mode = fast_mode
        self.xy_ratio = xy_ratio
        super(ElectricPotential,
              self).__init__(inst_sizes=inst_sizes,
                             # inst_area_types=inst_area_types,
                             initial_density_maps=initial_density_maps,
                             bin_map_dims=bin_map_dims,
                             area_type_mask=area_type_mask,
                             xl=xl,
                             yl=yl,
                             xh=xh,
                             yh=yh,
                             movable_range=movable_range,
                             filler_range=filler_range,
                             fixed_range=fixed_range,
                             target_density=target_density,
                             smooth_flag=smooth_flag,
                             deterministic_flag=deterministic_flag,
                             movable_macro_mask=movable_macro_mask)

    def reset(self):
        """ Compute members derived from input
        """
        super(ElectricPotential, self).reset()
        logger.info("regard %d cells as movable macros in global placement" %
                    self.num_movable_macros)

        self.electrostatic_system = ElectrostaticSystem(
            bin_map_dims=self.bin_map_dims,
            bin_sizes=self.bin_sizes,
            dtype=self.inst_sizes.dtype,
            device=self.inst_sizes.device,
            fast_mode=self.fast_mode,
            xy_ratio=self.xy_ratio)

    def forward(self, pos):
        if self.fixed_density_maps is None:
            self.fixed_density_maps = self.fixedForward(pos)

        density_maps = [x.clone() for x in self.fixed_density_maps]
        return ElectricPotentialFunction.apply(
            pos, self.inst_sizes_stretched,
            self.inst_weights,  # self.inst_area_types,
            self.bin_map_dims,
            self.bin_map_areas,
            self.area_type_mask,
            self.xl, self.yl, self.xh, self.yh,
            self.movable_range, self.filler_range, self.smooth_flag, self.deterministic_flag,
            density_maps, self.electrostatic_system)
