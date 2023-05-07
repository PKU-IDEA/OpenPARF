##
# @file   density_map.py
# @author Yibo Lin
# @date   Apr 2020
# @brief  Compute density map
#

import torch
from torch.autograd import Function
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
import pdb

from . import density_map_cpp
from openparf import configure
if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import density_map_cuda


class DensityMapFunction(object):
    """Compute density map
    """
    @staticmethod
    def forward(pos, inst_sizes_stretched, inst_weights,
                bin_map_dims, area_type_mask,
                xl, yl, xh, yh, movable_range, filler_range,
                smooth_flag,
                deterministic_flag,
                density_maps):
        if pos.is_cuda:
            func = density_map_cuda.movableForward
        else:
            func = density_map_cpp.movableForward
        func(pos, inst_sizes_stretched, inst_weights,
             bin_map_dims, xl, yl, xh, yh, movable_range, filler_range,
             inst_sizes_stretched.shape[1],
             deterministic_flag,
             density_maps)

        # clear density map for area_type_mask = 0
        for area_type in range(len(density_maps)):
            if not area_type_mask[area_type]:
                density_maps[area_type].zero_()

        # Laplacian filter to smooth
        if smooth_flag:
            for area_type, density_map in enumerate(density_maps):
                if area_type_mask[area_type]:
                    density_maps[area_type] = F.avg_pool2d(density_map.view(
                        [1, 1, density_map.shape[0], density_map.shape[1]]), kernel_size=[3, 1], stride=1, padding=[1, 0])
                    density_maps[area_type] = density_maps[area_type].view(
                        [density_maps[area_type].shape[2], density_maps[area_type].shape[3]])

        return density_maps


class DensityMap(object):
    """
    @brief Compute density map for both movable and fixed cells.
    The density map for fixed cells is pre-computed.
    Each call will only compute the density map for movable cells.
    """

    def __init__(self, inst_sizes,
                 initial_density_maps,
                 bin_map_dims, area_type_mask,
                 xl, yl, xh, yh, movable_range, filler_range,
                 fixed_range, stretch_flag, smooth_flag, deterministic_flag):
        """
        @brief initialization
        @param inst_sizes cell (width, height) array consisting of movable cells, fixed cells, and filler cells in order
        @param inst_area_types cell area types
        @param bin_map_dims length of #bin maps, dimension (x, y) of bin maps for each area type
        @param area_type_mask if area type = 1, compute density map; otherwise, skip
        @param xl left boundary
        @param yl bottom boundary
        @param xh right boundary
        @param yh top boundary
        @param movable_range index range of movable cells
        @param filler_range index range of filler cells
        @param fixed_range index range of fixed cells
        @param stretch_flag whether stretch cell area
        @param smooth_flag whether perform smoothing
        @param deterministic_flag whether use deterministic mode
        """
        super(DensityMap, self).__init__()
        self.inst_sizes = inst_sizes
        assert len(self.inst_sizes.shape) == 3 and self.inst_sizes.shape[-1] == 2
        self.num_insts = self.inst_sizes.shape[0]
        #self.inst_area_types = inst_area_types
        self.initial_density_maps = initial_density_maps
        self.bin_map_dims = bin_map_dims
        self.area_type_mask = area_type_mask
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.movable_range = movable_range
        self.filler_range = filler_range
        self.fixed_range = fixed_range
        self.stretch_flag = stretch_flag
        self.smooth_flag = smooth_flag
        self.deterministic_flag = deterministic_flag
        self.fixed_density_maps = None

        self.reset()

    def reset(self):
        """Reset stretched sizes, and weights
        """
        self.bin_sizes = [None] * self.num_area_types
        self.bin_map_areas = [None] * self.num_area_types
        for area_type in range(self.num_area_types):
            M = self.bin_map_dims[area_type][0].item()
            N = self.bin_map_dims[area_type][1].item()
            self.bin_sizes[area_type] = [(self.xh - self.xl) / M,
                                         (self.yh - self.yl) / N]
            self.bin_map_areas[area_type] = self.bin_sizes[area_type][
                0] * self.bin_sizes[area_type][1]
        if self.stretch_flag:
            sqrt2 = math.sqrt(2)
            # clamped means stretch a cell to bin size
            # clamped = max(bin_size*sqrt2, inst_size)
            # ratio means the original area over the stretched area
            self.inst_sizes_stretched = self.inst_sizes.clone()
            self.inst_weights = self.inst_sizes.new_ones(self.num_insts, self.num_area_types)

            if self.inst_sizes.is_cuda:
                func = density_map_cuda.stretchForward
            else:
                func = density_map_cpp.stretchForward

            # only for movable and filler cells
            func(self.inst_sizes,
                 # self.inst_area_types,
                 self.bin_map_dims,
                 sqrt2, self.xl, self.yl, self.xh, self.yh,
                 self.movable_range, self.filler_range,
                 self.num_area_types,
                 self.inst_sizes_stretched,
                 self.inst_weights)
        else:
            self.inst_sizes_stretched = self.inst_sizes
            self.inst_weights = self.inst_sizes.new_ones(self.num_insts, self.num_area_types)

    @property
    def num_area_types(self):
        return len(self.bin_map_dims)

    def fixedForward(self, pos):
        """Compute density map for fixed cells
        """
        fixed_density_maps = [None] * self.num_area_types
        for area_type in range(self.num_area_types):
            M = self.bin_map_dims[area_type][0].item()
            N = self.bin_map_dims[area_type][1].item()
            if self.initial_density_maps[area_type] is None:
                fixed_density_maps[area_type] = pos.new_zeros([M, N])
            else:
                fixed_density_maps[area_type] = self.initial_density_maps[
                    area_type].clone()
        if pos.is_cuda:
            func = density_map_cuda.fixedForward
        else:
            func = density_map_cpp.fixedForward
        func(pos, self.inst_sizes_stretched,
             self.inst_weights,
             # self.inst_area_types,
             self.bin_map_dims,
             self.xl, self.yl, self.xh, self.yh,
             self.fixed_range,
             self.num_area_types,
             self.deterministic_flag,
             fixed_density_maps
             )
        # In case some fixed cells overlap with each other,
        # the area should not be repeatedly computed.
        # This is just a rough way to avoid such an issue.
        for area_type in range(len(fixed_density_maps)):
            fixed_density_maps[area_type].clamp_(
                max=self.bin_map_areas[area_type])
        return fixed_density_maps

    def forward(self, pos):
        """
        @brief API
        @param pos cell centers. The array consists of (x, y) locations of all cells
        """
        if self.fixed_density_maps is None:
            self.fixed_density_maps = self.fixedForward(pos)

        density_maps = [x.clone() for x in self.fixed_density_maps]
        DensityMapFunction.forward(
            pos, self.inst_sizes_stretched,
            self.inst_weights,
            self.bin_map_dims, self.area_type_mask,
            self.xl, self.yl, self.xh, self.yh, self.movable_range,
            self.filler_range, self.smooth_flag, self.deterministic_flag, density_maps)

        return density_maps

    def __call__(self, pos):
        """
        @brief Alias of top API
        """
        return self.forward(pos)


class DensityOverflow(DensityMap):
    """Compute overflows for different area types
    """

    def __init__(self, inst_sizes,
                 # inst_area_types,
                 initial_density_maps,
                 bin_map_dims, area_type_mask,
                 xl, yl, xh, yh, movable_range, filler_range,
                 fixed_range, stretch_flag, smooth_flag,
                 deterministic_flag,
                 target_density):
        self.target_density = target_density
        super(DensityOverflow,
              self).__init__(inst_sizes,
                             # inst_area_types,
                             initial_density_maps,
                             bin_map_dims, area_type_mask,
                             xl, yl, xh, yh, movable_range,
                             filler_range, fixed_range, stretch_flag, smooth_flag, deterministic_flag)

    def fixedForward(self, pos):
        """Fixed density map should have an upper limit of 1 even with target density
        """
        fixed_density_maps = super(DensityOverflow, self).fixedForward(pos)
        for area_type in range(len(fixed_density_maps)):
            fixed_density_maps[area_type].mul_(self.target_density[area_type])
        return fixed_density_maps

    def forward(self, pos):
        """
        @brief API
        @param pos cell centers. The array consists of (x, y) locations of all cells
        """
        # ignore fillers when computing the density overflow
        filler_range = self.filler_range
        self.filler_range = (0, 0)
        density_maps = super(DensityOverflow, self).forward(pos)
        self.filler_range = filler_range

        # compute overflow
        overflows = pos.new_zeros(len(density_maps))
        for area_type in range(len(density_maps)):
            overflows[area_type] = (density_maps[area_type] -
                                    self.bin_map_areas[area_type] *
                                    self.target_density[area_type]).clamp_(
                                        min=0).sum()
        return overflows
