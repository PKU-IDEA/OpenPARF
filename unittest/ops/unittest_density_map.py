##
# @file   density_map_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import os
import sys
import numpy as np
import unittest
import math

import torch
from torch.autograd import Variable
from parameterized import parameterized

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = sys.argv[1]
print("use project_dir = %s" % (project_dir))
sys.path.append(project_dir)
if True:
    import openparf.configure as configure
    from openparf.ops.density_map import density_map
sys.path.pop()


def binXL(id_x, xl, bin_size_x):
    """
    return bin xl
    """
    return xl + id_x * bin_size_x


def binXH(id_x, xl, xh, bin_size_x):
    """
    return bin xh
    """
    return min(binXL(id_x, xl, bin_size_x) + bin_size_x, xh)


def binYL(id_y, yl, bin_size_y):
    """
    return bin yl
    """
    return yl + id_y * bin_size_y


def binYH(id_y, yl, yh, bin_size_y):
    """
    return bin yh
    """
    return min(binYL(id_y, yl, bin_size_y) + bin_size_y, yh)


def computeDensityMap(pos, node_sizes, node_area_types, dims, xl, yl, xh, yh,
                      node_range, area_type, stretch_flag):
    """Compute triangle and exact density maps for 1 area type
    """
    triangle_map = np.zeros(dims, dtype=pos.dtype)
    exact_map = np.zeros_like(triangle_map)
    bin_sizes = ((xh - xl) / dims[0], (yh - yl) / dims[1])
    for i in range(node_range[0], node_range[1]):
        if node_area_types[i] == area_type:
            node_xl = pos[i][0]
            node_yl = pos[i][1]
            node_xh = node_xl + node_sizes[i][0]
            node_yh = node_yl + node_sizes[i][1]
            if stretch_flag:
                sqrt2 = math.sqrt(2)
                node_w = max(node_sizes[i][0], bin_sizes[0] * sqrt2)
                node_h = max(node_sizes[i][1], bin_sizes[1] * sqrt2)
                # there is a gap of 2 in total area
                node_weight = node_sizes[i][0] * node_sizes[i][1] / (node_w *
                                                                     node_h)
                node_cx = node_xl + node_sizes[i][0] / 2
                node_cy = node_yl + node_sizes[i][1] / 2
                node_xl = node_cx - node_w / 2
                node_xh = node_cx + node_w / 2
                node_yl = node_cy - node_h / 2
                node_yh = node_cy + node_h / 2
            bidxl = max(int((node_xl - xl) / bin_sizes[0]), 0)
            bidxh = min(int((node_xh - xl) / bin_sizes[0]) + 1, dims[0])
            bidyl = max(int((node_yl - xl) / bin_sizes[1]), 0)
            bidyh = min(int((node_yh - xl) / bin_sizes[1]) + 1, dims[1])
            for ix in range(bidxl, bidxh):
                for iy in range(bidyl, bidyh):
                    bxl = binXL(ix, xl, bin_sizes[0])
                    bxh = binXH(ix, xl, xh, bin_sizes[0])
                    byl = binXL(iy, yl, bin_sizes[1])
                    byh = binXH(iy, yl, yh, bin_sizes[1])

                    triangle_overlap_x = min(node_xh, bxh) - max(node_xl, bxl)
                    triangle_overlap_y = min(node_yh, byh) - max(node_yl, byl)
                    triangle_overlap = triangle_overlap_x * triangle_overlap_y
                    exact_overlap = max(triangle_overlap_x, 0) * max(
                        triangle_overlap_y, 0)

                    if stretch_flag:
                        triangle_overlap *= node_weight
                        exact_overlap *= node_weight

                    triangle_map[ix, iy] += triangle_overlap
                    exact_map[ix, iy] += exact_overlap

    return triangle_map, exact_map


def computeDensityOverflow(density_map, dims, xl, yl, xh, yh, target_density):
    """Compute density overflow from density map
    """
    bin_area = (xh - xl) * (yh - yl) / (dims[0] * dims[1])
    overflow = density_map - target_density * bin_area
    overflow = overflow.clip(min=0.0).sum()

    return overflow


def printMap(density_map):
    """Print map with consistent visual layout
    """
    content = "["
    delimeter2 = ""
    for iy in range(density_map.shape[1] - 1, -1, -1):
        content += delimeter2 + "["
        delimeter2 = ",\n"
        delimeter = ""
        for ix in range(density_map.shape[0]):
            content += delimeter + "%g" % density_map[ix, iy]
            delimeter = ", "
        content += "]"
    content += "]"
    print(content)


class DensityMapOpTest(unittest.TestCase):
    @parameterized.expand([
        [True],
        [False],
    ])
    def testDensity1Map(self, deterministic_flag):
        """
        @brief test one map
        ============================
        |        |        |        |
        |        |        |        |
        |        |        |        |
        |===========================
        |        ||------||        |
        ||------||| MOV  |||------------|
        ||  M   |||------|||----FIX-----|
        |===O=======================
        ||  V   ||        |        |
        ||      ||        |        |
        ||------||        |        |
        ============================
        ============================
        |        |        |        |
        |        |        |        |
        |        |        |        |
        |===========================
        |        ||-----------|    |
        |        ||---FILLER--|    |
        |        |        |        |
        |===========================
        |        |        |        |
        |        |        |        |
        |        |        |        |
        ============================
        """
        dtype = np.float32
        pos = np.array([[1, 1], [2, 2], [3, 2], [2, 2.5]], dtype=dtype)
        node_sizes = np.array([[1, 1.5], [1, 1], [1.5, 0.5], [1.5, 0.5]],
                              dtype=dtype)
        node_area_types = np.array([0, 0, 0, 0], dtype=np.uint8)
        node_sizes_2d = np.zeros([node_sizes.shape[0], np.amax(node_area_types) + 1, 2], dtype=dtype)
        for i, area_type in enumerate(node_area_types):
            node_sizes_2d[i, area_type] = node_sizes[i]

        xl = 1.0
        yl = 1.0
        xh = 4.0
        yh = 4.0
        bin_map_dims = np.array([[3, 3]], dtype=np.int32)
        movable_range = (0, 2)
        fixed_range = (2, 3)
        filler_range = (3, 4)
        stretch_flag = 0  # no stretching
        initial_density_maps = []
        for area_type in range(len(bin_map_dims)):
            initial_density_maps.append(np.zeros(bin_map_dims[area_type], dtype=dtype))

        fixed_triangle_map_0, fixed_exact_map_0 = computeDensityMap(
            pos,
            node_sizes,
            node_area_types,
            bin_map_dims[0],
            xl,
            yl,
            xh,
            yh,
            fixed_range,
            area_type=0,
            stretch_flag=False)
        print("fixed_exact_map")
        printMap(fixed_exact_map_0)
        movable_triangle_map_0, movable_exact_map_0 = computeDensityMap(
            pos,
            node_sizes,
            node_area_types,
            bin_map_dims[0],
            xl,
            yl,
            xh,
            yh,
            movable_range,
            area_type=0,
            stretch_flag=stretch_flag)
        print("movable_triangle_map")
        printMap(movable_triangle_map_0)
        filler_triangle_map_0, filler_exact_map_0 = computeDensityMap(
            pos,
            node_sizes,
            node_area_types,
            bin_map_dims[0],
            xl,
            yl,
            xh,
            yh,
            filler_range,
            area_type=0,
            stretch_flag=stretch_flag)
        print("filler_triangle_map")
        printMap(filler_triangle_map_0)

        pos_var = Variable(torch.from_numpy(pos))
        # test cpu
        custom = density_map.DensityMap(torch.from_numpy(node_sizes_2d),
                                        [torch.from_numpy(x) for x in initial_density_maps],
                                        torch.from_numpy(bin_map_dims),
                                        torch.ones((len(bin_map_dims),), dtype=torch.int32),
                                        xl=xl,
                                        yl=yl,
                                        xh=xh,
                                        yh=yh,
                                        movable_range=movable_range,
                                        filler_range=filler_range,
                                        fixed_range=fixed_range,
                                        stretch_flag=stretch_flag,
                                        smooth_flag=False,
                                        deterministic_flag=deterministic_flag)

        # convert to centers
        result = custom.forward(pos_var + torch.from_numpy(node_sizes) / 2)
        for area_type in range(len(result)):
            print("custom[%d]" % (int(area_type)))
            printMap(result[area_type].numpy())
        np.testing.assert_allclose(fixed_exact_map_0,
                                   custom.fixed_density_maps[0],
                                   rtol=1e-6,
                                   atol=1e-6)
        np.testing.assert_allclose(fixed_exact_map_0 + movable_triangle_map_0 +
                                   filler_triangle_map_0,
                                   result[0],
                                   rtol=1e-6,
                                   atol=1e-6)
        # test cuda
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            custom_cuda = density_map.DensityMap(
                torch.from_numpy(node_sizes_2d).cuda(),
                [torch.from_numpy(x).cuda() for x in initial_density_maps],
                torch.from_numpy(bin_map_dims).cuda(),
                torch.ones((len(bin_map_dims),), dtype=torch.int32).cuda(),
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                movable_range=movable_range,
                filler_range=filler_range,
                fixed_range=fixed_range,
                stretch_flag=stretch_flag,
                smooth_flag=False,
                deterministic_flag=deterministic_flag)

            pos_var = Variable(torch.from_numpy(pos)).cuda()
            # convert to centers
            result_cuda = custom_cuda.forward(pos_var + torch.from_numpy(node_sizes).cuda() / 2)
            for area_type in range(len(result_cuda)):
                print("custom cuda[%d]" % (int(area_type)))
                printMap(result_cuda[area_type].cpu().numpy())

            np.testing.assert_allclose(
                fixed_exact_map_0,
                custom_cuda.fixed_density_maps[0].cpu().numpy(),
                rtol=1e-6,
                atol=1e-6)
            np.testing.assert_allclose(fixed_exact_map_0 +
                                       movable_triangle_map_0 +
                                       filler_triangle_map_0,
                                       result_cuda[0].cpu().numpy(),
                                       rtol=1e-6,
                                       atol=1e-6)

    @parameterized.expand([
        [True],
        [False],
    ])
    def testDensity2Maps(self, deterministic_flag):
        """
        @brief test one map
        """
        dtype = np.float32
        pos = np.array([[1, 1], [2, 2], [3, 2], [2, 2.5]], dtype=dtype)
        node_sizes = np.array([[1, 1.5], [1, 1], [1.5, 0.5], [1.5, 0.5]],
                              dtype=dtype)
        node_area_types = np.array([0, 1, 0, 1], dtype=np.uint8)
        node_sizes_2d = np.zeros([node_sizes.shape[0], np.amax(node_area_types) + 1, 2], dtype=dtype)
        for i, area_type in enumerate(node_area_types):
            node_sizes_2d[i, area_type] = node_sizes[i]
        print(node_sizes_2d)

        xl = 1.0
        yl = 1.0
        xh = 4.0
        yh = 4.0
        bin_map_dims = np.array([[3, 3], [1, 1]], dtype=np.int32)
        movable_range = (0, 2)
        fixed_range = (2, 3)
        filler_range = (3, 4)
        stretch_flag = 1  # control stretching
        target_density = [0.5, 0.5]
        initial_density_maps = []
        for area_type in range(len(bin_map_dims)):
            initial_density_maps.append(np.zeros(bin_map_dims[area_type], dtype=dtype))

        fixed_exact_maps = [None] * len(bin_map_dims)
        for i in range(len(bin_map_dims)):
            tmap, emap = computeDensityMap(pos,
                                           node_sizes,
                                           node_area_types,
                                           bin_map_dims[i],
                                           xl,
                                           yl,
                                           xh,
                                           yh,
                                           fixed_range,
                                           area_type=i,
                                           stretch_flag=False)
            fixed_exact_maps[i] = emap
            print("fixed_exact_maps[%d]" % (i))
            printMap(fixed_exact_maps[i])
        movable_triangle_maps = [None] * len(bin_map_dims)
        for i in range(len(bin_map_dims)):
            tmap, emap = computeDensityMap(pos,
                                           node_sizes,
                                           node_area_types,
                                           bin_map_dims[i],
                                           xl,
                                           yl,
                                           xh,
                                           yh,
                                           movable_range,
                                           area_type=i,
                                           stretch_flag=stretch_flag)
            movable_triangle_maps[i] = tmap
            print("movable_triangle_maps[%d]" % (i))
            printMap(movable_triangle_maps[i])
        filler_triangle_maps = [None] * len(bin_map_dims)
        for i in range(len(bin_map_dims)):
            tmap, emap = computeDensityMap(pos,
                                           node_sizes,
                                           node_area_types,
                                           bin_map_dims[i],
                                           xl,
                                           yl,
                                           xh,
                                           yh,
                                           filler_range,
                                           area_type=i,
                                           stretch_flag=stretch_flag)
            filler_triangle_maps[i] = tmap
            print("filler_triangle_maps[%d]" % (i))
            printMap(filler_triangle_maps[i])
        overflows = [None] * len(bin_map_dims)
        for i in range(len(bin_map_dims)):
            golden_map = fixed_exact_maps[i] * target_density[
                i] + movable_triangle_maps[i] + filler_triangle_maps[i]
            overflows[i] = computeDensityOverflow(golden_map, bin_map_dims[i],
                                                  xl, yl, xh, yh,
                                                  target_density[i])

        pos_var = Variable(torch.from_numpy(pos))
        # test cpu
        custom = density_map.DensityMap(torch.from_numpy(node_sizes_2d),
                                        [torch.from_numpy(x) for x in initial_density_maps],
                                        torch.from_numpy(bin_map_dims),
                                        torch.ones((len(bin_map_dims),), dtype=torch.int32),
                                        xl=xl,
                                        yl=yl,
                                        xh=xh,
                                        yh=yh,
                                        movable_range=movable_range,
                                        filler_range=filler_range,
                                        fixed_range=fixed_range,
                                        stretch_flag=stretch_flag,
                                        smooth_flag=False,
                                        deterministic_flag=deterministic_flag)

        # convert to centers
        result = custom.forward(pos_var + torch.from_numpy(node_sizes) / 2)
        for area_type in range(len(result)):
            print("custom[%d]" % (int(area_type)))
            printMap(result[area_type].numpy())
        # compare map for each area type
        for i in range(len(bin_map_dims)):
            np.testing.assert_allclose(fixed_exact_maps[i],
                                       custom.fixed_density_maps[i],
                                       rtol=1e-6,
                                       atol=1e-6)
            np.testing.assert_allclose(fixed_exact_maps[i] +
                                       movable_triangle_maps[i] +
                                       filler_triangle_maps[i],
                                       result[i],
                                       rtol=1e-6,
                                       atol=1e-6)

        # test density overflow
        custom_overflow = density_map.DensityOverflow(
            torch.from_numpy(node_sizes_2d),
            # torch.from_numpy(node_area_types),
            [torch.from_numpy(x) for x in initial_density_maps],
            torch.from_numpy(bin_map_dims),
            torch.ones((len(bin_map_dims),), dtype=torch.int32),
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            movable_range=movable_range,
            filler_range=filler_range,
            fixed_range=fixed_range,
            stretch_flag=stretch_flag,
            smooth_flag=False,
            target_density=target_density,
            deterministic_flag=deterministic_flag)
        # convert to centers
        result_overflows = custom_overflow.forward(pos_var + torch.from_numpy(node_sizes) / 2)
        print("result_overflows")
        print(result_overflows)
        np.testing.assert_allclose(overflows,
                                   result_overflows,
                                   rtol=1e-6,
                                   atol=1e-6)

        # test cuda
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            custom_cuda = density_map.DensityMap(
                torch.from_numpy(node_sizes_2d).cuda(),
                # torch.from_numpy(node_area_types).cuda(),
                [torch.from_numpy(x).cuda() for x in initial_density_maps],
                torch.from_numpy(bin_map_dims).cuda(),
                torch.ones((len(bin_map_dims),), dtype=torch.int32).cuda(),
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                movable_range=movable_range,
                filler_range=filler_range,
                fixed_range=fixed_range,
                stretch_flag=stretch_flag,
                smooth_flag=False,
                deterministic_flag=deterministic_flag)

            pos_var = Variable(torch.from_numpy(pos)).cuda()
            result_cuda = custom_cuda.forward(pos_var + torch.from_numpy(node_sizes).cuda() / 2)
            for area_type in range(len(result_cuda)):
                print("custom cuda[%d]" % (int(area_type)))
                printMap(result_cuda[area_type].cpu().numpy())
            # compare map for each area type
            for i in range(len(bin_map_dims)):
                np.testing.assert_allclose(
                    fixed_exact_maps[i],
                    custom.fixed_density_maps[i].cpu().numpy(),
                    rtol=1e-6,
                    atol=1e-6)
                np.testing.assert_allclose(fixed_exact_maps[i] +
                                           movable_triangle_maps[i] +
                                           filler_triangle_maps[i],
                                           result[i].cpu().numpy(),
                                           rtol=1e-6,
                                           atol=1e-6)
            # test density overflow
            custom_overflow_cuda = density_map.DensityOverflow(
                torch.from_numpy(node_sizes_2d).cuda(),
                # torch.from_numpy(node_area_types).cuda(),
                [torch.from_numpy(x).cuda() for x in initial_density_maps],
                torch.from_numpy(bin_map_dims).cuda(),
                torch.ones((len(bin_map_dims),), dtype=torch.int32).cuda(),
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                movable_range=movable_range,
                filler_range=filler_range,
                fixed_range=fixed_range,
                stretch_flag=stretch_flag,
                smooth_flag=False,
                deterministic_flag=deterministic_flag,
                target_density=target_density)
            result_overflows_cuda = custom_overflow_cuda.forward(pos_var + torch.from_numpy(node_sizes).cuda() / 2)
            result_overflows_cuda = [
                x.cpu().item() for x in result_overflows_cuda
            ]
            print("result_overflows_cuda")
            print(result_overflows_cuda)
            np.testing.assert_allclose(overflows,
                                       result_overflows_cuda,
                                       rtol=1e-6,
                                       atol=1e-6)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
