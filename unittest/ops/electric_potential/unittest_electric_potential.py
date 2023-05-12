##
# @file   electric_potential_unitest.py
# @author Yibo Lin
# @date   Mar 2019
#

import time
import numpy as np
import unittest
import logging

import torch
from torch.autograd import Function, Variable
import os
import sys
import gzip

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
else:
    project_dir = sys.argv[1]
print("use project_dir = %s" % (project_dir))

sys.path.append(project_dir)
from openparf.ops.electric_potential import electric_potential
import openparf.configure as configure
sys.path.pop()

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import pdb
from scipy import fftpack


def torchDType(numpy_dtype):
    if numpy_dtype == np.float64:
        dtype = torch.float64
    elif numpy_dtype == np.float32:
        dtype = torch.float32
    return dtype


def parseDensityPotentialFieldFile(filename, dtype):
    with open(filename, "r") as f:
        line_number = 0
        header_touched = False
        for line in f:
            ++line_number
            if line[0] == '#':
                continue
            if not header_touched:
                tokens = line.split()
                M = int(tokens[0])
                N = int(tokens[1])
                bin_w = float(tokens[2])
                bin_h = float(tokens[3])

                density_map = np.zeros([M, N], dtype=dtype)
                potential_map = np.zeros([M, N], dtype=dtype)
                field_map_x = np.zeros([M, N], dtype=dtype)
                field_map_y = np.zeros([M, N], dtype=dtype)

                header_touched = True
            else:
                tokens = line.split()
                i = int(tokens[0])
                j = int(tokens[1])
                density_map[i, j] = float(tokens[2])
                potential_map[i, j] = float(tokens[3])
                field_map_x[i, j] = float(tokens[4])
                field_map_y[i, j] = float(tokens[5])
    return M, N, bin_w, bin_h, density_map, potential_map, field_map_x, field_map_y


class ElectricPotentialOpTest(unittest.TestCase):
    def testRandom(self):
        """Test random locations
        """
        dtype = np.float64
        xx = np.array([
            1000, 11148, 11148, 11148, 11148, 11148, 11124, 11148, 11148,
            11137, 11126, 11148, 11130, 11148, 11148, 11148, 11148, 11148,
            11148, 0, 11148, 11148, 11150, 11134, 11148, 11148, 11148, 10550,
            11148, 11148, 11144, 11148, 11148, 11148, 11148, 11140, 11120,
            11154, 11148, 11133, 11148, 11148, 11134, 11125, 11148, 11148,
            11148, 11155, 11127, 11148, 11148, 11148, 11148, 11131, 11148,
            11148, 11148, 11148, 11136, 11148, 11146, 11148, 11135, 11148,
            11125, 11150, 11148, 11139, 11148, 11148, 11130, 11148, 11128,
            11148, 11138, 11148, 11148, 11148, 11130, 11148, 11132, 11148,
            11148, 11090
        ]).astype(dtype)
        yy = np.array([
            1000, 11178, 11178, 11190, 11400, 11178, 11172, 11178, 11178,
            11418, 11418, 11178, 11418, 11178, 11178, 11178, 11178, 11178,
            11178, 11414, 11178, 11178, 11172, 11418, 11406, 11184, 11178,
            10398, 11178, 11178, 11172, 11178, 11178, 11178, 11178, 11418,
            11418, 11172, 11178, 11418, 11178, 11178, 11172, 11418, 11178,
            11178, 11178, 11418, 11418, 11178, 11178, 11178, 11178, 11418,
            11178, 11178, 11394, 11178, 11418, 11178, 11418, 11178, 11418,
            11178, 11418, 11418, 11178, 11172, 11178, 11178, 11418, 11178,
            11418, 11178, 11418, 11412, 11178, 11178, 11172, 11178, 11418,
            11178, 11178, 11414
        ]).astype(dtype)
        inst_size_x = np.array([
            6, 3, 3, 3, 3, 3, 5, 3, 3, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3, 16728, 3,
            3, 5, 1, 3, 3, 3, 740, 3, 3, 5, 3, 3, 3, 3, 5, 5, 5, 3, 1, 3, 3, 5,
            1, 3, 3, 3, 5, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 5, 3, 5, 3, 1, 3, 5,
            5, 3, 5, 3, 3, 5, 3, 1, 3, 1, 3, 3, 3, 5, 3, 1, 3, 3, 67
        ]).astype(dtype)
        inst_size_y = np.array([
            6, 240, 240, 6, 6, 240, 6, 240, 240, 6, 6, 240, 6, 240, 240, 240,
            240, 240, 240, 10, 240, 6, 6, 6, 6, 6, 240, 780, 240, 240, 6, 240,
            240, 240, 240, 6, 6, 6, 240, 6, 240, 240, 6, 6, 240, 240, 240, 6,
            6, 240, 240, 240, 240, 6, 240, 240, 6, 240, 6, 240, 6, 240, 6, 240,
            6, 6, 240, 6, 240, 240, 6, 240, 6, 240, 6, 6, 240, 240, 6, 240, 6,
            240, 240, 10
        ]).astype(dtype)

        pos = np.stack([xx, yy], axis=-1)
        inst_sizes = np.stack([inst_size_x, inst_size_y], axis=-1)
        inst_area_types = np.zeros(len(xx), dtype=np.uint8)
        inst_sizes_2d = np.zeros([inst_sizes.shape[0], np.amax(inst_area_types) + 1, 2], dtype=dtype)
        for i, area_type in enumerate(inst_area_types):
            inst_sizes_2d[i, area_type] = inst_sizes[i]

        movable_range = (0, 1)
        fixed_range = (1, len(xx))
        filler_range = (len(xx), len(xx))

        scale_factor = 1.0

        xl = 0.0
        yl = 6.0
        xh = 16728.0
        yh = 11430.0
        target_density = np.array([0.7], dtype=dtype)
        bin_map_dims = np.array([[1024, 1024]], dtype=np.int32)
        bin_size_x = (xh - xl) / bin_map_dims[0][0]
        bin_size_y = (yh - yl) / bin_map_dims[0][1]
        initial_density_maps = []
        for area_type in range(len(bin_map_dims)):
            initial_density_maps.append(np.zeros(bin_map_dims[area_type], dtype=dtype))

        print("target_area = ", target_density[0] * bin_size_x * bin_size_y)

        # hard-coded golden values
        #golden_result = -1.117587e-08 # sum of potential
        golden_result = 2873978.75842769 # real energy
        golden_grad = np.zeros_like(pos)
        golden_grad[0] = [-2.7204, -3.3773]

        # test cpu
        custom = electric_potential.ElectricPotential(
            inst_sizes=torch.from_numpy(inst_sizes_2d),
            #inst_area_types=torch.from_numpy(inst_area_types),
            initial_density_maps=[torch.from_numpy(x) for x in initial_density_maps],
            bin_map_dims=torch.from_numpy(bin_map_dims),
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            movable_range=movable_range,
            filler_range=filler_range,
            fixed_range=fixed_range,
            target_density=target_density,
            smooth_flag=False,
            fast_mode=False)

        pos_var = Variable(torch.from_numpy(pos), requires_grad=True)
        result = custom.forward(pos_var + torch.from_numpy(inst_sizes) / 2)
        print("custom_result = ", result)
        print(result.type())
        result.backward()
        grad = pos_var.grad.clone()
        print("custom_grad[movable %d:%d] = %s" %
              (movable_range[0], movable_range[1],
               grad[movable_range[0]:movable_range[1]]))

        np.testing.assert_allclose(result.detach().numpy(),
                                   golden_result,
                                   rtol=1e-6,
                                   atol=1e-6)
        np.testing.assert_allclose(-grad.detach().numpy(),
                                   golden_grad,
                                   rtol=1e-6,
                                   atol=1e-6)

        # test cuda
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            custom_cuda = electric_potential.ElectricPotential(
                inst_sizes=torch.from_numpy(inst_sizes_2d).cuda(),
                #inst_area_types=torch.from_numpy(inst_area_types).cuda(),
                initial_density_maps=[torch.from_numpy(x).cuda() for x in initial_density_maps],
                bin_map_dims=torch.from_numpy(bin_map_dims).cuda(),
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                movable_range=movable_range,
                filler_range=filler_range,
                fixed_range=fixed_range,
                target_density=target_density,
                smooth_flag=False,
                fast_mode=False)

            pos_var = Variable(torch.from_numpy(pos).cuda(),
                               requires_grad=True)
            result_cuda = custom_cuda.forward(pos_var + torch.from_numpy(inst_sizes).cuda() / 2)
            print("custom_result_cuda = ", result_cuda.data.cpu())
            print(result_cuda.type())
            result_cuda.backward()
            grad_cuda = pos_var.grad.clone()
            print("custom_grad_cuda[movable %d:%d] = %s" %
                  (movable_range[0], movable_range[1],
                   grad_cuda[movable_range[0]:movable_range[1]].data.cpu()))

            np.testing.assert_allclose(result.detach().numpy(),
                                       result_cuda.data.cpu().detach().numpy(),
                                       rtol=1e-6,
                                       atol=1e-6)
            np.testing.assert_allclose(grad.detach().numpy(),
                                       grad_cuda.data.cpu().detach().numpy(),
                                       rtol=1e-6,
                                       atol=1e-6)

    def testElectrostaticSystem(self):
        """Test electrostatic system
        """
        dtype = np.float64

        for i in range(4):
            filename = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/density_potential_field_%d.dat" % (i + 1))
            print("read %s" % (filename))
            M, N, bin_w, bin_h, density_map, potential_map, \
                    field_map_x, field_map_y = parseDensityPotentialFieldFile(filename, dtype)

            # test cpu
            es = electric_potential.ElectrostaticSystem(
                bin_map_dims=torch.tensor([[M, N]], dtype=torch.int32),
                bin_sizes=[[bin_w, bin_h]],
                dtype=torchDType(dtype),
                device=torch.device('cpu'),
                fast_mode=False,
                xy_ratio=1
                )

            custom_potential_maps, custom_field_map_xs, custom_field_map_ys, energy = es.forward(
                [torch.from_numpy(density_map)])

            np.testing.assert_allclose(potential_map,
                                       custom_potential_maps[0],
                                       rtol=1e-3,
                                       atol=1e-5)
            np.testing.assert_allclose(field_map_x,
                                       custom_field_map_xs[0],
                                       rtol=1e-3,
                                       atol=1e-5)
            np.testing.assert_allclose(field_map_y,
                                       custom_field_map_ys[0],
                                       rtol=1e-3,
                                       atol=1e-5)

            # test cuda
            if configure.compile_configurations[
                    "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
                # test cpu
                es = electric_potential.ElectrostaticSystem(
                    bin_map_dims=torch.tensor([[M, N]], dtype=torch.int32),
                    bin_sizes=[[bin_w, bin_h]],
                    dtype=torchDType(dtype),
                    device=torch.device('cuda'),
                    fast_mode=False,
                    xy_ratio=1
                    )

                custom_potential_maps, custom_field_map_xs, custom_field_map_ys, energy = es.forward(
                    [torch.from_numpy(density_map).cuda()])

                np.testing.assert_allclose(
                    potential_map,
                    custom_potential_maps[0].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-5)
                np.testing.assert_allclose(
                    field_map_x,
                    custom_field_map_xs[0].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-5)
                np.testing.assert_allclose(
                    field_map_y,
                    custom_field_map_ys[0].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-5)

    def testMultiElectrostaticSystems(self):
        """Test multiple electrostatic systems
        """
        dtype = np.float64

        MNs = []
        bin_sizes = []
        density_maps = []
        potential_maps = []
        field_map_xs = []
        field_map_ys = []
        for i in range(4):
            filename = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data/density_potential_field_%d.dat" % (i + 1))
            print("read %s" % (filename))
            M, N, bin_w, bin_h, density_map, potential_map, \
                    field_map_x, field_map_y = parseDensityPotentialFieldFile(filename, dtype)
            MNs.append([M, N])
            bin_sizes.append([bin_w, bin_h])
            density_maps.append(density_map)
            potential_maps.append(potential_map)
            field_map_xs.append(field_map_x)
            field_map_ys.append(field_map_y)

        # test cpu
        es = electric_potential.ElectrostaticSystem(bin_map_dims=torch.tensor(
            MNs, dtype=torch.int32),
                                                    bin_sizes=bin_sizes,
                                                    dtype=torchDType(dtype),
                                                    device=torch.device('cpu'),
                                                    fast_mode=False,
                                                    xy_ratio=1
                                                    )

        custom_potential_maps, custom_field_map_xs, custom_field_map_ys, energy = es.forward(
            [torch.from_numpy(density_map) for density_map in density_maps])

        for i in range(4):
            np.testing.assert_allclose(potential_maps[i],
                                       custom_potential_maps[i],
                                       rtol=1e-3,
                                       atol=1e-5)
            np.testing.assert_allclose(field_map_xs[i],
                                       custom_field_map_xs[i],
                                       rtol=1e-3,
                                       atol=1e-5)
            np.testing.assert_allclose(field_map_ys[i],
                                       custom_field_map_ys[i],
                                       rtol=1e-3,
                                       atol=1e-5)

        # test cuda
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            # test cpu
            es = electric_potential.ElectrostaticSystem(
                bin_map_dims=torch.tensor(MNs, dtype=torch.int32),
                bin_sizes=bin_sizes,
                dtype=torchDType(dtype),
                device=torch.device('cuda'),
                fast_mode=False,
                xy_ratio=1
                )

            custom_potential_maps, custom_field_map_xs, custom_field_map_ys, energy = es.forward(
                [
                    torch.from_numpy(density_map).cuda()
                    for density_map in density_maps
                ])

            for i in range(4):
                np.testing.assert_allclose(
                    potential_maps[i],
                    custom_potential_maps[i].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-5)
                np.testing.assert_allclose(
                    field_map_xs[i],
                    custom_field_map_xs[i].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-5)
                np.testing.assert_allclose(
                    field_map_ys[i],
                    custom_field_map_ys[i].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-5)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
