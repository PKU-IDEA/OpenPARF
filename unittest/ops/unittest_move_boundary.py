##
# @file   unitest_move_boundary.py
# @author Yibo Lin
# @date   Apr 2020
#

import pdb
import os
import sys
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = sys.argv[1]
print("use project_dir = %s" % (project_dir))

sys.path.append(project_dir)
from openparf.ops.move_boundary import move_boundary
import openparf.configure as configure
sys.path.pop()


def moveBoundary(x, y, w, h, xl, yl, xh, yh):
    x = max(x, xl)
    x = min(x, xh - w)
    y = max(y, yl)
    y = min(y, yh - h)
    return x, y


def nodeMoveBoundary(pos, node_sizes, xl, yl, xh, yh, movable_range,
                     filler_range):
    res = np.zeros_like(pos)
    for i in range(movable_range[0], movable_range[1]):
        xx = pos[i][0]
        yy = pos[i][1]
        w = node_sizes[i][0]
        h = node_sizes[i][1]
        res[i][0], res[i][1] = moveBoundary(xx, yy, w, h, xl, yl, xh, yh)
    for i in range(filler_range[0], filler_range[1]):
        xx = pos[i][0]
        yy = pos[i][1]
        w = node_sizes[i][0]
        h = node_sizes[i][1]
        res[i][0], res[i][1] = moveBoundary(xx, yy, w, h, xl, yl, xh, yh)
    return res


class MoveBoundaryOpTest(unittest.TestCase):
    def test_move_boundaryRandom(self):
        dtype = np.float32
        pos = np.array([[1, 3], [-1, 2], [-1, -2], [5, 6], [7, 8]],
                       dtype=dtype)
        node_sizes = np.array([[1, 1], [0.5, 1], [1, 0.5], [0.5, 0.5], [2, 3]],
                              dtype=dtype)
        xl = 1.0
        yl = 1.0
        xh = 5.0
        yh = 5.0
        movable_range = (0, len(pos) - 1)
        filler_range = (len(pos) - 1, len(pos))

        golden = nodeMoveBoundary(pos, node_sizes, xl, yl, xh, yh,
                                  movable_range, filler_range)
        print("golden")
        print(golden)

        pos_var = Variable(torch.from_numpy(pos))

        # test cpu
        custom = move_boundary.MoveBoundary(
            inst_sizes=torch.from_numpy(node_sizes),
            xl=xl,
            yl=yl,
            xh=xh,
            yh=yh,
            movable_range=movable_range,
            filler_range=filler_range)
        # convert to centers
        result = custom(pos_var + torch.from_numpy(node_sizes) / 2) - torch.from_numpy(node_sizes) / 2
        print("result")
        print(result)
        np.testing.assert_allclose(golden, result.numpy())

        # test cuda
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            custom_cuda = move_boundary.MoveBoundary(
                inst_sizes=torch.from_numpy(node_sizes).cuda(),
                xl=xl,
                yl=yl,
                xh=xh,
                yh=yh,
                movable_range=movable_range,
                filler_range=filler_range)
            result_cuda = custom_cuda(pos_var.cuda() + torch.from_numpy(node_sizes).cuda() / 2) - torch.from_numpy(node_sizes).cuda() / 2
            print("result_cuda")
            print(result_cuda)
            np.testing.assert_allclose(golden, result_cuda.cpu().numpy())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
