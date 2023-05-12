# MIT License
#
# Copyright (c) 2020 Jing Mai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import unittest
import torch
import numpy as np
from parameterized import parameterized

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = os.path.abspath(sys.argv[1])
print("use project_dir = %s" % project_dir)

sys.path.append(project_dir)
if True:
    from openparf.ops.rudy import rudy
sys.path.pop()


class RudyUnittest(unittest.TestCase):
    @parameterized.expand([
        [False],
        [True],
    ])
    def test_rudy(self, deterministic_flag):
        dtype = torch.float32
        pin_pos = torch.Tensor([[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1], [0.6, 1.1]]).to(dtype)
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        net_weights = torch.Tensor([1, 2]).to(dtype)

        # construct flat_net2pin_map and flat_net2pin_start_map
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin_pos), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
        flat_net2pin_start_map = np.zeros(len(net2pin_map) + 1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count + len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count
            count += len(net2pin_map[i])
        flat_net2pin_start_map[len(net2pin_map)] = len(pin_pos)
        print("flat_net2pin_map = ", flat_net2pin_map)
        print("flat_net2pin_start_map = ", flat_net2pin_start_map)
        flat_net2pin_map = torch.from_numpy(flat_net2pin_map)
        flat_net2pin_start_map = torch.from_numpy(flat_net2pin_start_map)

        # parameters for this test
        xl, xh = 0.0, 2.0
        yl, yh = 0.0, 4.0
        num_bins_x = 8
        num_bins_y = 8
        bin_size_x = (xh - xl) / num_bins_x
        bin_size_y = (yh - yl) / num_bins_y
        unit_horizontal_capacity = 0.1
        unit_vertical_capacity = 0.2

        # test cpu
        rudy_op = rudy.Rudy(
            netpin_start=flat_net2pin_start_map,
            flat_netpin=flat_net2pin_map,
            net_weights=net_weights,
            xl=xl,
            xh=xh,
            yl=yl,
            yh=yh,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            unit_horizontal_capacity=unit_horizontal_capacity,
            unit_vertical_capacity=unit_vertical_capacity,
            deterministic_flag=deterministic_flag
        )
        result_cpu, _, _ = rudy_op.forward(pin_pos.contiguous().view(-1))
        print("Test on CPU. rudy map = ", result_cpu)

        # test gpu
        if torch.cuda.device_count():
            rudy_op_cuda = rudy.Rudy(
                netpin_start=flat_net2pin_start_map.cuda(),
                flat_netpin=flat_net2pin_map.cuda(),
                net_weights=net_weights.cuda(),
                xl=xl,
                xh=xh,
                yl=yl,
                yh=yh,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                unit_horizontal_capacity=unit_horizontal_capacity,
                unit_vertical_capacity=unit_vertical_capacity,
                deterministic_flag=deterministic_flag
            )
            result_cuda, _, _ = rudy_op_cuda.forward(pin_pos.contiguous().view(-1).cuda())
            print("Test on GPU. rudy map = ", result_cuda)
            np.testing.assert_allclose(result_cpu, result_cuda.cpu())


# ground truth:
#
# flat_net2pin_map =  [0 4 1 2 3]
# flat_net2pin_start_map =  [0 2 5]
# Test on CPU. rudy map =  tensor([[ 9.0909,  9.0909,  1.8182,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 9.0909,  9.0909,  1.8182,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 9.3333, 13.3333, 10.6667, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 6.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 6.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 6.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
# Test on GPU. rudy map =  tensor([[ 9.0909,  9.0909,  1.8182,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 9.0909,  9.0909,  1.8182,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 9.3333, 13.3333, 10.6667, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 6.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 6.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 6.0000, 10.0000, 10.0000, 10.0000, 10.0000, 10.0000,  2.0000,  0.0000],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
#         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]],
#        device='cuda:0')
# .
# ----------------------------------------------------------------------
# Ran 1 test in 3.574s
#
# OK
#

if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
