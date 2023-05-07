##
# @file   pin_utilization_unitest.py
# @author Zixuan Jiang, Jiaqi Gu, Yibai Meng, Jing Mai
# @date   Dec 2019
#

import os
import sys
import unittest
import torch
import numpy as np
import math
from parameterized import parameterized

print(sys.argv)
if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = os.path.abspath(sys.argv[1])
print("use project_dir = %s" % project_dir)

sys.path.append(project_dir)
if True:
    from openparf.ops.pin_utilization import pin_utilization
sys.path.pop()


class PinUtilizationUnittest(unittest.TestCase):
    @parameterized.expand([
        [False],
        [True],
    ])
    def test_pin_utilization(self, deterministic_flag):
        # the data of insts are from unitest/ops/pin_pos_unitest.py
        dtype = torch.float32

        pos = torch.Tensor([[1, 10], [2, 20], [3, 30]]).to(dtype)
        inst2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        num_movable_insts = len(inst2pin_map)
        num_filler_insts = 1
        num_insts = num_movable_insts + num_filler_insts

        num_pins = 0
        for pins in inst2pin_map:
            num_pins += len(pins)
        pin2inst_map = np.zeros(num_pins, dtype=np.int32)
        for inst_id, pins in enumerate(inst2pin_map):
            for pin in pins:
                pin2inst_map[pin] = inst_id

        # construct flat_inst2pin_map and flat_inst2pin_start_map
        # flat instpin map, length of #pins
        flat_inst2pin_map = np.zeros(num_pins, dtype=np.int32)
        # starting index in instpin map for each inst, length of #insts+1, the last entry is #pins
        flat_inst2pin_start_map = np.zeros(len(inst2pin_map) + 1,
                                           dtype=np.int32)
        count = 0
        for i in range(len(inst2pin_map)):
            flat_inst2pin_map[count:count +
                              len(inst2pin_map[i])] = inst2pin_map[i]
            flat_inst2pin_start_map[i] = count
            count += len(inst2pin_map[i])
        flat_inst2pin_start_map[len(inst2pin_map)] = len(pin2inst_map)
        flat_inst2pin_start_map = torch.from_numpy(flat_inst2pin_start_map)

        inst_sizes = torch.Tensor([[3, 6], [3, 6], [3, 3]]).to(dtype)
        xl, xh = 0, 8
        yl, yh = 0, 64
        num_bins_x, num_bins_y = 2, 16
        bin_size_x = (xh - xl) / num_bins_x
        bin_size_y = (yh - yl) / num_bins_y

        pin_weights = (flat_inst2pin_start_map[1:] - flat_inst2pin_start_map[:-1]).to(dtype)
        unit_pin_capacity = 0.5
        pin_stretch_ratio = math.sqrt(2)

        # test cpu
        pin_utilization_op = pin_utilization.PinUtilization(
            inst_pin_weights=pin_weights,
            xl=xl,
            xh=xh,
            yl=yl,
            yh=yh,
            inst_range=(0, num_insts),
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            unit_pin_capacity=unit_pin_capacity,
            pin_stretch_ratio=pin_stretch_ratio,
            deterministic_flag=deterministic_flag)

        result_cpu = pin_utilization_op.forward(
            inst_sizes=inst_sizes,
            inst_pos=pos.t().contiguous().view(-1),
        )
        print("Test on CPU. pin_utilization map = ", result_cpu)

        if torch.cuda.device_count():
            # test gpu
            pin_utilization_op_cuda = pin_utilization.PinUtilization(
                inst_pin_weights=pin_weights.cuda(),
                xl=xl,
                xh=xh,
                yl=yl,
                yh=yh,
                inst_range=(0, num_insts),
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                unit_pin_capacity=unit_pin_capacity,
                pin_stretch_ratio=pin_stretch_ratio,
                deterministic_flag=deterministic_flag)

            result_cuda = pin_utilization_op_cuda.forward(
                inst_sizes=inst_sizes.cuda(),
                inst_pos=pos.t().contiguous().view(-1).cuda(),
            )
            print("Test on GPU. pin_utilization map = ", result_cuda)
            np.testing.assert_allclose(result_cpu, result_cuda.cpu())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()  # Ignore the first one!
    unittest.main()
