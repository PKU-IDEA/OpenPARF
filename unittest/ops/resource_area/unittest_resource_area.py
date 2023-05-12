##
# @file   resource_area_unitest.py
# @author Yibai Meng
# @date   Aug 2020
#

import os
import sys
import unittest
import torch
import numpy as np
import math

print(sys.argv)
if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = os.path.abspath(sys.argv[1])
print("use project_dir = %s" % project_dir)

sys.path.append(project_dir)
from openparf.ops.resource_area import resource_area
sys.path.pop()


class ResourceAreaUnittest(unittest.TestCase):
    def test_resource_area_fpga02(self):
        print("Test on ISPD2017 FPGA02")
        dump = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/unittest_resource_area_dump_2.pt")
        num_insts = 166218 + 156
        num_movable_insts = 166218
        stddev_trunc = 2.5
        stddev = math.sqrt(2.5e-4 * num_insts) / (2.0 * stddev_trunc)
        x_len, y_len = 168, 480
        num_bins_x, num_bins_y = math.ceil(x_len / stddev), math.ceil(y_len / stddev)
        slice_capacity = 16
        resource_area_op = resource_area.ResourceArea(
                     dump["is_luts"],
                     dump["is_ffs"],
                     dump["ff_ctrlsets"],
                     11,
                     11,
                     num_bins_x,
                     num_bins_y,
                     stddev,
                     stddev,
                     stddev_trunc,
                     slice_capacity)
        result_cpu = resource_area_op.forward(
            inst_pos=dump["inst_pos"],
        )
        result_cpu = result_cpu[:num_movable_insts]
        print("Test on CPU: ")
        print("    resource_area by instances = ", result_cpu)
        is_close = torch.allclose(result_cpu, dump["inst_area"])
        self.assertTrue(is_close)
        # diff = torch.abs(result_cpu - dump["inst_area"])
        # print(diff, torch.max(diff), torch.nonzero(diff >= 1e-6))
    def test_resource_area_fpga06(self):
        print("Test on ISPD2017 FPGA06")
        dump = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/unittest_resource_area_dump_6.pt")
        num_insts = 703888 + 606
        num_movable_insts = 703888
        stddev_trunc = 2.5
        stddev = math.sqrt(2.5e-4 * num_insts) / (2.0 * stddev_trunc)
        x_len, y_len = 168, 480
        num_bins_x, num_bins_y = math.ceil(x_len / stddev), math.ceil(y_len / stddev)
        slice_capacity = 16
        resource_area_op = resource_area.ResourceArea(
                     dump["is_luts"],
                     dump["is_ffs"],
                     dump["ff_ctrlsets"],
                     21,
                     121,
                     num_bins_x,
                     num_bins_y,
                     stddev,
                     stddev,
                     stddev_trunc,
                     slice_capacity)
        result_cpu = resource_area_op.forward(
            inst_pos=dump["inst_pos"],
        )
        result_cpu = result_cpu[:num_movable_insts]
        print("Test on CPU: ")
        print("    resource_area by instances = ", result_cpu)
        is_close = torch.allclose(result_cpu, dump["inst_area"])
        self.assertTrue(is_close)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()  # Ignore the first one!
    unittest.main()
