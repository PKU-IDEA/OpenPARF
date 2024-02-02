##
# @file   unitest_sll.py
# @author Runzhe Tao
# @date   Dec 2023
#

import pdb
import os
import sys
import math
import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
else:
    project_dir = sys.argv[1]
print("use project_dir = %s" % (project_dir))

sys.path.append(project_dir)
from openparf.ops.sll import sll
import openparf.configure as configure

sys.path.pop()


def computeSllFor1x4SlrTopology(
    pos, flat_netpin, netpin_start, net_mask, slr_height
):
    num_nets = len(netpin_start) - 1
    partial_slls = torch.zeros(num_nets, dtype=torch.int32)

    for i in range(num_nets):
        if net_mask[i] == 0:
            continue
        pin_start, pin_end = netpin_start[i], netpin_start[i + 1]
        pins_id = flat_netpin[pin_start:pin_end]

        y_positions = pos[pins_id, 1]
        yy = torch.floor(y_positions / slr_height).to(torch.int32)
        sll_counts = yy.max() - yy.min()

        partial_slls[i] = sll_counts

    return partial_slls.sum()


def computeSllFor2x2SlrTopology(
    pos, flat_netpin, netpin_start, net_mask, slr_width, slr_height, num_slrX,
    num_slrY
):
    num_nets = len(netpin_start) - 1
    partial_slls = torch.zeros(num_nets, dtype=torch.int32)

    sll_counts_table = torch.tensor(
        [0, 0, 0, 1, 0, 1, 2, 2, 0, 2, 1, 2, 1, 2, 2, 3], dtype=torch.int32
    )

    for i in range(num_nets):
        if net_mask[i] == 0:
            continue

        pin_start, pin_end = netpin_start[i], netpin_start[i + 1]
        pins_id = flat_netpin[pin_start:pin_end]

        x_positions = torch.floor(pos[pins_id, 0] / slr_width).to(torch.int32)
        y_positions = torch.floor(pos[pins_id, 1] / slr_height).to(torch.int32)
        slr_ids = torch.unique(y_positions * num_slrX +
                               x_positions).numpy().tolist()

        slr_ocpt_idx = sum(
            1 << (num_slrX * num_slrY - id - 1) for id in slr_ids
        )

        partial_slls[i] = sll_counts_table[slr_ocpt_idx]

    return partial_slls.sum()


class SLLOpTest(unittest.TestCase):

    def testSllFor1x4SlrTopology(self):
        num_nets = 10000
        pin_each_net = 4
        num_pins = num_nets * pin_each_net
        flat_netpin = np.arange(num_pins, dtype=np.int32)
        netpin_start = np.arange(num_pins + 1, step=4, dtype=np.int32)
        net_weights = np.ones(num_nets, dtype=np.float64)
        net_mask = np.ones(num_nets, dtype=np.uint8)
        xl, yl, xh, yh = 0., 0., 168.0, 480.0
        num_slrX, num_slrY = 1, 4
        slr_width = (xh - xl) / num_slrX
        slr_height = (yh - yl) / num_slrY

        # generate random pos
        torch.manual_seed(42)
        pos_groups = []
        for i in range(4):
            pos = torch.rand(10000, 2, dtype=torch.float64)
            pos[:, 0] = 168 * pos[:, 0]
            pos[:, 1] = 120 * pos[:, 1] + 120 * i
            pos_groups.append(pos)
        var = torch.cat(pos_groups, dim=0)
        # generate random indexes
        indices = torch.randperm(var.size(0))
        var = var[indices]

        custom = sll.SLL(
            flat_netpin=torch.from_numpy(flat_netpin),
            netpin_start=torch.from_numpy(netpin_start),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            xl=xl,
            yl=yl,
            slr_width=slr_width,
            slr_height=slr_height,
            num_slrX=num_slrX,
            num_slrY=num_slrY
        )

        # test cpu
        golden_sll_counts = computeSllFor1x4SlrTopology(
            var, flat_netpin, netpin_start, net_mask, slr_height
        )
        print(
            f"SLR Topology: {num_slrX}x{num_slrY}, #num_slls: {golden_sll_counts}"
        )
        custom_sll_counts = custom.forward(var)
        np.testing.assert_allclose(
            golden_sll_counts.data.numpy(), custom_sll_counts.data.numpy()
        )

        # test gpu, the project must be compiled with cuda enabled
        # and run with cuda available
        if configure.compile_configurations[
            "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            custom_cuda = sll.SLL(
                flat_netpin=torch.from_numpy(flat_netpin).cuda(),
                netpin_start=torch.from_numpy(netpin_start).cuda(),
                net_weights=torch.from_numpy(net_weights).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                xl=xl,
                yl=yl,
                slr_width=slr_width,
                slr_height=slr_height,
                num_slrX=num_slrX,
                num_slrY=num_slrY
            )
            custom_cuda_sll_counts = custom_cuda.forward(var.cuda())
            np.testing.assert_allclose(
                golden_sll_counts.data.numpy(),
                custom_cuda_sll_counts.data.cpu().numpy()
            )

    def testSllFor2x2SlrTopology(self):
        num_nets = 10000
        pin_each_net = 4
        num_pins = num_nets * pin_each_net
        flat_netpin = np.arange(num_pins, dtype=np.int32)
        netpin_start = np.arange(num_pins + 1, step=4, dtype=np.int32)
        net_weights = np.ones(num_nets, dtype=np.float64)
        net_mask = np.ones(num_nets, dtype=np.uint8)
        xl, yl, xh, yh = 0., 0., 168.0, 480.0
        num_slrX, num_slrY = 2, 2
        slr_width = (xh - xl) / num_slrX
        slr_height = (yh - yl) / num_slrY

        # generate random pos
        torch.manual_seed(42)
        pos_groups = []
        for i in range(4):
            pos = torch.rand(10000, 2, dtype=torch.float64)
            pos[:, 0] = 84 * pos[:, 0] + 84 * int(i % 2)
            pos[:, 1] = 240 * pos[:, 1] + 240 * int(i / 2)
            pos_groups.append(pos)
        var = torch.cat(pos_groups, dim=0)
        # generate random indexes
        indices = torch.randperm(40000)
        var = var[indices]

        # test cpu
        custom = sll.SLL(
            flat_netpin=torch.from_numpy(flat_netpin),
            netpin_start=torch.from_numpy(netpin_start),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            xl=xl,
            yl=yl,
            slr_width=slr_width,
            slr_height=slr_height,
            num_slrX=num_slrX,
            num_slrY=num_slrY
        )

        golden_sll_counts = computeSllFor2x2SlrTopology(
            var, flat_netpin, netpin_start, net_mask, slr_width, slr_height,
            num_slrX, num_slrY
        )
        print(
            f"\nSLR Topology: {num_slrX}x{num_slrY}, #num_slls: {golden_sll_counts}"
        )
        custom_sll_counts = custom.forward(var)

        np.testing.assert_allclose(
            golden_sll_counts.data.numpy(), custom_sll_counts.data.numpy()
        )

        # test gpu, the project must be compiled with cuda enabled
        # and run with cuda available
        if configure.compile_configurations[
            "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            custom_cuda = sll.SLL(
                flat_netpin=torch.from_numpy(flat_netpin).cuda(),
                netpin_start=torch.from_numpy(netpin_start).cuda(),
                net_weights=torch.from_numpy(net_weights).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                xl=xl,
                yl=yl,
                slr_width=slr_width,
                slr_height=slr_height,
                num_slrX=num_slrX,
                num_slrY=num_slrY
            )
            custom_cuda_sll_counts = custom_cuda.forward(var.cuda())
            np.testing.assert_allclose(
                golden_sll_counts.data.numpy(),
                custom_cuda_sll_counts.data.cpu().numpy()
            )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
