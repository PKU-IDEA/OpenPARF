#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : unittest_pin_pos.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 05.01.2020
# Last Modified Date: 05.01.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>

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
from openparf.ops.pin_pos import pin_pos
import openparf.configure as configure
sys.path.pop()


def build_pin_pos(pos, pin_offset_x, pin_offset_y, pin2node_map,
                  num_physical_nodes):
    num_nodes = pos.numel() // 2
    print(
        torch.index_select(pos[0:num_physical_nodes],
                           dim=0,
                           index=pin2node_map.long()))
    pin_x = pin_offset_x.add(
        torch.index_select(pos[0:num_physical_nodes],
                           dim=0,
                           index=pin2node_map.long()))
    print(
        torch.index_select(pos[num_nodes:num_nodes + num_physical_nodes],
                           dim=0,
                           index=pin2node_map.long()))
    pin_y = pin_offset_y.add(
        torch.index_select(pos[num_nodes:num_nodes + num_physical_nodes],
                           dim=0,
                           index=pin2node_map.long()))
    pin_pos = torch.cat([pin_x, pin_y], dim=0)
    #pin_pos = torch.stack([pin_x, pin_y], dim=-1)
    return pin_pos


class PinPosOpTest(unittest.TestCase):
    def test_pin_pos_random(self):
        dtype = torch.float32
        pos = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.float32)
        node2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        num_physical_nodes = len(node2pin_map)
        num_pins = 0
        for pins in node2pin_map:
            num_pins += len(pins)
        pin2node_map = np.zeros(num_pins, dtype=np.int32)
        for node_id, pins in enumerate(node2pin_map):
            for pin in pins:
                pin2node_map[pin] = node_id

        pin_offset_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=pos.dtype)
        pin_offset_y = np.array([0.01, 0.02, 0.03, 0.04, 0.05],
                                dtype=pos.dtype)
        pin_offsets = np.stack([pin_offset_x, pin_offset_y], axis=-1)

        # construct flat_node2pin_map and flat_node2pin_start_map
        # flat nodepin map, length of #pins
        flat_node2pin_map = np.zeros(num_pins, dtype=np.int32)
        # starting index in nodepin map for each node, length of #nodes+1, the last entry is #pins
        flat_node2pin_start_map = np.zeros(len(node2pin_map) + 1,
                                           dtype=np.int32)
        count = 0
        for i in range(len(node2pin_map)):
            flat_node2pin_map[count:count +
                              len(node2pin_map[i])] = node2pin_map[i]
            flat_node2pin_start_map[i] = count
            count += len(node2pin_map[i])
        flat_node2pin_start_map[len(node2pin_map)] = len(pin2node_map)

        print("flat_node2pin_map = ", flat_node2pin_map)
        print("flat_node2pin_start_map = ", flat_node2pin_start_map)
        print("pin2node_map = ", pin2node_map)
        print("num_physical_nodes = ", num_physical_nodes)

        pos_var = Variable(torch.from_numpy(pos).reshape([-1]),
                           requires_grad=True)

        # a bit tricky
        # pos_var is xxxyyy
        # golden_value is xyxyxy
        golden_value = build_pin_pos(pos_var, torch.from_numpy(pin_offset_x),
                                     torch.from_numpy(pin_offset_y),
                                     torch.from_numpy(pin2node_map),
                                     num_physical_nodes)
        golden_loss = golden_value.sum()
        golden_loss.backward()
        golden_grad = pos_var.grad.clone()
        golden_value = golden_value.detach().numpy().reshape([2, -1])
        golden_grad = golden_grad.detach().numpy().reshape([2, -1])
        # convert from xxxyyy to xyxyxy
        golden_value = np.stack([golden_value[0], golden_value[1]], axis=-1)
        golden_grad = np.stack([golden_grad[0], golden_grad[1]], axis=-1)
        print("golden_value = ", golden_value)
        print("golden_loss = ", golden_loss)
        print("golden grad = ", golden_grad)

        pos_var = Variable(torch.from_numpy(
            np.stack([pos[0, :], pos[1, :]], axis=-1)),
                           requires_grad=True)

        # test cpu
        print("pos_var")
        print(pos_var)
        # clone is very important, because the custom op cannot deep copy the data
        custom = pin_pos.PinPos(
            pin_offsets=torch.from_numpy(pin_offsets),
            inst_pins=torch.from_numpy(flat_node2pin_map),
            inst_pins_start=torch.from_numpy(flat_node2pin_start_map),
            pin2inst_map=torch.from_numpy(pin2node_map))
        result = custom.forward(pos_var)
        custom_loss = result.sum()
        print("custom = ", result)
        if pos_var.grad:
            pos_var.grad.zero_()
        custom_loss.backward()
        grad = pos_var.grad.clone()
        print("custom_grad = ", grad)

        np.testing.assert_allclose(result.data.detach().numpy(),
                                   golden_value.reshape([-1, 2]),
                                   atol=1e-6)
        np.testing.assert_allclose(grad.data.detach().numpy(),
                                   golden_grad.reshape([-1, 2]),
                                   atol=1e-6)

        # test gpu
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            pos_var.grad.zero_()
            custom_cuda = pin_pos.PinPos(
                pin_offsets=torch.from_numpy(pin_offsets).cuda(),
                inst_pins=torch.from_numpy(flat_node2pin_map).cuda(),
                inst_pins_start=torch.from_numpy(
                    flat_node2pin_start_map).cuda(),
                pin2inst_map=torch.from_numpy(pin2node_map).cuda())
            result_cuda = custom_cuda.forward(pos_var.cuda())
            custom_cuda_loss = result_cuda.sum()
            print("custom_cuda_result = ", result_cuda.data.cpu())
            custom_cuda_loss.backward()
            grad_cuda = pos_var.grad.clone()
            print("custom_grad_cuda = ", grad_cuda.data.cpu())

            np.testing.assert_allclose(result_cuda.data.cpu().numpy(),
                                       golden_value.reshape([-1, 2]),
                                       atol=1e-6)
            np.testing.assert_allclose(grad_cuda.data.cpu().numpy(),
                                       grad.data.numpy().reshape([-1, 2]),
                                       rtol=1e-6,
                                       atol=1e-6)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
