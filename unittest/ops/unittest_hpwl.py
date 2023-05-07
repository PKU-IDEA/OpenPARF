##
# @file   unitest_hpwl.py
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
from openparf.ops.hpwl import hpwl
import openparf.configure as configure
sys.path.pop()


def netHPWL(x, y, net2pin_map, net_weights, net_id):
    """
    return hpwl of a net
    """
    pins = net2pin_map[net_id]
    hpwl_x = np.amax(x[pins]) - np.amin(x[pins])
    hpwl_y = np.amax(y[pins]) - np.amin(y[pins])

    return np.array([hpwl_x, hpwl_y]) * net_weights[net_id]


def allHPWL(x, y, net2pin_map, net_weights):
    """
    return hpwl of all nets
    """
    wl = np.zeros(2, dtype=net_weights.dtype)
    for net_id in range(len(net2pin_map)):
        wl += netHPWL(x, y, net2pin_map, net_weights, net_id)
    return wl


class HPWLOpTest(unittest.TestCase):
    def test_hpwlRandom(self):
        pin_pos = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1], [0.6, 1.1]],
            dtype=np.float32)
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        # net weights
        net_weights = np.array([1, 2], dtype=np.float32)
        print("net_weights = ", net_weights)
        wirelength_weights = np.array([0.7, 1.2], dtype=np.float32)

        # construct flat_net2pin_map and flat_net2pin_start_map
        # flat netpin map, length of #pins
        flat_net2pin_map = np.zeros(len(pin_pos), dtype=np.int32)
        # starting index in netpin map for each net, length of #nets+1, the last entry is #pins
        flat_net2pin_start_map = np.zeros(len(net2pin_map) + 1, dtype=np.int32)
        count = 0
        for i in range(len(net2pin_map)):
            flat_net2pin_map[count:count +
                             len(net2pin_map[i])] = net2pin_map[i]
            flat_net2pin_start_map[i] = count
            count += len(net2pin_map[i])
        flat_net2pin_start_map[len(net2pin_map)] = len(pin_pos)

        print("flat_net2pin_map = ", flat_net2pin_map)
        print("flat_net2pin_start_map = ", flat_net2pin_start_map)

        # net degrees
        net_degrees = np.array([len(net2pin) for net2pin in net2pin_map])
        net_mask = (net_degrees <= np.amax(net_degrees)).astype(np.uint8)
        print("net_mask = ", net_mask)

        golden_value = allHPWL(pin_pos[:, 0], pin_pos[:, 1], net2pin_map,
                               net_weights)
        print("golden_value = ", golden_value)

        # test cpu
        pin_pos_var = Variable(torch.from_numpy(pin_pos))
        print(pin_pos_var)
        custom = hpwl.HPWL(
            flat_netpin=torch.from_numpy(flat_net2pin_map),
            netpin_start=torch.from_numpy(flat_net2pin_start_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            #wirelength_weights=torch.from_numpy(wirelength_weights)
            )
        hpwl_value = custom.forward(pin_pos_var)
        print("hpwl_value = ", hpwl_value.data.numpy())
        np.testing.assert_allclose(hpwl_value.data.numpy(), golden_value)

        # test gpu, the project must be compiled with cuda enabled
        # and run with cuda available
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            custom_cuda = hpwl.HPWL(
                flat_netpin=torch.from_numpy(flat_net2pin_map).cuda(),
                netpin_start=torch.from_numpy(flat_net2pin_start_map).cuda(),
                net_weights=torch.from_numpy(net_weights).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                #wirelength_weights=torch.from_numpy(wirelength_weights).cuda()
                )
            hpwl_value = custom_cuda.forward(pin_pos_var.cuda())
            print("hpwl_value cuda = ", hpwl_value.data.cpu().numpy())
            np.testing.assert_allclose(hpwl_value.data.cpu().numpy(),
                                       golden_value)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
