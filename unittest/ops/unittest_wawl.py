##
# @file   unitest_wawl.py
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
from openparf.ops.wawl import wawl
import openparf.configure as configure
sys.path.pop()


def unsortedSegmentSum(pin_x, pin2net_map, num_nets):
    result = np.zeros(num_nets, dtype=pin_x.dtype)
    for i in range(len(pin2net_map)):
        result[pin2net_map[i]] += pin_x[i]
    return result


def buildWirelength(pin_x, pin_y, pin2net_map, net2pin_map, gamma,
                    ignore_net_degree, net_weights):
    """
    @brief A naive way to implement the wirelength computation
    without considering overflow. We use the automatic gradient
    package to get the gradient.
    """
    # wirelength cost
    # weighted-average

    # temporily store exp(x)
    scaled_pin_x = pin_x / gamma
    scaled_pin_y = pin_y / gamma

    exp_pin_x = np.exp(scaled_pin_x)
    exp_pin_y = np.exp(scaled_pin_y)
    nexp_pin_x = np.exp(-scaled_pin_x)
    nexp_pin_y = np.exp(-scaled_pin_y)

    # sum of exp(x)
    sum_exp_pin_x = unsortedSegmentSum(exp_pin_x, pin2net_map,
                                       len(net2pin_map))
    sum_exp_pin_y = unsortedSegmentSum(exp_pin_y, pin2net_map,
                                       len(net2pin_map))
    sum_nexp_pin_x = unsortedSegmentSum(nexp_pin_x, pin2net_map,
                                        len(net2pin_map))
    sum_nexp_pin_y = unsortedSegmentSum(nexp_pin_y, pin2net_map,
                                        len(net2pin_map))

    # sum of x*exp(x)
    sum_x_exp_pin_x = unsortedSegmentSum(pin_x * exp_pin_x, pin2net_map,
                                         len(net2pin_map))
    sum_y_exp_pin_y = unsortedSegmentSum(pin_y * exp_pin_y, pin2net_map,
                                         len(net2pin_map))
    sum_x_nexp_pin_x = unsortedSegmentSum(pin_x * nexp_pin_x, pin2net_map,
                                          len(net2pin_map))
    sum_y_nexp_pin_y = unsortedSegmentSum(pin_y * nexp_pin_y, pin2net_map,
                                          len(net2pin_map))

    sum_exp_pin_x = sum_exp_pin_x
    sum_x_exp_pin_x = sum_x_exp_pin_x

    wl = sum_x_exp_pin_x / sum_exp_pin_x - sum_x_nexp_pin_x / sum_nexp_pin_x \
        + sum_y_exp_pin_y / sum_exp_pin_y - sum_y_nexp_pin_y / sum_nexp_pin_y

    for i in range(len(net2pin_map)):
        if len(net2pin_map[i]) >= ignore_net_degree:
            wl[i] = 0

    wl *= net_weights

    wirelength = np.sum(wl)

    return wirelength


class WAWLOpTest(unittest.TestCase):
    def testWAWLRandom(self):
        dtype = torch.float32
        pin_pos = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.5, 0.2], [0.5, 3.1], [0.6, 1.1]],
            dtype=np.float32)
        net2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])])
        pin2net_map = np.zeros(len(pin_pos), dtype=np.int32)
        for net_id, pins in enumerate(net2pin_map):
            for pin in pins:
                pin2net_map[pin] = net_id
        net_weights = np.array([1, 2], dtype=np.float32)

        pin_x = pin_pos[:, 0]
        pin_y = pin_pos[:, 1]
        gamma = 0.5
        ignore_net_degree = 4
        pin_mask = np.zeros(len(pin2net_map), dtype=np.uint8)

        # net mask
        net_mask = np.ones(len(net2pin_map), dtype=np.uint8)
        for i in range(len(net2pin_map)):
            if len(net2pin_map[i]) >= ignore_net_degree:
                net_mask[i] = 0

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

        golden_value = np.array([
            buildWirelength(pin_x, pin_y, pin2net_map, net2pin_map, gamma,
                            ignore_net_degree, net_weights)
        ])
        print("golden_value = ", golden_value)

        pin_pos_var = Variable(torch.tensor(pin_pos,
                                            dtype=dtype).reshape([-1]),
                               requires_grad=True)
        print(pin_pos_var)
        # clone is very important, because the custom op cannot deep copy the data

        # test cpu
        if pin_pos_var.grad:
            pin_pos_var.grad.zero_()
        custom = wawl.WAWL(
            flat_netpin=Variable(torch.from_numpy(flat_net2pin_map)),
            netpin_start=Variable(torch.from_numpy(flat_net2pin_start_map)),
            pin2net_map=torch.from_numpy(pin2net_map),
            net_weights=torch.from_numpy(net_weights),
            net_mask=torch.from_numpy(net_mask),
            pin_mask=torch.from_numpy(pin_mask),
            gamma=torch.tensor(gamma, dtype=dtype))
        result = custom.forward(pin_pos_var)
        print("custom_cpu_result = ", result.data)
        result.backward()
        grad = pin_pos_var.grad.clone()
        print("custom_grad_cpu = ", grad.data)

        np.testing.assert_allclose(result.data.numpy(),
                                   golden_value,
                                   atol=1e-6)
        np.testing.assert_allclose(grad.data.numpy(),
                                   grad.data.numpy(),
                                   rtol=1e-6,
                                   atol=1e-6)

        # test gpu
        if configure.compile_configurations[
                "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            pin_pos_var.grad.zero_()
            custom_cuda = wawl.WAWL(
                flat_netpin=Variable(
                    torch.from_numpy(flat_net2pin_map)).cuda(),
                netpin_start=Variable(
                    torch.from_numpy(flat_net2pin_start_map)).cuda(),
                pin2net_map=torch.from_numpy(pin2net_map).cuda(),
                net_weights=torch.from_numpy(net_weights).cuda(),
                net_mask=torch.from_numpy(net_mask).cuda(),
                pin_mask=torch.from_numpy(pin_mask).cuda(),
                gamma=torch.tensor(gamma, dtype=dtype).cuda())
            result_cuda = custom_cuda.forward(pin_pos_var.cuda())
            print("custom_cuda_result = ", result_cuda.data.cpu())
            result_cuda.backward()
            grad_cuda = pin_pos_var.grad.clone()
            print("custom_grad_cuda = ", grad_cuda.data.cpu())

            np.testing.assert_allclose(result_cuda.data.cpu().numpy(),
                                       golden_value,
                                       atol=1e-6)
            np.testing.assert_allclose(grad_cuda.data.cpu().numpy(),
                                       grad.data.numpy(),
                                       rtol=1e-6,
                                       atol=1e-6)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
