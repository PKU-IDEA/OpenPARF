##
# @file   unitest_wasll.py
# @author Runzhe Tao
# @date   Dec 2023
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
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
else:
    project_dir = sys.argv[1]
print("use project_dir = %s" % (project_dir))

sys.path.append(project_dir)
from openparf.ops.soft_floor import soft_floor
import openparf.configure as configure

sys.path.pop()


def sigmoid(x, sigmoid_gamma):
    return 1.0 / (1.0 + np.exp(-sigmoid_gamma * x))


def sigmoidGrad(x, sigmoid_gamma):
    sig = sigmoid(x, sigmoid_gamma)
    return sigmoid_gamma * sig * (1.0 - sig)


def softFloor(x, soft_floor_gamma, num_floor):
    res = np.zeros_like(x)
    for i in range(1, num_floor):
        res += sigmoid(x - i, soft_floor_gamma)
    return res


def softFloorGrad(x, soft_floor_gamma, num_floor):
    res = np.zeros_like(x)
    for i in range(1, num_floor):
        res += sigmoidGrad(x - i, soft_floor_gamma)
    return res


def computeSoftFloor(
    pos, xl, yl, slr_width, slr_height, num_slrX, num_slrY, soft_floor_gamma
):
    result = np.zeros(pos.size, dtype=np.float32)
    grad_intermediate = np.zeros(pos.size, dtype=np.float32)
    num_pins = result.size // 2

    for i in range(num_pins):
        offset = i * 2

        if num_slrX > 1:
            scaled_pos_x = (pos[i][0] - xl) / slr_width
            result[offset] = softFloor(scaled_pos_x, soft_floor_gamma, num_slrX)
            grad_intermediate[offset] = softFloorGrad(
                scaled_pos_x, soft_floor_gamma, num_slrX
            ) / slr_width

        if num_slrY > 1:
            scaled_pos_y = (pos[i][1] - yl) / slr_height
            result[offset +
                   1] = softFloor(scaled_pos_y, soft_floor_gamma, num_slrY)
            grad_intermediate[offset + 1] = softFloorGrad(
                scaled_pos_y, soft_floor_gamma, num_slrY
            ) / slr_height

    return result, grad_intermediate


class SoftFloorOpTest(unittest.TestCase):

    def testSoftFloorRandom(self):
        xl, yl = 0., 0.
        slr_width, slr_height = 168., 480.
        num_slrX, num_slrY = 1, 4
        pos = np.array([[80., 60.], [80., 360.]], dtype=np.float32)
        soft_floor_gamma = torch.tensor([5.0], dtype=torch.float32)
        pos_var = Variable(
            torch.tensor(pos, dtype=torch.float32).reshape([-1]),
            requires_grad=True
        )
        golden_value, golden_grad = computeSoftFloor(
            pos, xl, yl, slr_width, slr_height, num_slrX, num_slrY,
            soft_floor_gamma.item()
        )

        # test cpu
        if pos_var.grad is not None:
            pos_var.grad.zero_()
        custom = soft_floor.SoftFloor(
            xl=xl,
            yl=yl,
            slr_width=slr_width,
            slr_height=slr_height,
            num_slrX=num_slrX,
            num_slrY=num_slrY,
            soft_floor_gamma=soft_floor_gamma
        )
        result = custom.forward(pos_var)
        print("custom_cpu_result = ", result.data)
        gradient = torch.ones_like(result)
        result.backward(gradient)
        grad = pos_var.grad.clone()
        print("custom_grad_cpu = ", grad.data)

        np.testing.assert_allclose(result.data.numpy(), golden_value, atol=1e-6)
        np.testing.assert_allclose(
            grad.data.numpy(), golden_grad, rtol=1e-6, atol=1e-6
        )

        # test gpu
        if configure.compile_configurations[
            "CUDA_FOUND"] == "TRUE" and torch.cuda.device_count():
            if pos_var.grad is not None:
                pos_var.grad.zero_()
            custom_cuda = soft_floor.SoftFloor(
                xl=xl,
                yl=yl,
                slr_width=slr_width,
                slr_height=slr_height,
                num_slrX=num_slrX,
                num_slrY=num_slrY,
                soft_floor_gamma=soft_floor_gamma.cuda()
            )
            result_cuda = custom_cuda.forward(pos_var.cuda())
            print("custom_cuda_result = ", result_cuda.data.cpu())
            gradient = torch.ones_like(result_cuda)
            result_cuda.backward(gradient)
            grad_cuda = pos_var.grad.clone()
            print("custom_grad_cuda = ", grad_cuda.data.cpu())

            np.testing.assert_allclose(
                result_cuda.data.cpu().numpy(), golden_value, atol=1e-6
            )
            np.testing.assert_allclose(
                grad.data.cpu().numpy(), golden_grad, rtol=1e-6, atol=1e-6
            )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
