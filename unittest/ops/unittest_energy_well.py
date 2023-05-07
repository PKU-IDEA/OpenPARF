import os
import sys
import unittest
import torch
import numpy as np

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = os.path.abspath(sys.argv[1])
print("use project_dir = %s" % project_dir)

sys.path.append(project_dir)
from openparf.ops.energy_well import energy_well

sys.path.pop()


class EnergyWellUnittest(unittest.TestCase):
    def test_energy_well(self):
        dtype = torch.float32

        # box format: (xl, yl, xr, yr)

        well_boxes = torch.Tensor([[0, 0, 2, 2], [3, 3, 4, 10]]).to(dtype)

        inst_boxes = torch.Tensor([[1, 1, 5, 3], [2, 5, 6, 7]]).to(dtype)

        inst_pos = torch.cat((((inst_boxes[..., 0] + inst_boxes[..., 2]) * 0.5).unsqueeze(dim=1),
                              ((inst_boxes[..., 1] + inst_boxes[..., 3]) * 0.5).unsqueeze(dim=1)), dim=1)

        inst_sizes = torch.cat(((inst_boxes[..., 2] - inst_boxes[..., 0]).unsqueeze(dim=1),
                                (inst_boxes[..., 3] - inst_boxes[..., 1]).unsqueeze(dim=1)), dim=1)

        inst_areas = inst_sizes[..., 0] * inst_sizes[..., 1]

        energy_function_exponent = torch.Tensor([2, 2]).to(dtype)
        inst_to_well_indexes = torch.LongTensor([0, 1]).to(torch.int32)

        energy_well_op = energy_well.EnergyWell(
            well_boxes=well_boxes,
            energy_function_exponents=energy_function_exponent,
            inst_areas=inst_areas,
            inst_sizes=inst_sizes,
            inst_to_well_indexes=inst_to_well_indexes,
        )

        inst_pos.requires_grad_(True)
        relative_well_energy = energy_well_op(inst_pos)
        np.testing.assert_allclose(relative_well_energy.detach(), torch.Tensor([1, 0]))

        relative_well_energy_sum = relative_well_energy.sum()
        relative_well_energy_sum.backward()
        inst_pos_grad = inst_pos.grad
        np.testing.assert_allclose(inst_pos_grad.detach(), torch.Tensor([[2, 0], [0, 0]]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
