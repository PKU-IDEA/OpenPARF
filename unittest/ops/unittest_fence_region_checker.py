import os
import sys
import unittest
import torch
import random
import numpy as np

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir]")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
    project_dir = os.path.abspath(sys.argv[1])
print("use project_dir = %s" % project_dir)

sys.path.append(project_dir)
from openparf.ops.fence_region_checker import fence_region_checker

sys.path.pop()


def box_generator_factory(layout_xl, layout_yl, layout_xh, layout_yh, max_box_width, max_box_height):
    assert max_box_width <= layout_xh - layout_xl
    assert max_box_height <= layout_yh - layout_yl

    def gen():
        while True:
            width = random.uniform(0, max_box_width)
            height = random.uniform(0, max_box_height)
            xl = random.uniform(layout_xl, layout_xh - width)
            yl = random.uniform(layout_yl, layout_yh - height)
            xh, yh = xl + width, yl + height
            yield xl, yl, xh, yh

    return gen()


def solve(region_boxes, instance_boxes, inst_to_fence_region_indexes) -> torch.Tensor:
    num_regions = len(region_boxes)
    num_instances = len(instance_boxes)
    legal_inst_counts = torch.zeros(num_regions, requires_grad=False, dtype=torch.int32)
    for i in range(num_instances):
        region_index = inst_to_fence_region_indexes[i]
        box_xl, box_yl, box_xh, box_yh = region_boxes[region_index]
        xl, yl, xh, yh = instance_boxes[i]

        is_intersect_x = max(xl, box_xl) >= min(xh, box_xh)
        is_intersect_y = max(yl, box_yl) >= min(yh, box_yh)

        legal_inst_counts[region_index] += is_intersect_x & is_intersect_y
    return legal_inst_counts


class FenceRegionCheckerUnittest(unittest.TestCase):
    def test_fence_region_checker(self):
        random.seed(114514)
        dtype = torch.float32

        layout_xl, layout_yl, layout_xh, layout_yh = 0, 0, 100, 100
        max_region_width, max_region_height = 20, 40
        max_instance_width, max_instance_height = 4, 8
        num_regions = 100
        num_instances = 100000

        region_generator = box_generator_factory(layout_xl, layout_yl, layout_xh, layout_yh,
                                                 max_region_width, max_region_height)
        instance_generator = box_generator_factory(layout_xl, layout_yl, layout_xh, layout_yh,
                                                   max_instance_width, max_instance_height)

        region_boxes = torch.Tensor([next(region_generator) for i in range(num_regions)]).to(dtype)
        instance_boxes = torch.Tensor([next(instance_generator) for i in range(num_instances)]).to(dtype)
        inst_to_fence_region_indexes = torch.Tensor(
            [random.randint(0, num_regions - 1) for i in range(num_instances)]).to(torch.int32)

        inst_pos = torch.cat((((instance_boxes[..., 0] + instance_boxes[..., 2]) * 0.5).unsqueeze(dim=1),
                              ((instance_boxes[..., 1] + instance_boxes[..., 3]) * 0.5).unsqueeze(dim=1)), dim=1)

        inst_sizes = torch.cat(((instance_boxes[..., 2] - instance_boxes[..., 0]).unsqueeze(dim=1),
                                (instance_boxes[..., 3] - instance_boxes[..., 1]).unsqueeze(dim=1)), dim=1)

        ground_truth = solve(
            region_boxes=region_boxes,
            instance_boxes=instance_boxes,
            inst_to_fence_region_indexes=inst_to_fence_region_indexes
        )

        op = fence_region_checker.FenceRegionChecker(
            fence_region_boxes=region_boxes,
            inst_sizes=inst_sizes,
            inst_to_fence_region_indexes=inst_to_fence_region_indexes
        )
        cpu_results = op(inst_pos)

        np.testing.assert_allclose(ground_truth.detach(), cpu_results.detach())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        pass
    else:
        sys.argv.pop()
    unittest.main()
