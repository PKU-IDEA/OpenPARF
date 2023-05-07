import torch
from torch import nn
import logging
from openparf.py_utils import stopwatch
from openparf.ops.stable_div import stable_div
from openparf import configure
from . import fence_region_checker_cpp

# if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
#     from . import fence_region_checker_cuda

logger = logging.getLogger(__name__)


class FenceRegionChecker(nn.Module):
    def __init__(self, fence_region_boxes, inst_sizes, inst_avail_crs):
        super(FenceRegionChecker, self).__init__()
        self.fence_region_boxes = None
        self.half_inst_sizes = None
        self.inst_avail_crs = None
        self.reset(fence_region_boxes, inst_sizes, inst_avail_crs)

    def reset(self, fence_region_boxes, inst_sizes, inst_avail_crs):
        self.fence_region_boxes = fence_region_boxes
        self.half_inst_sizes = None if inst_sizes is None else inst_sizes.mul(0.5)
        self.inst_avail_crs =inst_avail_crs

    def forward(self,
                inst_pos):
        """
        :param inst_pos: center of instances, array of (x, y) pairs, shape of (#instance, )
        :return: The number of instances within each fence region, tensor with shape of (#fence_regions,)
        """
        assert self.inst_avail_crs is not None
        assert self.half_inst_sizes is not None
        assert self.fence_region_boxes is not None

        num_insts = int(self.half_inst_sizes.shape[0])

        assert self.half_inst_sizes.shape == (num_insts, 2)
        assert inst_pos.shape == (num_insts, 2)

        func = fence_region_checker_cpp.forward

        return func(
            self.fence_region_boxes.cpu(),
            self.half_inst_sizes.cpu(),
            self.inst_avail_crs,
            inst_pos.cpu()
        )
