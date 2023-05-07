import logging
import torch
import pdb

logger = logging.getLogger(__name__)

class InstCrMover(object):
    def __init__(self, cr_boxes: torch.tensor):
        self.num_crs = cr_boxes.size()[0]
        self._cr_centers = torch.zeros(size=(self.num_crs, 2), dtype=cr_boxes.dtype,
                                     device=cr_boxes.device)
        self._cr_centers[:, 0] = (cr_boxes[:, 0] + cr_boxes[:, 2]) * 0.5
        self._cr_centers[:, 1] = (cr_boxes[:, 1] + cr_boxes[:, 3]) * 0.5

        self._half_cr_size_x = (cr_boxes[0, 2] - cr_boxes[0, 0]) * 0.5
        self._half_cr_size_y = (cr_boxes[0, 3] - cr_boxes[0, 1]) * 0.5

    def __call__(self, inst_to_crs: torch.tensor):
        # return self._cr_centers.index_select(dim=0, index=inst_to_crs.to(torch.long)).contiguous()
        pos = self._cr_centers.index_select(dim=0, index=inst_to_crs.to(torch.long)).contiguous()
        # add noise
        scale = 1e-3
        noise = torch.randn(pos.size(), dtype=pos.dtype)
        noise[:, 0] *= self._half_cr_size_x * scale
        noise[:, 1] *= self._half_cr_size_y * scale
        noise[:, 0] = torch.clamp(noise[:, 0], min=-self._half_cr_size_x, max=self._half_cr_size_x)
        noise[:, 1] = torch.clamp(noise[:, 1], min=-self._half_cr_size_y, max=self._half_cr_size_y)
        pos += noise
        return pos

if __name__ == '__main__':
    torch.manual_seed(114514)
    cr_boxes = torch.arange(40).reshape(10, 4).to(torch.float32)
    inst_to_crs = torch.arange(10)
    inst_cr_mover = InstCrMover(cr_boxes)
    rv = inst_cr_mover(inst_to_crs)
    print(rv)