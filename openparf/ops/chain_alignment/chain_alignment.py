import torch
import pdb

from openparf import configure
from . import chain_alignment_cpp

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import chain_alignment_cuda


class ChainAlignment(object):
    def __init__(self, params, data_cls):
        super(ChainAlignment, self).__init__()
        self.chain_cla_ids = data_cls.chain_cla_ids
        self.inst_sizes_max = data_cls.inst_sizes_max
        self.num_threads = params.num_threads
        self.offset_from_grav_core = chain_alignment_cpp.buildOffset(
            self.chain_cla_ids.bs.cpu(),
            self.chain_cla_ids.b_starts.cpu(),
            self.inst_sizes_max.cpu()
        ).to(self.inst_sizes_max.device)

    def forward(self, pos):
        foo = chain_alignment_cuda.forward if pos.is_cuda else chain_alignment_cpp.forward
        with torch.no_grad():
            foo(pos,
                self.chain_cla_ids.bs,
                self.chain_cla_ids.b_starts,
                self.offset_from_grav_core, self.num_threads)

    def __call__(self, pos):
        return self.forward(pos)
