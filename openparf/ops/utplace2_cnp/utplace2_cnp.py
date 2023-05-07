from . import utplace2_cnp_torch

class UTPlace2CNP(object):
    def __init__(self, placedb):
        self.placedb = placedb

    def forward(self, pos, areas):
        local_pos = pos.cpu() if pos.is_cuda else pos
        local_areas = areas.cpu() if areas.is_cuda else areas
        return utplace2_cnp_torch.forward(self.placedb, local_pos, local_areas)

    def __call__(self, pos, areas):
        return self.forward(pos, areas)
