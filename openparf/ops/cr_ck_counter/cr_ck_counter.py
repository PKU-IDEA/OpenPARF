from torch import nn
import logging

from . import cr_ck_counter_cpp

logger = logging.getLogger(__name__)


class CrCkCounter(nn.Module):
    def __init__(self, inst_cks, num_crs, num_cks, placedb):
        super(CrCkCounter, self).__init__()
        self.inst_cks = inst_cks
        self.num_crs = num_crs
        self.num_cks = num_cks
        self.inst_cks = inst_cks
        self.placedb = placedb
    
    def forward(self, pos):
        func = cr_ck_counter_cpp.forward
        return func(pos.cpu(), self.inst_cks, self.num_crs, self.num_cks, self.placedb)
