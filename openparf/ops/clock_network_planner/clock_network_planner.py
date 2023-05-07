##
# @file   clock_network_planner.py
# @author Yibai Meng
# @date   Sep 2020
#

import torch
from torch import nn
import numpy as np

from . import clock_network_planner_cpp

class ClockNetworkPlannerParam:
    pass
class ClockNetworkPlanner:
    """ 
    @brief Deal with clock constraints. Currently only the assignment of clock regions is implemented
    The pos here is only to supply the type

    Because PyTorch only binds at::Tensor, and not its other types, I'll just
    pass a string
    https://discuss.pytorch.org/t/how-to-pass-python-device-and-dtype-to-c/69577/2
    """
    def __init__(self, placedb, float_type : str, maximum_clock_per_clock_region): 
        """
        @brief initialization 
        @param[in] placedb class PlaceDB, the placement database
        @param[in] float_type The datatype the position vector uses.
        """
        self.param = ClockNetworkPlannerParam()
        self.param.maximum_clock_per_clock_region = maximum_clock_per_clock_region
        self.placedb = placedb
        assert(float_type.lower() in ["float64", "float32"])
        clock_network_planner_cpp.init(self.placedb, float_type.lower())

    def forward(self, pos):
        """
        @brief Assign clock regions based on the current position of instance
               The goal is to satisfy global clock constraints while minimizing 
               the movement needed to move the instances into their respective clock regions. 

               See UTPlaceF 2.0: A High-Performance Clock-Aware FPGA Placement Engine
               doi:10.1145/3174849 for more detail. This implementation is directly
               ported for UTPlaceF 2.0
        
        @param[in] pos Current position of the instances, #(num_insts, 2) shaped torch.Tensor
        @return A map of clock nets to the bounding box of its clock region.
        @return 1st tensor: nodeToCRclock, a #(num_insts) torch.Tensor. res[i] is the clock region id to which instance i is assigned.
                2nd tensor: clkAvailCR, a #(num_clk_nets, clk_region_width, clk_region_height) shaped torch.Tensor. If res[clk_idx][x][y] is 1, then clock clk_id is allowed in x, y, otherwise clk_idx is not allowed in x, y.
        """    
        # The algorithm is too complex to reimplement in CUDA. 
        # TODO: consider implementing it in CUDA, should the runtime be great
        # Further more, it's simply not worth it, considering the already short runtime.
        if pos.is_cuda:
            local_pos = pos.cpu()
        else:
            local_pos = pos
        t = torch.zeros([2, 4], dtype=torch.float64)
        return clock_network_planner_cpp.forward(
                self.placedb, 
                self.param,
                local_pos,
                t)

    def __call__(self, pos):
        return self.forward(pos)

