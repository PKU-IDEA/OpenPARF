import torch
from torch import nn
import math
from PIL import Image
import pdb


from openparf import configure
from .model import model
from . import congestion_prediction_cpp

if configure.compile_configurations["CUDA_FOUND"] == "TRUE":
    from . import congestion_prediction_cuda

class Congestion_prediction(nn.Module):
    def __init__(self,
                 netpin_start,
                 flat_netpin,
                 net_weights,
                 xl,
                 xh,
                 yl,
                 yh,
                 num_bins_x, num_bins_y,
                 unit_horizontal_capacity,
                 unit_vertical_capacity,
                 pinDirects,
                 initial_horizontal_utilization_map=None,
                 initial_vertical_utilization_map=None,
                 initial_pin_density_map=None):
        super(Congestion_prediction, self).__init__()

        self.netpin_start = netpin_start
        self.flat_netpin = flat_netpin
        self.net_weights = net_weights
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.bin_size_x = (xh - xl) / num_bins_x
        self.bin_size_y = (yh - yl) / num_bins_y
        self.unit_horizontal_capacity = unit_horizontal_capacity
        self.unit_vertical_capacity = unit_vertical_capacity
        self.pinDirects = pinDirects
        self.initial_horizontal_utilization_map = initial_horizontal_utilization_map
        self.initial_vertical_utilization_map = initial_vertical_utilization_map
        self.initial_pin_density_map = initial_pin_density_map


    def forward(self,pin_pos):
        #computing eigenvalue
        horizontal_utilization_map = torch.zeros((self.num_bins_x, self.num_bins_y),
                                                 dtype=pin_pos.dtype,
                                                 device=pin_pos.device)
        vertical_utilization_map = torch.zeros_like(horizontal_utilization_map)
        pin_density_map=torch.zeros_like(horizontal_utilization_map)

        function1 = congestion_prediction_cuda.forward if pin_pos.is_cuda else congestion_prediction_cpp.forward
        function1(pin_pos, self.netpin_start, self.flat_netpin,
                  self.net_weights, self.bin_size_x, self.bin_size_y, self.xl,
                  self.yl, self.xh, self.yh, self.num_bins_x, self.num_bins_y,
                  self.pinDirects, horizontal_utilization_map,
                  vertical_utilization_map, pin_density_map)

        # Convert demand to utilization in each bin
        bin_area = self.bin_size_x * self.bin_size_y
        horizontal_utilization_map.mul_(1.0 / 512)
        vertical_utilization_map.mul_(1.0 / 512)
        if self.initial_horizontal_utilization_map is not None:
            horizontal_utilization_map.add_(self.initial_horizontal_utilization_map)
        if self.initial_vertical_utilization_map is not None:
            vertical_utilization_map.add_(self.initial_vertical_utilization_map)
        pin_density_map = pin_density_map / 250
        # route_utilization_map = torch.max(horizontal_utilization_map.abs(), vertical_utilization_map.abs())

        #modify the dimension
        horizontal_utilization_map = horizontal_utilization_map[:,:].reshape(1,168,480)
        vertical_utilization_map = vertical_utilization_map[:,:].reshape(1,168,480)
        pin_density_map = pin_density_map[:,:].reshape(1,168,480)

        #neural network
        function2 = model
        if horizontal_utilization_map.is_cuda:
            horizontal_utilization_map = horizontal_utilization_map.cpu()
            vertical_utilization_map = vertical_utilization_map.cpu()
            pin_density_map = pin_density_map.cpu()
        result = function2(horizontal_utilization_map,
                           vertical_utilization_map,
                           pin_density_map)

        horizontal_utilization_map_result = result[0,0,0:168,:].reshape(168,480).mul_(512/self.unit_horizontal_capacity).to(torch.float64)
        vertical_utilization_map_result = result[0,1,0:168,:].reshape(168,480).mul_(512/self.unit_vertical_capacity).to(torch.float64)
        route_utilization_map_result = torch.max(horizontal_utilization_map_result.abs(), vertical_utilization_map_result.abs())

        return route_utilization_map_result, horizontal_utilization_map_result, vertical_utilization_map_result