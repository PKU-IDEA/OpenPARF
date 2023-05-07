from torch import nn
import torch
from torch._C import device
from . import delay_estimation_cpp
import pdb
from hummingbird import ml
from torch.utils.data import dataloader


class DelayEstimation(nn.Module):
    def __init__(self, placedb, data_cls, delay_model, batch_size=1000000):
        super(DelayEstimation, self).__init__()
        self.placedb = placedb
        self.data_cls = data_cls
        self.delay_model = delay_model
        self.batch_size = batch_size

    def forward(self, pos):
        pin_features, ignore_pin_masks = delay_estimation_cpp.forward(
            self.placedb, self.data_cls.net_mask_ignore_large.cpu(), pos.cpu())
        if pin_features.device != pos.device:
            pin_features = pin_features.to(pos.device)
        self.delay_model = self.delay_model.to(pos.device)
        loader = dataloader.DataLoader(
            pin_features, batch_size=self.batch_size)
        pin_delays = [torch.tensor(self.delay_model.predict(samples), device=pos.device)
                      for samples in loader]
        pin_delays = torch.cat(pin_delays)
        pin_delays[ignore_pin_masks] = 0
        return pin_delays, ignore_pin_masks
