from torch import nn

from . import static_timing_analysis_cpp


class StaticTimingAnalysis(nn.Module):
    def __init__(self, placedb, data_cls, timing_period):
        super(StaticTimingAnalysis, self).__init__()
        self.placedb = placedb
        self.data_cls = data_cls
        self.timing_period = timing_period

    def forward(self, pin_delays):
        device = pin_delays.device
        pin_arrivals, pin_requires, ignored_net_masks = static_timing_analysis_cpp.forward(
            self.placedb, pin_delays.cpu(), self.data_cls.net_mask_ignore_large.cpu(), self.timing_period)
        pin_slacks = pin_requires - pin_arrivals
        if pin_arrivals.device != device:
            pin_arrivals = pin_arrivals.to(device)
        if pin_requires.device != device:
            pin_requires = pin_requires.to(device)
        if pin_slacks.device != device:
            pin_slacks = pin_slacks.to(device)
        if ignored_net_masks != device:
            ignored_net_masks = ignored_net_masks.to(device)
        return pin_arrivals, pin_requires, pin_slacks, ignored_net_masks
