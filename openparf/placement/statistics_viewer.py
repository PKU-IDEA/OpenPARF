#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : statistics_viewer.py
# Author            : Jing Mai <jingmai@pku.edu.cn>
# Date              : 12.21.2020
# Last Modified Date: 12.21.2020
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
import aim
from openparf.placement import metric


class PlacerStatisticsViewer(aim.Session):
    def __init__(self, dataset: str, *args, **kargs):
        super(PlacerStatisticsViewer, self).__init__(*args, **kargs)
        self.dataset = dataset

    def recordMetric(self, cur_metric: metric.EvalMetric):
        epoch = cur_metric.opt_iter.iteration

        if cur_metric.hpwl is not None:
            hpwl = (cur_metric.hpwl[0] * cur_metric.params.wirelength_weights[0]
                    + cur_metric.hpwl[1] * cur_metric.params.wirelength_weights[1])
            self.track(hpwl.item(), name='hpwl', epoch=epoch, dataset=self.dataset)

        if cur_metric.objective is not None:
            self.track(cur_metric.objective.item(), name='objective', epoch=epoch, dataset=self.dataset)

        if cur_metric.density is not None:
            for idx, value in enumerate(cur_metric.density):
                self.track(value.item(), name='density_{:02d}'.format(idx), epoch=epoch,
                           dataset=self.dataset)

        if cur_metric.lambdas is not None:
            for idx, value in enumerate(cur_metric.lambdas):
                self.track(value.item(), name='lambda_{:02d}'.format(idx), epoch=epoch, dataset=self.dataset)

        if cur_metric.fence_region is not None:
            self.track(cur_metric.fence_region.item(), name='fence_region', epoch=epoch, dataset=self.dataset)
        else:
            self.track(0, name='fence_region', epoch=epoch, dataset=self.dataset)

        if cur_metric.step_size is not None:
            self.track(cur_metric.step_size, name='step_size', epoch=epoch, dataset=self.dataset)

        if cur_metric.overflow is not None:
            for idx, value in enumerate(cur_metric.overflow):
                name = 'overflow_{:02d}'.format(idx)
                self.track(value.item(), name=name, epoch=epoch, dataset=self.dataset)

        if cur_metric.gamma is not None:
            self.track(cur_metric.gamma.item(), name='gamma', epoch=epoch, dataset=self.dataset)

        if cur_metric.eta is not None:
            self.track(cur_metric.eta.item(), name='eta', epoch=epoch, dataset=self.dataset)
        else:
            self.track(0, name='eta', epoch=epoch, dataset=self.dataset)

        if cur_metric.ck_illegal_insts_num is not None:
            self.track(cur_metric.ck_illegal_insts_num,
                       name='ck_illegal_insts_num',
                       epoch=epoch, dataset=self.dataset)
        else:
            self.track(0, name='ck_illegal_insts_num', epoch=epoch, dataset=self.dataset)

        if cur_metric.eval_time is not None:
            self.track(cur_metric.eval_time, name='eval_time', epoch=epoch, dataset=self.dataset)

        if cur_metric.grad_dicts is not None:
            for grad_name in ['wirelength_grad_norm', 'density_grad_norm', 'fence_region_grad_norm']:
                if grad_name in cur_metric.grad_dicts:
                    self.track(cur_metric.grad_dicts[grad_name].item(), name=grad_name,
                               epoch=epoch,
                               dataset=self.dataset)
                else:
                    self.track(0, name=grad_name, epoch=epoch, dataset=self.dataset)

        if cur_metric.obj_terms_dict is not None:
            for term_name in ['wirelength_term', "density_term", "fence_region_term"]:
                if term_name in cur_metric.obj_terms_dict:
                    self.track(cur_metric.obj_terms_dict[term_name].item(), name=term_name,
                               epoch=epoch, dataset=self.dataset)
                else:
                    self.track(0, name=term_name, epoch=epoch, dataset=self.dataset)

    def setParams(self, params_dict):
        self.set_params(params_dict, name='params')
