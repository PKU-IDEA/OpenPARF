#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : metric.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.28.2020
# Last Modified Date: 08.26.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>
##

import time
import torch
import numpy as np

class OptIter(object):
    """ Record iterations in optimization
    """

    def __init__(self, iteration, iter_eta, iter_gamma, iter_lambda, iter_sub):
        self.iteration = iteration
        self.iter_eta = iter_eta
        self.iter_gamma = iter_gamma
        self.iter_lambda = iter_lambda
        self.iter_sub = iter_sub

    def __str__(self):
        content = "Iter %4d" % self.iteration
        content += " (%2d, %4d, %2d, %2d)" % (self.iter_eta, self.iter_gamma, self.iter_lambda,
                                          self.iter_sub)
        return content

    def __repr__(self):
        return self.__str__()


def array2str(arr):
    """ Convert array to string
    """
    content = "["
    delimiter = ""
    for v in arr:
        content += "%s%.3E" % (delimiter, v)
        delimiter = ", "
    content += "]"
    return content

def iarray2str(arr):
    content = "["
    delimiter = ""
    for v in arr:
        content += "%s%2d" % (delimiter, v)
        delimiter = ", "
    content += "]"
    return content


def array2d2str(arr):
    """ Convert array to string
    """
    content = "["
    delimiter = ""
    for v in arr:
        content += "%s[%g %g]" % (delimiter, v[0], v[1])
        delimiter = ", "
    content += "]"
    return content


class EvalMetric(object):
    """
    @brief evaluation metrics at one step
    """

    def __init__(self, params, opt_iter):
        """
        @brief initialization
        @param opt_iter optimization step
        """
        self.params = params
        self.opt_iter = opt_iter
        self.objective = None
        self.wirelength = None
        self.density = None
        self.lambdas = None
        self.step_size = None
        self.hpwl = None
        self.overflow = None
        self.gamma = None
        self.eta = None
        self.ck_illegal_insts_num, self.movable_insts_num = None, None
        self.cr_max_displacement = None
        self.eval_time = None
        self.fence_region = None
        self.grad_dicts = None
        self.obj_terms_dict = None
        self.cr_ck_count = None
        self.current_grad = None
        self.at_avg_grad_norms = None

    def __str__(self):
        """
        @brief convert to string
        """
        content = ""
        if self.opt_iter is not None:
            content += str(self.opt_iter)
        if self.objective is not None:
            content += ", Obj %.6E" % self.objective
        if self.wirelength is not None:
            content += ", WL %.3E" % self.wirelength
        if self.density is not None:
            content += ", Density %s" % array2str(self.density)
        if self.lambdas is not None:
            content += ", Lambdas "
            content += array2str(self.lambdas)
        if self.step_size is not None:
            content += ", Step size "
            content += "%.3E" % self.step_size
        if self.hpwl is not None:
            content += ", HPWL %.3E (%g*%.3E + %g*%.3E)" % (
                    self.hpwl[0] * self.params.wirelength_weights[0] \
                            + self.hpwl[1] * self.params.wirelength_weights[1],
                    self.params.wirelength_weights[0], self.hpwl[0],
                    self.params.wirelength_weights[1], self.hpwl[1])
        if self.overflow is not None:
            content += ", Overflow "
            content += array2str(self.overflow)
        if self.gamma is not None:
            content += ", gamma %.3E" % self.gamma
        if self.eta is not None:
            content += ", eta %.3E" % self.eta
        if self.at_avg_grad_norms is not None:
            content += ", avg_at_grad_norm " + array2str(self.at_avg_grad_norms)
        if self.ck_illegal_insts_num is not None:
            content += ", ck illegal instances : {}/{}, dist-max: {}".format(
                self.ck_illegal_insts_num,
                self.movable_insts_num,
                self.cr_max_displacement)
        if self.cr_ck_count is not None:
            content += ", CR-CK Count {}".format(iarray2str(self.cr_ck_count))
        if self.grad_dicts is not None:
            d_o_w = self.grad_dicts['density_grad_norm'] /  self.grad_dicts['wirelength_grad_norm']
            if 'fence_region_grad_norm' in self.grad_dicts:
                f_o_w = self.grad_dicts['fence_region_grad_norm'] /  self.grad_dicts['wirelength_grad_norm']
                content += ", nomalized-grad(wirelength/density/clock): %.3E/%.3E/%.3E" % (1, d_o_w, f_o_w)
            else:
                content += ", nomalized-grad(wirelength/density): %.3E/%.3E" % (1, d_o_w)

        if self.eval_time is not None:
            content += ", time %.3fms" % (self.eval_time * 1000)

        return content

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def evaluate(self, data_cls, ops, var):
        """
        @brief evaluate metrics
        @param placedb placement database
        @param ops a list of ops
        @param var variables
        """
        tt = time.time()
        with torch.no_grad():
            if "objective" in ops:
                obj, self.obj_terms_dict = ops["objective"](var)
                self.objective = obj.data
            if "wirelength" in ops:
                self.wirelength = ops["wirelength"](var).data
            if "density" in ops:
                self.density = ops["density"](var).data
            if "hpwl" in ops:
                self.hpwl = ops["hpwl"](var).data
            if "overflow" in ops:
                self.overflow = ops["overflow"](var)
            if self.current_grad is not None:
                self.at_avg_grad_norms = []
                for at_type in range(data_cls.num_area_types):
                    at_grad = self.current_grad[data_cls.area_type_inst_groups[at_type]]
                    if at_grad.size()[0] > 0:
                        avg_at_grad_norm = at_grad.norm(dim=1).mean().item()
                    else:
                        avg_at_grad_norm = 0
                    self.at_avg_grad_norms.append(avg_at_grad_norm)
            # if "fence_region" in ops:
            #     self.fence_region = ops["fence_region"](var).sum()
        self.eval_time = time.time() - tt
