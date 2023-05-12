#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# File         : router.py
# Author       : Jing Mai (jingmai@pku.edu.cn)
# Created Date : March 30 2023, 21:43:59, Thursday
# Modified By  : Jing Mai (jingmai@pku.edu.cn)
# Last Modified: April 27 2023, 16:18:49, Thursday
# Description  :
# Copyright (c) 2023 Jing Mai
# -----
# HISTORY:
# Date         	By     	Comments
# -------------	-------	----------------------------------------------------------
###
import torch.nn as nn
import logging
import os.path as osp
import sys, os

sys.path.append(os.path.dirname(__file__))
import router_cpp
sys.path.pop()

logger = logging.getLogger(__name__)


class Router(nn.Module):
    def __init__(self, params):
        super(Router, self).__init__()
        self.params = params

    def forward(self, pl_path):
        net_path = osp.join(osp.dirname(self.params.aux_input), "design.nets")
        node_path = osp.join(osp.dirname(self.params.aux_input), "design.nodes")
        routing_output_path = osp.join(
            self.params.result_dir, self.params.benchmark_name + ".xml"
        )
        router_cpp.forward(
            self.params.routing_architecture_input,
            pl_path,
            net_path,
            node_path,
            routing_output_path
        )

if __name__ == '__main__':
    class Params(object):
        aux_input = "data/ispd19_test1/ispd19_test1.aux"
        routing_architecture_input = "data/ispd19_test1/ispd19_test1.routing.arch"
        output_dir = "data/ispd19_test1"
        benchmark_name = "ispd19_test1"

    params = Params()
    router = Router(params)
    router("data/ispd19_test1/ispd19_test1.pl")