#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : flow.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.21.2020
# Last Modified Date: 08.10.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

import time
import logging
import torch
import os

from . import openparf as of
from .placement import placer
from openparf import configure
if configure.compile_configurations["ENABLE_ROUTER"] == "ON":
    from .routing import router
import yaml
import os.path as osp


def place(params, pl_path):
    tt = time.time()

    torch.set_num_threads(params.num_threads)

    db = of.database.Database(0)
    if params.benchmark_format == "bookshelf":
        db.readBookshelf(params.aux_input)
    elif params.benchmark_format == "xarch":
        db.readXML(params.xml_input)
        db.readVerilog(params.verilog_input)
    elif params.benchmark_format == "flexshelf":
        with open(params.design_input, "r") as f:
            data = yaml.safe_load(f)
            netlist_file = osp.join(osp.dirname(params.design_input),
                                    data["netlist"])
            place_file = osp.join(osp.dirname(params.design_input),
                                  data["place"])
        db.readFlexshelf(params.layout_input, netlist_file, place_file)
    else:
        raise RuntimeError("Unknown benchmark format: {}".format(
            params.benchmark_format))
    print("read benchmark takes %.3f seconds" % (time.time() - tt))

    # extract information from params
    place_params = of.database.PlaceParams()
    place_params.benchmark_name = params.benchmark_name
    for value_map in params.gp_model2area_types_map.values():
        for at_name in value_map.keys():
            if not at_name.startswith("is"):
                if at_name not in place_params.at_name2id_map:
                    place_params.at_name2id_map[at_name] = len(
                        place_params.at_name2id_map)
                    place_params.area_type_names.append(at_name)
    place_params.x_wirelength_weight = params.wirelength_weights[0]
    place_params.y_wirelength_weight = params.wirelength_weights[1]
    place_params.x_cnp_bin_size = params.clock_network_planner_bin_sizes[0]
    place_params.y_cnp_bin_size = params.clock_network_planner_bin_sizes[1]
    place_params.honor_clock_region_constraints = params.honor_clock_region_constraints
    place_params.honor_half_column_constraints = params.honor_half_column_constraints
    place_params.cnp_utplace2_enable_packing = params.cnp_utplace2_enable_packing
    place_params.cnp_utplace2_max_num_clock_pruning_per_iteration = params.cnp_utplace2_max_num_clock_pruning_per_iteration
    place_params.clock_region_capacity = params.clock_region_capacity
    place_params.model_area_types.resize(db.design().numModels())
    place_params.model_is_lut.assign(db.design().numModels(), 0)
    place_params.model_is_ff.assign(db.design().numModels(), 0)
    place_params.model_is_clocksource.assign(db.design().numModels(), 0)
    for model_name, value_map in params.gp_model2area_types_map.items():
        model_id = db.design().modelId(model_name)
        for at_name, size in value_map.items():
            if at_name.startswith("is"):
                if at_name == "isLUT":
                    place_params.model_is_lut[model_id] = size
                elif at_name == "isFF":
                    place_params.model_is_ff[model_id] = size
                elif at_name == "isClockSource":
                    place_params.model_is_clocksource[model_id] = size
            else:
                at_id = place_params.at_name2id_map[at_name]
                model_area_type = of.database.ModelAreaType()
                model_area_type.model_id = model_id
                model_area_type.area_type_id = at_id
                model_area_type.model_size.set(size[0], size[1])
                place_params.model_area_types[model_id].append(model_area_type)
    place_params.resource2area_types.resize(
        db.layout().resourceMap().numResources())
    for resource_name, at_names in params.gp_resource2area_types_map.items():
        resource_id = db.layout().resourceMap().resourceId(resource_name)
        for at_name in at_names:
            at_id = place_params.at_name2id_map[at_name]
            place_params.resource2area_types[resource_id].append(at_id)

    def getResourceCategory(value):
        if value == "LUTL":
            return of.ResourceCategory.kLUTL
        elif value == "LUTM":
            return of.ResourceCategory.kLUTM
        elif value == "FF":
            return of.ResourceCategory.kFF
        elif value == "Carry":
            return of.ResourceCategory.kCarry
        elif value == "SSSIR":
            return of.ResourceCategory.kSSSIR
        elif value == "SSMIR":
            return of.ResourceCategory.kSSMIR
        else:
            assert 0, "unknown ResourceCategory %s" % (value)

    place_params.resource_categories.assign(
        db.layout().resourceMap().numResources(), of.ResourceCategory.kUnknown)
    for resource_name, category in params.resource_categories.items():
        resource_id = db.layout().resourceMap().resourceId(resource_name)
        place_params.resource_categories[resource_id] = getResourceCategory(
            category)
    # area adjustment
    place_params.adjust_area_at_ids.clear()
    for at_name in params.gp_adjust_area_types:
        at_id = place_params.at_name2id_map[at_name]
        place_params.adjust_area_at_ids.append(at_id)
    # architecture
    place_params.CLB_capacity = int(params.CLB_capacity)
    place_params.BLE_capacity = int(params.BLE_capacity)
    place_params.num_ControlSets_per_CLB = int(params.num_ControlSets_per_CLB)
    place_params.enable_carry_chain_packing = params.carry_chain_module_name != ""
    place_params.carry_chain_module_name = params.carry_chain_module_name
    place_params.ssr_chain_module_name = params.ssr_chain_module_name
    # print
    lines = [["AreaType", "(id)", "=>", "Resources",
              "(id)", "=>", "Models", "(id)"]]
    for at_id in range(len(place_params.at_name2id_map)):
        for key, value in place_params.at_name2id_map.items():
            if value == at_id:
                at_name = key
                break

        line = [at_name, "(%d)" % at_id, "=>"]
        for resource_id in range(len(place_params.resource2area_types)):
            for value in place_params.resource2area_types[resource_id]:
                if value == at_id:
                    resource_name = db.layout().resourceMap().resource(resource_id).name()
                    line.extend([resource_name, "(%d)" % resource_id])
        line.append("=>")
        for model_id in range(len(place_params.model_area_types)):
            for value in place_params.model_area_types[model_id]:
                if value.area_type_id == at_id:
                    model_name = db.design().model(model_id).name()
                    line.extend([model_name, "(%d)" % model_id])
        lines.append(line)
    num_cols = 0
    for line in lines:
        num_cols = max(num_cols, len(line))
    widths = [0] * num_cols
    for line in lines:
        for j, col in enumerate(line):
            widths[j] = max(widths[j], len(col))
    content = "\n"
    for line in lines:
        content += " ".join((val.ljust(width)
                            for val, width in zip(line, widths)))
        content += "\n"
    logging.info(content[:-1])

    os.makedirs(params.plot_dir, exist_ok=True)

    placedb = of.database.PlaceDB(db, place_params)
    place_engine = placer.Placer(params, placedb)
    place_engine()
    place_engine.write(pl_path)
    logging.info("placement takes %.3f seconds" % (time.time() - tt))

def route(params, pl_path):
    tt = time.time()
    route_engine = router.Router(params)
    route_engine(pl_path)
    logging.info("route takes %.3f seconds" % (time.time() - tt))