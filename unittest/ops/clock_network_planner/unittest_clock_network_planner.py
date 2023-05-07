##
# @file   clock_network_planner_unitest.py
# @author Yibai Meng
# @date   Oct 2020
#
import os
import sys
import unittest
import torch
import numpy as np
import math
import time

if len(sys.argv) != 3:
    print("usage: python script.py project_build_dir project_source_dir")
    sys.exit(1)
else:
    project_dir = os.path.abspath(sys.argv[1])
    project_source_dir = os.path.abspath(sys.argv[2])
print("use project_dir = %s, project_source_dir = %s" % (project_dir, project_source_dir))

sys.path.append(project_dir)
from openparf.ops.clock_network_planner import clock_network_planner
import openparf.openparf as of
sys.path.pop()
from params import Params

def load_design(json_file):
        # The following are copied from openparf.py
        # and openparf/flow.py
        # TODO: standardize the way they are loaded
        params = Params()
        params.load(json_file)
        torch.set_num_threads(params.num_threads)
        db = of.database.Database(0)
        db.readBookshelf(project_source_dir + "/" + params.aux_input)
        place_params = of.database.PlaceParams()
        for value_map in params.gp_model2area_types_map.values():
            for at_name in value_map.keys():
                if not at_name.startswith("is"):
                    if at_name not in place_params.at_name2id_map:
                        place_params.at_name2id_map[at_name] = len(place_params.at_name2id_map)
                        place_params.area_type_names.append(at_name)
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
        place_params.resource2area_types.resize(db.layout().resourceMap().numResources())
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

        place_params.resource_categories.assign(db.layout().resourceMap().numResources(), of.ResourceCategory.kUnknown)
        for resource_name, category in params.resource_categories.items():
            resource_id = db.layout().resourceMap().resourceId(resource_name)
            place_params.resource_categories[resource_id] = getResourceCategory(category)
        # architecture
        place_params.CLB_capacity = int(params.CLB_capacity)
        place_params.BLE_capacity = int(params.BLE_capacity)
        place_params.num_ControlSets_per_CLB = int(params.num_ControlSets_per_CLB)
        # print
        lines = [["AreaType", "(id)", "=>", "Resources", "(id)", "=>", "Models", "(id)"]]
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
            content += " ".join((val.ljust(width) for val, width in zip(line, widths)))
            content += "\n"
        placedb = of.database.PlaceDB(db, place_params)
        return placedb, params, db


class ClockNetworkPlannerUnittest(unittest.TestCase):
    def test_cnp(self):
        # First we build the placedb
        print("Loading ISPD2017 CLK FPGA01 File")
        # TODO: Pass the project source dir?
        # TODO: something wrong with the move or copy constructor?
        #  if load_design does not return db, then placedb.isInstFF(0) would segfault.
        placedb, params, db = load_design(project_source_dir + "/unittest/regression/ispd2017/CLK-FPGA01.json")
        # TODO: why placedb.isInstLUTs() also breaks?
        assert(params.dtype in ["float64", "float32"])
        cnp = clock_network_planner.ClockNetworkPlanner(placedb, params.dtype)
        # Then we load the pos
        dump = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/unittest_CLK-FPGA01_pos_400.pt")
        pos = dump[0]
        t1 = time.time()
        res, res2 = cnp(pos)
        t2 = time.time()
        print(t2-t1)
        for i in range(placedb.numInsts()):
            if res[i] == -1 and placedb.instPlaceStatus(i) == of.PlaceStatus.kMovable:
                self.fail("All movable instances must have an assigned clock region.")

if __name__ == '__main__':
    # unittest uses sys.argv to find the main function
    # So we need to pop the not needed arguments.
    # So stupid...
    sys.argv = sys.argv[0:1]
    unittest.main()
