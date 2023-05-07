##
# @file   unittest_placedb.py
# @author Yibo Lin
# @date   Apr 2020
#

import pdb
import os
import sys
import unittest

if len(sys.argv) < 2:
    print("usage: python script.py [project_dir] test_dir")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
elif len(sys.argv) < 3:
    print("usage: python script.py [project_dir] test_dir")
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dir = sys.argv[1]
else:
    project_dir = sys.argv[1]
    test_dir = sys.argv[2]
print("use project_dir = %s" % (project_dir))

sys.path.append(project_dir)
import openparf.openparf as of
sys.path.pop()


class PlaceDBTest(unittest.TestCase):
    def testSample1(self):
        db = of.database.Database(0)
        db.readBookshelf(test_dir + "/sample1/design.aux")
        design = db.design()
        top_module_inst_id = design.topModuleInstId()
        self.assertTrue(top_module_inst_id < design.numModuleInsts())
        top_module_inst = design.topModuleInst()
        self.assertTrue(top_module_inst)
        netlist = top_module_inst.netlist()

        placedb = of.database.PlaceDB(db)

        # test netlist
        self.assertEqual(placedb.instPins().size1(), netlist.numInsts())
        self.assertEqual(placedb.instPins().size(), len(placedb.pin2Inst()))
        self.assertEqual(placedb.netPins().size1(), netlist.numNets())
        self.assertEqual(placedb.netPins().size(), len(placedb.pin2Net()))

        for pin_id in range(placedb.numPins()):
            inst_id = placedb.pin2Inst(pin_id)
            net_id = placedb.pin2Net(pin_id)
            pin = netlist.pin(pin_id)
            self.assertEqual(pin_id, pin.id())
            # be careful about new/old instance index
            self.assertEqual(inst_id, placedb.newInstId(pin.instId()))
            self.assertEqual(net_id, pin.netId())
            self.assertTrue(pin_id in placedb.instPins().at(inst_id))
            self.assertTrue(pin_id in placedb.netPins().at(net_id))

        # test layout
        layout = db.layout()
        self.assertEqual(len(placedb.binMapDims()), 7)
        self.assertEqual(len(placedb.binMapSizes()), 7)
        for i in range(7):
            self.assertEqual(len(placedb.binCapacityMap(i)),
                             placedb.binMapDims()[i].product())

        # check whether all the resources in sites are distributed correctly to bins
        site_map = layout.siteMap()
        num_resources = layout.resourceMap().numResources()
        num_area_types = placedb.numAreaTypes()

        content = ""
        for resource_id in range(num_resources):
            resource = layout.resourceMap().resource(resource_id)
            area_types = placedb.resourceAreaTypes(resource.name())
            content += "%d (%s) -> %s\n" % (resource.id(), resource.name(),
                                            str(area_types.tolist()))
        self.assertEqual(
            content, """\
0 (LUT) -> [0]
1 (FF) -> [2]
2 (CARRY8) -> [3]
3 (DSP48E2) -> [5]
4 (RAMB36E2) -> [6]
5 (IO) -> [4]
""")

        area_type_total_sites = [0] * num_area_types
        for site in site_map:
            site_type = layout.siteType(site)
            for resource_id in range(num_resources):
                if site_type.resourceCapacity(resource_id) > 0:
                    resource = layout.resourceMap().resource(resource_id)
                    area_types = placedb.resourceAreaTypes(resource.name())
                    for area_type in area_types:
                        if area_type < num_area_types:
                            area_type_total_sites[area_type] += site.bbox().area()

        area_type_total_bins = [0] * num_area_types
        for area_type in range(num_area_types):
            for bin_cap in placedb.binCapacityMap(area_type):
                area_type_total_bins[area_type] += bin_cap

        for area_type in range(num_area_types):
            self.assertAlmostEqual(area_type_total_sites[area_type],
                                   area_type_total_bins[area_type])

        #for model in design.models():
        #  print(model.name())
        #  print(placedb.modelSize(model.name()))

        # check model sizes
        self.assertAlmostEqual(placedb.modelSize("LUT1").product(), 1.0 / 16)
        self.assertAlmostEqual(placedb.modelSize("LUT2").product(), 1.0 / 16)
        self.assertAlmostEqual(placedb.modelSize("LUT3").product(), 1.0 / 16)
        self.assertAlmostEqual(placedb.modelSize("LUT4").product(), 1.0 / 8)
        self.assertAlmostEqual(placedb.modelSize("LUT5").product(), 1.0 / 8)
        self.assertAlmostEqual(placedb.modelSize("LUT6").product(), 1.0 / 8)
        self.assertAlmostEqual(placedb.modelSize("FDRE").product(), 1.0 / 16)
        self.assertAlmostEqual(placedb.modelSize("CARRY8").product(), 1.0)
        self.assertAlmostEqual(placedb.modelSize("DSP48E2").product(), 2.5)
        self.assertAlmostEqual(placedb.modelSize("RAMB36E2").product(), 5.0)
        # IO is fixed, so I do not care

        # check instance sizes
        for inst_id in range(placedb.numInsts()):
            inst_size = placedb.instSize(inst_id)
            inst_area_type = placedb.instAreaType(inst_id)
            # be careful about new/old instance index
            inst = netlist.inst(placedb.oldInstId(inst_id))
            model_id = inst.attr().modelId()
            model = design.model(model_id)
            if model.name() in [
                    "LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"
            ]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("LUT"))
            elif model.name() in ["FDRE"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("FF"))
            elif model.name() in ["CARRY8"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("CARRY8"))
            elif model.name() in ["DSP48E2"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("DSP48E2"))
            elif model.name() in ["RAMB36E2"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("RAMB36E2"))

    def testSample2(self):
        db = of.database.Database(0)
        db.readBookshelf(test_dir + "/sample2/design.aux")
        design = db.design()
        top_module_inst_id = design.topModuleInstId()
        self.assertTrue(top_module_inst_id < design.numModuleInsts())
        top_module_inst = design.topModuleInst()
        self.assertTrue(top_module_inst)
        netlist = top_module_inst.netlist()

        placedb = of.database.PlaceDB(db)

        # test netlist
        self.assertEqual(placedb.instPins().size1(), netlist.numInsts())
        self.assertEqual(placedb.instPins().size(), len(placedb.pin2Inst()))
        self.assertEqual(placedb.netPins().size1(), netlist.numNets())
        self.assertEqual(placedb.netPins().size(), len(placedb.pin2Net()))

        for pin_id in range(placedb.numPins()):
            inst_id = placedb.pin2Inst(pin_id)
            net_id = placedb.pin2Net(pin_id)
            pin = netlist.pin(pin_id)
            self.assertEqual(pin_id, pin.id())
            # be careful about new/old instance index
            self.assertEqual(inst_id, placedb.newInstId(pin.instId()))
            self.assertEqual(net_id, pin.netId())
            self.assertTrue(pin_id in placedb.instPins().at(inst_id))
            self.assertTrue(pin_id in placedb.netPins().at(net_id))

        # test layout
        layout = db.layout()
        num_resources = layout.resourceMap().numResources()
        num_area_types = placedb.numAreaTypes()
        self.assertEqual(num_resources, 8)
        self.assertEqual(len(placedb.binMapDims()), num_area_types)
        self.assertEqual(len(placedb.binMapSizes()), num_area_types)
        for i in range(num_area_types):
            self.assertEqual(len(placedb.binCapacityMap(i)),
                             placedb.binMapDims()[i].product())

        # check whether all the resources in sites are distributed correctly to bins
        site_map = layout.siteMap()

        content = ""
        for resource_id in range(num_resources):
            resource = layout.resourceMap().resource(resource_id)
            area_types = placedb.resourceAreaTypes(resource.name())
            content += "%d (%s) -> %s\n" % (resource.id(), resource.name(),
                                            str(area_types.tolist()))
        self.assertEqual(
            content, """\
0 (LUTL) -> [0]
1 (LUTM) -> [0, 1]
2 (DFF) -> [2]
3 (CLA4) -> [3]
4 (GCU) -> [5]
5 (BRAM) -> [6]
6 (HRAM) -> [7]
7 (IO) -> [4]
""")

        area_type_total_sites = [0] * num_area_types
        for site in site_map:
            site_type = layout.siteType(site)
            for resource_id in range(num_resources):
                if site_type.resourceCapacity(resource_id) > 0:
                    resource = layout.resourceMap().resource(resource_id)
                    area_types = placedb.resourceAreaTypes(resource.name())
                    for area_type in area_types:
                        if area_type < num_area_types:
                            area_type_total_sites[area_type] += site.bbox().area()

        area_type_total_bins = [0] * num_area_types
        for area_type in range(num_area_types):
            for bin_cap in placedb.binCapacityMaps()[area_type]:
                area_type_total_bins[area_type] += bin_cap

        for area_type in range(num_area_types):
            self.assertAlmostEqual(area_type_total_sites[area_type],
                                   area_type_total_bins[area_type])

        #for model in design.models():
        #  print(model.name())
        #  print(placedb.modelSize(model.name()))
        # check model sizes
        self.assertAlmostEqual(placedb.modelSize("LUT5").product(), 1.0 / 8)
        self.assertAlmostEqual(placedb.modelSize("LUT6").product(), 1.0 / 8)
        self.assertAlmostEqual(placedb.modelSize("DFF").product(), 1.0 / 16)
        self.assertAlmostEqual(placedb.modelSize("CLA4").product(), 1.0 / 2)
        self.assertAlmostEqual(placedb.modelSize("LRAM").product(), 1.0 / 8)
        self.assertAlmostEqual(placedb.modelSize("SHIFT").product(), 1.0 / 8)
        self.assertAlmostEqual(placedb.modelSize("GCU0").product(), 4.0)
        self.assertAlmostEqual(placedb.modelSize("BRAM36K").product(), 4.0)
        self.assertAlmostEqual(placedb.modelSize("HRAM").product(), 4.0)
        self.assertAlmostEqual(
            placedb.modelSize("GLOBAL_CLK").product(), 1.0 / 24)
        self.assertAlmostEqual(
            placedb.modelSize("GCLK_BUF").product(), 1.0 / 24)

        # check instance sizes
        for inst_id in range(placedb.numInsts()):
            inst_size = placedb.instSize(inst_id)
            inst_area_type = placedb.instAreaType(inst_id)
            # be careful about new/old instance index
            inst = netlist.inst(placedb.oldInstId(inst_id))
            model_id = inst.attr().modelId()
            model = design.model(model_id)
            if model.name() in [
                    "LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"
            ]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("LUTL"))
            elif model.name() in ["DFF"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("DFF"))
            elif model.name() in ["CLA4"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("CLA4"))
            elif model.name() in ["BRAM36K"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("BRAM"))
            elif model.name() in ["HRAM"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("HRAM"))
            elif model.name() in ["OUTPAD" "INPAD" "GCLK_BUF" "GLOBAL_CLK"]:
                self.assertEqual(inst_area_type,
                                 placedb.resourceAreaTypes("IO"))
            elif model.name() in ["LRAM", "SHIFT"]:
                self.assertEqual(inst_area_type, placedb.resourceAreaTypes("LUTM"))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    elif len(sys.argv) < 3:
        sys.argv.pop()
    else:
        sys.argv.pop()
        sys.argv.pop()
    unittest.main()
