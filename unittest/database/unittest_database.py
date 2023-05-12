##
# @file   unittest_database.py
# @author Yibo Lin
# @date   Mar 2020
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


class DatabaseTest(unittest.TestCase):
    def testSample1(self):
        db = of.database.Database(0)
        db.readBookshelf(test_dir + "/sample1/design.aux")
        design = db.design()

        # test model
        self.assertEqual(design.numModels(), 14)
        model_id = design.modelId("Bookshelf.TOP")
        model = design.model(model_id)
        if model.modelType() == of.ModelType.kModule:
            netlist = model.netlist()
            self.assertTrue(netlist)
            self.assertEqual(netlist.numInsts(), 12)
            self.assertEqual(netlist.numNets(), 5)
            for pin in netlist.pins():
                inst = netlist.inst(pin.instId())
                net = netlist.net(pin.netId())
                self.assertTrue(pin.id() in inst.pinIds())
                self.assertTrue(pin.id() in net.pinIds())
                self.assertEqual(inst.id(), netlist.inst(inst.id()).id())
                self.assertEqual(net.id(), netlist.net(net.id()).id())

        # test design
        top_module_inst_id = design.topModuleInstId()
        self.assertTrue(top_module_inst_id < design.numModuleInsts())
        top_module_inst = design.topModuleInst()
        self.assertTrue(top_module_inst)
        netlist = top_module_inst.netlist()
        for pin_id in netlist.pinIds():
            pin = netlist.pin(pin_id)
            inst = netlist.inst(pin.instId())
            net = netlist.net(pin.netId())
            self.assertTrue(pin.id() in inst.pinIds())
            self.assertTrue(pin.id() in net.pinIds())
            self.assertTrue(inst.id() in netlist.instIds())
            self.assertTrue(net.id() in netlist.netIds())

        layout = db.layout()

        # test layout
        site_map = layout.siteMap()
        self.assertTrue(site_map.width(), 168)
        self.assertTrue(site_map.height(), 480)
        for site in site_map:
            if layout.siteType(site).name() in ["SLICE", "SLICEL", "SLICEM"]:
                self.assertEqual(site.bbox().width(), 1)
                self.assertEqual(site.bbox().height(), 1)
            elif layout.siteType(site).name() == "DSP":
                self.assertEqual(site.bbox().width(), 1)
                if (site.siteMapId().y() % 5) == 0:
                    self.assertEqual(site.bbox().height(), 2)
                else:
                    self.assertEqual(site.bbox().height(), 3)
            elif layout.siteType(site).name() == "BRAM":
                self.assertEqual(site.bbox().width(), 1)
                self.assertEqual(site.bbox().height(), 5)
            elif layout.siteType(site).name() == "IO":
                self.assertEqual(site.bbox().width(), 1)
                if site.siteMapId().x() in [0, 67, 104, 167]:
                    self.assertEqual(site.bbox().height(), 60)
                else:
                    self.assertEqual(site.bbox().height(), 30)

        site_type_map = layout.siteTypeMap()
        resource_map = layout.resourceMap()
        self.assertEqual(
            site_type_map.siteType("SLICE").resourceCapacity(
                resource_map.resourceId("LUT")), 16)
        self.assertEqual(
            site_type_map.siteType("SLICE").resourceCapacity(
                resource_map.resourceId("FF")), 16)
        self.assertEqual(
            site_type_map.siteType("SLICE").resourceCapacity(
                resource_map.resourceId("CARRY8")), 1)
        self.assertEqual(
            site_type_map.siteType("DSP").resourceCapacity(
                resource_map.resourceId("DSP48E2")), 1)
        self.assertEqual(
            site_type_map.siteType("BRAM").resourceCapacity(
                resource_map.resourceId("RAMB36E2")), 1)
        self.assertEqual(
            site_type_map.siteType("IO").resourceCapacity(
                resource_map.resourceId("IO")), 64)

        self.assertTrue(
            resource_map.resourceId("LUT") in resource_map.modelResourceIds(
                design.modelId("LUT1")))
        self.assertTrue(
            resource_map.resourceId("LUT") in resource_map.modelResourceIds(
                design.modelId("LUT2")))
        self.assertTrue(
            resource_map.resourceId("LUT") in resource_map.modelResourceIds(
                design.modelId("LUT3")))
        self.assertTrue(
            resource_map.resourceId("LUT") in resource_map.modelResourceIds(
                design.modelId("LUT4")))
        self.assertTrue(
            resource_map.resourceId("LUT") in resource_map.modelResourceIds(
                design.modelId("LUT5")))
        self.assertTrue(
            resource_map.resourceId("LUT") in resource_map.modelResourceIds(
                design.modelId("LUT6")))
        self.assertTrue(
            resource_map.resourceId("FF") in resource_map.modelResourceIds(
                design.modelId("FDRE")))
        self.assertTrue(
            resource_map.resourceId("CARRY8") in resource_map.modelResourceIds(
                design.modelId("CARRY8")))
        self.assertTrue(
            resource_map.resourceId("DSP48E2") in
            resource_map.modelResourceIds(design.modelId("DSP48E2")))
        self.assertTrue(
            resource_map.resourceId("RAMB36E2") in
            resource_map.modelResourceIds(design.modelId("RAMB36E2")))
        self.assertTrue(
            resource_map.resourceId("IO") in resource_map.modelResourceIds(
                design.modelId("IBUF")))
        self.assertTrue(
            resource_map.resourceId("IO") in resource_map.modelResourceIds(
                design.modelId("OBUF")))
        self.assertTrue(
            resource_map.resourceId("IO") in resource_map.modelResourceIds(
                design.modelId("BUFGCE")))

        clock_region_map = layout.clockRegionMap()
        self.assertEqual(clock_region_map.width(), 5)
        self.assertEqual(clock_region_map.height(), 8)
        clock_region_x0y0 = clock_region_map.at(0, 0)
        self.assertEqual(clock_region_x0y0.bbox().xl(), 0)
        self.assertEqual(clock_region_x0y0.bbox().yl(), 0)
        self.assertEqual(clock_region_x0y0.bbox().xh(), 29)
        self.assertEqual(clock_region_x0y0.bbox().yh(), 59)
        clock_region_x0y1 = clock_region_map.at(0, 1)
        self.assertEqual(clock_region_x0y1.bbox().xl(), 0)
        self.assertEqual(clock_region_x0y1.bbox().yl(), 60)
        self.assertEqual(clock_region_x0y1.bbox().xh(), 29)
        self.assertEqual(clock_region_x0y1.bbox().yh(), 119)
        clock_region_x0y2 = clock_region_map.at(0, 2)
        self.assertEqual(clock_region_x0y2.bbox().xl(), 0)
        self.assertEqual(clock_region_x0y2.bbox().yl(), 120)
        self.assertEqual(clock_region_x0y2.bbox().xh(), 29)
        self.assertEqual(clock_region_x0y2.bbox().yh(), 179)
        clock_region_x0y3 = clock_region_map.at(0, 3)
        self.assertEqual(clock_region_x0y3.bbox().xl(), 0)
        self.assertEqual(clock_region_x0y3.bbox().yl(), 180)
        self.assertEqual(clock_region_x0y3.bbox().xh(), 29)
        self.assertEqual(clock_region_x0y3.bbox().yh(), 239)
        clock_region_x1y3 = clock_region_map.at(1, 3)
        self.assertEqual(clock_region_x1y3.bbox().xl(), 30)
        self.assertEqual(clock_region_x1y3.bbox().yl(), 180)
        self.assertEqual(clock_region_x1y3.bbox().xh(), 65)
        self.assertEqual(clock_region_x1y3.bbox().yh(), 239)
        clock_region_x2y0 = clock_region_map.at(2, 0)
        self.assertEqual(clock_region_x2y0.bbox().xl(), 66)
        self.assertEqual(clock_region_x2y0.bbox().yl(), 0)
        self.assertEqual(clock_region_x2y0.bbox().xh(), 102)
        self.assertEqual(clock_region_x2y0.bbox().yh(), 59)
        clock_region_x2y7 = clock_region_map(2, 7)
        self.assertEqual(clock_region_x2y7.bbox().xl(), 66)
        self.assertEqual(clock_region_x2y7.bbox().yl(), 420)
        self.assertEqual(clock_region_x2y7.bbox().xh(), 102)
        self.assertEqual(clock_region_x2y7.bbox().yh(), 479)
        clock_region_x3y0 = clock_region_map.at(3, 0)
        self.assertEqual(clock_region_x3y0.bbox().xl(), 103)
        self.assertEqual(clock_region_x3y0.bbox().yl(), 0)
        self.assertEqual(clock_region_x3y0.bbox().xh(), 139)
        self.assertEqual(clock_region_x3y0.bbox().yh(), 59)
        clock_region_x3y7 = clock_region_map(3, 7)
        self.assertEqual(clock_region_x3y7.bbox().xl(), 103)
        self.assertEqual(clock_region_x3y7.bbox().yl(), 420)
        self.assertEqual(clock_region_x3y7.bbox().xh(), 139)
        self.assertEqual(clock_region_x3y7.bbox().yh(), 479)
        clock_region_x4y0 = clock_region_map.at(4, 0)
        self.assertEqual(clock_region_x4y0.bbox().xl(), 140)
        self.assertEqual(clock_region_x4y0.bbox().yl(), 0)
        self.assertEqual(clock_region_x4y0.bbox().xh(), 167)
        self.assertEqual(clock_region_x4y0.bbox().yh(), 59)
        clock_region_x4y7 = clock_region_map(4, 7)
        self.assertEqual(clock_region_x4y7.bbox().xl(), 140)
        self.assertEqual(clock_region_x4y7.bbox().yl(), 420)
        self.assertEqual(clock_region_x4y7.bbox().xh(), 167)
        self.assertEqual(clock_region_x4y7.bbox().yh(), 479)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        pass
    elif len(sys.argv) < 3:
        sys.argv.pop()
    else:
        sys.argv.pop()
        sys.argv.pop()
    unittest.main()
