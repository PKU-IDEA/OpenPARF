/**
 * @file   database.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "database/database.h"
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern std::string test_dir;

OPENPARF_BEGIN_NAMESPACE

namespace unitest {

/// GTest class for fft module testing
class DatabaseTest : public ::testing::Test {
public:
  void testSample1() {
    openparf::database::Database db(0);
    db.readBookshelf(test_dir + "/" + "../../benchmarks/sample1/design.aux");

    std::cout << db.design() << std::endl;
    // std::cout << db << std::endl;

    auto const &design = db.design();
    auto const &layout = db.layout();
    auto const &site_map = layout.siteMap();
    for (auto const &site : site_map) {
      ASSERT_EQ(
          &site,
          &site_map.at(site.siteMapId().x(), site.siteMapId().y()).value());
      if (layout.siteType(site).name() == "SLICE" ||
          layout.siteType(site).name() == "SLICEL" ||
          layout.siteType(site).name() == "SLICEM") {
        ASSERT_EQ(site.bbox().width(), 1);
        ASSERT_EQ(site.bbox().height(), 1);
      } else if (layout.siteType(site).name() == "DSP") {
        ASSERT_EQ(site.bbox().width(), 1);
        if ((site.siteMapId().y() % 5) == 0) {
          ASSERT_EQ(site.bbox().height(), 2);
        } else {
          ASSERT_EQ(site.bbox().height(), 3);
        }
      } else if (layout.siteType(site).name() == "BRAM") {
        ASSERT_EQ(site.bbox().width(), 1);
        ASSERT_EQ(site.bbox().height(), 5);
      } else if (layout.siteType(site).name() == "IO") {
        ASSERT_EQ(site.bbox().width(), 1);
        auto xs = {0, 67, 104, 167};
        if (std::any_of(xs.begin(), xs.end(),
                        [&](int x) { return x == site.siteMapId().x(); })) {
          ASSERT_EQ(site.bbox().height(), 60);
        } else {
          ASSERT_EQ(site.bbox().height(), 30);
        }
      } else {
        ASSERT_EQ(layout.siteType(site).name(), "SLICE");
      }
    }

    auto const &site_type_map = layout.siteTypeMap();
    auto const &resource_map = layout.resourceMap();
    ASSERT_EQ(site_type_map.siteType("SLICE")->resourceCapacity(
                  resource_map.resourceId("LUT")),
              16);
    ASSERT_EQ(site_type_map.siteType("SLICE")->resourceCapacity(
                  resource_map.resourceId("FF")),
              16);
    ASSERT_EQ(site_type_map.siteType("SLICE")->resourceCapacity(
                  resource_map.resourceId("CARRY8")),
              1);
    ASSERT_EQ(site_type_map.siteType("DSP")->resourceCapacity(
                  resource_map.resourceId("DSP48E2")),
              1);
    ASSERT_EQ(site_type_map.siteType("BRAM")->resourceCapacity(
                  resource_map.resourceId("RAMB36E2")),
              1);
    ASSERT_EQ(site_type_map.siteType("IO")->resourceCapacity(
                  resource_map.resourceId("IO")),
              64);

    ASSERT_EQ(resource_map.resourceId("LUT"),
              resource_map.modelResourceIds(design.modelId("LUT1"))[0]);
    ASSERT_EQ(resource_map.resourceId("LUT"),
              resource_map.modelResourceIds(design.modelId("LUT2"))[0]);
    ASSERT_EQ(resource_map.resourceId("LUT"),
              resource_map.modelResourceIds(design.modelId("LUT3"))[0]);
    ASSERT_EQ(resource_map.resourceId("LUT"),
              resource_map.modelResourceIds(design.modelId("LUT4"))[0]);
    ASSERT_EQ(resource_map.resourceId("LUT"),
              resource_map.modelResourceIds(design.modelId("LUT5"))[0]);
    ASSERT_EQ(resource_map.resourceId("LUT"),
              resource_map.modelResourceIds(design.modelId("LUT6"))[0]);
    ASSERT_EQ(resource_map.resourceId("FF"),
              resource_map.modelResourceIds(design.modelId("FDRE"))[0]);
    ASSERT_EQ(resource_map.resourceId("CARRY8"),
              resource_map.modelResourceIds(design.modelId("CARRY8"))[0]);
    ASSERT_EQ(resource_map.resourceId("DSP48E2"),
              resource_map.modelResourceIds(design.modelId("DSP48E2"))[0]);
    ASSERT_EQ(resource_map.resourceId("RAMB36E2"),
              resource_map.modelResourceIds(design.modelId("RAMB36E2"))[0]);
    ASSERT_EQ(resource_map.resourceId("IO"),
              resource_map.modelResourceIds(design.modelId("IBUF"))[0]);
    ASSERT_EQ(resource_map.resourceId("IO"),
              resource_map.modelResourceIds(design.modelId("OBUF"))[0]);
    ASSERT_EQ(resource_map.resourceId("IO"),
              resource_map.modelResourceIds(design.modelId("BUFGCE"))[0]);
  }
};

TEST_F(DatabaseTest, Sample1) { testSample1(); }

} // namespace unitest

OPENPARF_END_NAMESPACE
