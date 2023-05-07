/**
 * FIXME(jingmai@pku.edu.cn): do not include `placedb.h` in this file, coz it violates our principles.
 *  Use the site id instead of the class `Site`.
 */
#pragma once

// C++ standard library headers
#include <memory>

// project headers
#include "container/vector_2d.hpp"
#include "database/placedb.h"

// local headers
#include "assignment_group.h"
#include "clock_network_planner.h"

namespace clock_network_planner {
using namespace openparf;
using openparf::container::Vector2D;
using openparf::database::ClockRegion;
using openparf::database::PlaceDB;
using openparf::database::Site;
using openparf::geometry::Point;

using IndexType   = ClockNetworkPlannerImpl::IndexType;
using RealType    = AssignmentGroupImpl::RealType;
using FlowIntType = ClockNetworkPlannerImpl::FlowIntType;

class CrIdRsrcId2SiteMap {
  public:
  using InstanceType = Vector2D<std::vector<std::reference_wrapper<const Site>>>;

  private:
  std::shared_ptr<InstanceType> impl_;

  public:
  explicit CrIdRsrcId2SiteMap(std::shared_ptr<InstanceType> impl) : impl_(std::move(impl)) {}
  FORWARDED_METHOD(at)
};

CUSTOM_MAKE_SHARED_CLASS(CrIdRsrcId2SiteMap, CrIdRsrcId2SiteMap::InstanceType)

using GroupToRegionDistFunctor =
        std::function<RealType(AssignmentGroup, const ClockRegion &, const PlaceDB &db, CrIdRsrcId2SiteMap)>;

bool     groupSiteCompatibility(AssignmentGroup group, const Site &site, const PlaceDB &db);

/**
 * @brief Get the cost to a clock region for the packed group.
 * For packed group, we leverage the weighted mahan distance between the group gravity center and
 * the bounding box of clock region.
 * @param group Assignment group
 * @param cr Clock region
 * @param db Placement database.
 * @return The cost to a clock region for the packed group.
 */
RealType getPackedGroupToCrDist(AssignmentGroup    group,
                                const ClockRegion &cr,
                                const PlaceDB &    db,
                                CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map);


RealType getSingleEleGroupToCrDist(AssignmentGroup    group,
                                   const ClockRegion &cr,
                                   const PlaceDB &    db,
                                   CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map);

}   // namespace clock_network_planner