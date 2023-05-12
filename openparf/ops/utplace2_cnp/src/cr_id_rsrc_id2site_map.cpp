#include "cr_id_rsrc_id2site_map.h"

namespace clock_network_planner {

/**
 * @brief Get the cost to a clock region for the packed group.
 * For packed group, we leverage the weighted mahan distance between the group gravity center and
 * the bounding box of clock region.
 * @param group Assignment group
 * @param cr Clock region
 * @param db Placement database.
 * @return The cost to a clock region for the packed group.
 */
RealType getPackedGroupToCrDist(AssignmentGroup group, const ClockRegion &cr, const PlaceDB &db,
                                   CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map) {
    // Get clock region to group weighted distance
    auto     geometry_bbox = cr.geometry_bbox();
    auto     distXY = geometry_bbox.manhDist(Point<RealType, 2>(group.x_grav(), group.y_grav()));
    RealType x_wirelength_weight_ = db.wirelengthWeights().first;
    RealType y_wirelength_weight_ = db.wirelengthWeights().second;
    RealType dist =
            1.0 * x_wirelength_weight_ * distXY.x() + 1.0 * y_wirelength_weight_ * distXY.y();
    return 1.0 * group.pin_nums() / group.inst_num() * dist;
}


bool groupSiteCompatibility(AssignmentGroup group, const Site &site, const PlaceDB &db) {
    for (const auto &inst_id : group.inst_ids()) {
        if (!db.instToSiteRrscTypeCompatibility(inst_id, site)) { return false; }
    }
    return true;
}

RealType getSingleEleGroupToCrDist(AssignmentGroup group, const ClockRegion &cr,
                                      const PlaceDB &    db,
                                      CrIdRsrcId2SiteMap cr_id_rsrc_id2site_map) {
    RealType x_wirelength_weight_ = db.wirelengthWeights().first;
    RealType y_wirelength_weight_ = db.wirelengthWeights().second;
    openparfAssertMsg(group.inst_num() == 1, "A single-element group should have exactly instance");
    IndexType   inst_id      = group.inst_ids()[0];
    const auto &resource_ids = db.instResourceType(inst_id);
    openparfAssertMsg(resource_ids.size() == 1, "Here we assume each SSIR instance has only one "
                                                "resource type, but we can improve it 🤔");
    IndexType   resource_id     = resource_ids[0];
    const auto &candidate_sites = cr_id_rsrc_id2site_map.at(cr.id(), resource_id);
    if (candidate_sites.empty()) { return std::numeric_limits<RealType>::max(); }
    RealType dist = std::numeric_limits<RealType>::max();
    auto     grav = Point<RealType, 2>(group.x_grav(), group.y_grav());
    for (const auto &site : candidate_sites) {
        RealType x_center_site = site.get().siteMapId().x() + 0.5;
        RealType y_center_size = site.get().siteMapId().y() + 0.5;
        RealType x_dist        = std::fabs(x_center_site - group.x_grav());
        RealType y_dist        = std::fabs(y_center_size - group.y_grav());
        RealType cur = 1.0 * x_wirelength_weight_ * x_dist + 1.0 * y_wirelength_weight_ * y_dist;
        dist         = std::min(dist, cur);
    }
    return 1.0 * group.pin_nums() / group.inst_num() * dist;
}
}   // namespace clock_network_planner