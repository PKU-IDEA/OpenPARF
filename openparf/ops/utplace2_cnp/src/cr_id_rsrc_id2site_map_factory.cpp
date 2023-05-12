#include "cr_id_rsrc_id2site_map_factory.h"

namespace clock_network_planner {

CrIdRsrcId2SiteMap CrIdRsrcId2SiteMapFactory::genCrIdRsrcId2SiteMap(const PlaceDB &db) {
    IndexType   crs_num       = db.numCr();
    IndexType   x_crs_size    = db.clockRegionMapSize().first;
    IndexType   y_crs_size    = db.clockRegionMapSize().second;
    IndexType   resources_num = db.numResources();
    auto        rv            = detail::makeCrIdRsrcId2SiteMap(crs_num, resources_num);
    const auto &layout        = db.db()->layout();
    for (const auto &site : layout.siteMap()) {
        // only loop valid site
        IndexType cr_id = site.clockRegionId();
        for (IndexType resource_id = 0; resource_id < resources_num; resource_id++) {
            if (db.getSiteCapacity(site, resource_id) > 0) {
                rv.at(cr_id, resource_id).emplace_back(site);
            }
        }
    }
    for (int32_t x_cr_id = 0; x_cr_id < x_crs_size; x_cr_id++) {
        for (int32_t y_cr_id = 0; y_cr_id < y_crs_size; y_cr_id++) {
            for (IndexType resource_id = 0; resource_id < resources_num; resource_id++) {
                int32_t cr_id = y_cr_id + x_cr_id * y_crs_size;
                auto    t     = rv.at(cr_id, resource_id).size();
                openparfPrint(kDebug, "# of resource %d in CR(%d,%d): %d\n", resource_id, x_cr_id,
                              y_cr_id, t);
            }
        }
    }
    return rv;
}
}   // namespace clock_network_planner