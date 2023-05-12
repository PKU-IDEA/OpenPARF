#pragma once
#include "cr_id_rsrc_id2site_map.h"

namespace clock_network_planner {

struct CrIdRsrcId2SiteMapFactory {
    static CrIdRsrcId2SiteMap genCrIdRsrcId2SiteMap(const PlaceDB &db);
};
}   // namespace clock_network_planner