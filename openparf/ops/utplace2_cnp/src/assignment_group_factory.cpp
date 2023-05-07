#include "assignment_group_factory.h"

namespace clock_network_planner {

GroupArrayWrapper AssignmentGroupFactory::genGroupArrayWrapperFromPlaceDB(ClockNetworkPlanner cnp,
                                                                          const PlaceDB &     db) {
    using IndexVector = AssignmentGroupImpl::IndexVector;

    IndexType inst_num                      = cnp.insts_num();
    auto &    packed_resource_ids           = cnp.packed_resource_ids();
    auto &    single_ele_group_resource_ids = cnp.single_ele_group_resource_ids();

    auto impl = std::make_shared<GroupArrayWrapperImpl>();

    // Whether enable packing LUT && FF in a nearby region
    bool enable_packing = cnp.enable_packing();

    std::vector<std::map<IndexVector, IndexVector>> temp_packed_groups(
            cnp.bin_id_size_map().size());
    AssignmentGroupArray &packed_group_array = impl->packed_group_array();

    for (int32_t inst_id = 0; inst_id < inst_num; inst_id++) {
        RealType                    inst_x       = cnp.inst_x_pos()[inst_id];
        RealType                    inst_y       = cnp.inst_y_pos()[inst_id];
        const std::vector<uint8_t> &resource_ids = db.instResourceType(inst_id);
        openparfAssert(!resource_ids.empty());
        if (resource_ids.size() == 1 &&
            std::find(single_ele_group_resource_ids.begin(), single_ele_group_resource_ids.end(),
                      resource_ids[0]) != single_ele_group_resource_ids.end()) {
            IndexType             resource_id = resource_ids[0];
            AssignmentGroupArray &group_array = impl->single_ele_group_array_map()[resource_id];
            auto                  group_impl  = std::make_shared<AssignmentGroupImpl>();
            auto &                key         = group_impl->ck_sets();
            key.clear();
            if (!db.isInstClockSource(inst_id)) {
                // Ignore the clock source when computing the clock region bounding box.
                for (const PlaceDB::IndexType &e : db.instToClocks()[inst_id]) {
                    key.template emplace_back(e);
                }
            }
            std::sort(key.begin(), key.end());
            key.erase(std::unique(key.begin(), key.end()), key.end());

            auto &inst_ids = group_impl->inst_ids();

            inst_ids.clear();
            inst_ids.template emplace_back(inst_id);

            group_impl->x_grav()   = inst_x;
            group_impl->y_grav()   = inst_y;
            group_impl->area_dem() = 1.0;
            group_impl->pin_nums() = db.instPins().at(inst_id).size();

            group_array.emplace_back(group_impl);
            continue;
        }
        if (db.isInstLUT(inst_id) || db.isInstFF(inst_id)) {
            if (enable_packing) {
                // Packing LUT & FF in a nearby region
                IndexVector key;
                key.clear();
                if (!db.isInstClockSource(inst_id)) {
                    // Ignore the clock source when computing the clock region bounding box.
                    for (const PlaceDB::IndexType &e : db.instToClocks()[inst_id]) {
                        key.template emplace_back(e);
                    }
                }
                std::sort(key.begin(), key.end());
                key.erase(std::unique(key.begin(), key.end()), key.end());
                int32_t bin_id =
                        cnp.bin_id_size_map()
                                .at(static_cast<IndexType>(inst_x), static_cast<IndexType>(inst_y))
                                .id();
                temp_packed_groups.at(bin_id)[key].template emplace_back(inst_id);
            } else {
                auto group = detail::makeAssignmentGroup();
                group.ck_sets().clear();
                if (!db.isInstClockSource(inst_id)) {
                    for (auto &e : db.instToClocks()[inst_id]) { group.ck_sets().emplace_back(e); }
                }
                group.inst_ids() = {inst_id};
                group.x_grav()   = inst_x;
                group.y_grav()   = inst_y;
                group.pin_nums() = db.instPins().at(inst_id).size();
                group.area_dem() = cnp.inst_areas()[inst_id];
                packed_group_array.emplace_back(group);
            }
        }
    }

    if (enable_packing) {
        packed_group_array.clear();
        for (auto &bin_map : temp_packed_groups) {
            for (auto &p : bin_map) {
                const IndexVector &key        = p.first;
                IndexVector &      inst_ids   = p.second;
                auto               group_impl = std::make_shared<AssignmentGroupImpl>();
                group_impl->ck_sets()         = key;
                group_impl->inst_ids()        = inst_ids;

                group_impl->x_grav()   = 0;
                group_impl->y_grav()   = 0;
                group_impl->area_dem() = 0;
                group_impl->pin_nums() = 0;
                for (const IndexType &inst_id : inst_ids) {
                    group_impl->x_grav() += cnp.inst_x_pos()[inst_id];
                    group_impl->y_grav() += cnp.inst_y_pos()[inst_id];
                    group_impl->pin_nums() += db.instPins().at(inst_id).size();
                    group_impl->area_dem() += cnp.inst_areas()[inst_id];
                }
                group_impl->x_grav() /= inst_ids.size();
                group_impl->y_grav() /= inst_ids.size();
                packed_group_array.emplace_back(group_impl);
            }
        }
    }

    // Scale CLB group demand to make sure demand <= capacity
    // Assume a site is CLB iff. it contains the first resource id in the packing resource ids
    // array.
    IndexType site_capacity = 2;
    RealType  total_clb_area_cap =
            db.collectSiteBoxes(/*resource_id=*/packed_resource_ids[0]).size() * site_capacity;
    RealType total_clb_area_dem = 0;
    for (auto &group : packed_group_array) { total_clb_area_dem += group.area_dem(); }
    openparfPrint(kDebug, "original total clb area dem: %lf\n", total_clb_area_dem);
    openparfPrint(kDebug, "original # of LUT&FF site: %lf\n", total_clb_area_cap / site_capacity);
    if (total_clb_area_dem > total_clb_area_cap) {
        RealType scale_factor = total_clb_area_cap / (total_clb_area_dem + 1);
        for (auto &group : packed_group_array) { group.impl()->area_dem() *= scale_factor; }
    }
    return GroupArrayWrapper(impl);
}
}   // namespace clock_network_planner