#pragma once
#include "assignment_group.h"
#include "clock_network_planner.h"
#include "database/placedb.h"

namespace clock_network_planner {

using openparf::database::PlaceDB;

struct AssignmentGroupFactory {
    using IndexType   = ClockNetworkPlannerImpl::IndexType;
    using RealType    = AssignmentGroupImpl::RealType;
    using FlowIntType = ClockNetworkPlannerImpl::FlowIntType;

    template<typename scalar_t>
    static GroupArrayWrapper
    genGroupArrayWrapperFromPlaceDBTemplate(ClockNetworkPlanner cnp, const PlaceDB &db,
                                            scalar_t *pos, scalar_t *inst_areas) {
        IndexType insts_num = cnp.template insts_num();
        cnp.template inst_areas().resize(insts_num);
        cnp.template inst_x_pos().resize(insts_num);
        cnp.template inst_y_pos().resize(insts_num);
        for (int i = 0; i < insts_num; i++) {
            cnp.template inst_areas()[i] = static_cast<RealType>(inst_areas[i]);
            cnp.template inst_x_pos()[i] = static_cast<RealType>(pos[i << 1]);
            cnp.template inst_y_pos()[i] = static_cast<RealType>(pos[i << 1 | 1]);
        }
        return genGroupArrayWrapperFromPlaceDB(cnp, db);
    }
    static GroupArrayWrapper genGroupArrayWrapperFromPlaceDB(ClockNetworkPlanner cnp,
                                                             const PlaceDB &     db);
};
}   // namespace clock_network_planner
