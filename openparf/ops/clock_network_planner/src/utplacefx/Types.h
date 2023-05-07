/**
 * @file   types.h
 * @brief  Definition for the types used in UTPlaceFX
 * @author Yibai Meng
 * @date   Sep 2020
 */
#ifndef __OPS_CLOCK_NETWORK_PLANNER_TYPES_H__
#define __OPS_CLOCK_NETWORK_PLANNER_TYPES_H__

#include<stdint.h>

#include "util/namespace.h"
#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace utplacefx {

    using IndexType = uint32_t;
    using Byte = uint8_t;
    template<typename T>
    using XY = geometry::Point<T, 2>;
    template<typename T>
    using Box = geometry::Box<T>;
    using FlowIntType = std::int64_t;
    using IntType =  std::int32_t;
    using RealType = double; // TODO:

    constexpr IndexType INDEX_TYPE_MAX = 1000000000;// 1e+9
    constexpr IntType INT_TYPE_MAX = 1000000000;    // 1e+9
    constexpr IntType INT_TYPE_MIN = -1000000000;   // -1e+9
    constexpr RealType REAL_TYPE_MAX = 1e100;
    constexpr RealType REAL_TYPE_MIN = -1e100;
    constexpr RealType REAL_TYPE_TOL = 1e-6;

    enum class SiteType : Byte
    {
        SLICEL,
        SLICEM,
        DSP,
        RAM,
        IO,
        INVALID
    };

    enum class ShapeType : Byte
    {
        DSP_CHAIN,
        RAM_CHAIN,
        CARRY_CHAIN,
        INVALID
    };

    // Enum for resource types
    enum class RsrcType : Byte
    {
        DSP,
        RAM,
        LUTL,
        LUTM,
        FF,
        LUTL_PAIR,
        FF_PAIR,
        BLEL,
        SLICEL,
        SLICEM,
        IO,
        INVALID
    };
    constexpr const char* const RsrcTypeName[] = {
        "DSP",
        "RAM",
        "LUTL",
        "LUTM",
        "FF",
        "LUTL_PAIR",
        "FF_PAIR",
        "BLEL",
        "SLICEL",
        "SLICEM",
        "IO",
        "INVALID"
    };

}// namespace utplacefx
OPENPARF_END_NAMESPACE

#endif
