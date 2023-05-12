/**
 * @file   data_traits.hpp
 * @author Yibo Lin
 * @date   Apr 2020
 */

#ifndef OPENPARF_UTIL_DATA_TRAITS_HPP_
#define OPENPARF_UTIL_DATA_TRAITS_HPP_

#include "util/namespace.h"
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <typeinfo>

OPENPARF_BEGIN_NAMESPACE

/// data traits
/// define a template class of data traits
/// which will make it easier for generic change of data type
template<typename T>
struct CoordinateTraits;

/// specialization for int32_t
template<>
struct CoordinateTraits<int32_t> {
    using CoordinateType = int32_t;
    using EuclidieanDistanceType = float;
    using ManhattanDistanceType = int64_t;
    using AreaType = int64_t;
    using IndexType = uint32_t;///< index (id)
    using WeightType = float;  ///< type for net or node weights
};
/// specialization for uint32_t
template<>
struct CoordinateTraits<uint32_t> {
    using CoordinateType = uint32_t;
    using EuclidieanDistanceType = float;
    using ManhattanDistanceType = uint64_t;
    using AreaType = uint64_t;
    using IndexType = uint32_t;///< index (id)
    using WeightType = float;  ///< type for net or node weights
};
/// specialization for float
template<>
struct CoordinateTraits<float> {
    using CoordinateType = float;
    using EuclidieanDistanceType = double;
    using ManhattanDistanceType = double;
    using AreaType = double;
    using IndexType = uint32_t;///< index (id)
    using WeightType = float;  ///< type for net or node weights
};
/// specialization for double
template<>
struct CoordinateTraits<double> {
    using CoordinateType         = double;
    using EuclidieanDistanceType = long double;
    using ManhattanDistanceType  = long double;
    using AreaType               = long double;
    using IndexType              = uint64_t;   ///< index (id)
    using WeightType             = double;     ///< type for net or node weights
};


template<typename T, bool is_signed = std::is_signed<T>::value,
         typename = typename std::enable_if<std::is_integral<T>::value>::type>
struct InvalidIndex;

template<typename T>
struct InvalidIndex<T, true> {
    static_assert(std::numeric_limits<T>::lowest() <= -1 && -1 <= std::numeric_limits<T>::max(),
                  "-1 is not in the valid representation range of IndexType");
    static constexpr T value = -1;
};

template<typename T>
struct InvalidIndex<T, false> {
    static constexpr T value = std::numeric_limits<T>::max();
};


OPENPARF_END_NAMESPACE

#endif
