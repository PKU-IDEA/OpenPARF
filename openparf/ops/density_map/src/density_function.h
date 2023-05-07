/**
 * @file   density_function.h
 * @author Yibo Lin
 * @date   Apr 2020
 * @brief  Compute different density models
 */

#ifndef OPENPARF_DENSITY_MAP_DENSITY_FUNCTION_H
#define OPENPARF_DENSITY_MAP_DENSITY_FUNCTION_H

#include <algorithm>

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

/// @brief Allow negative
template<typename T>
inline T OPENPARF_HOST_DEVICE_FUNCTION triangleDensity(T node_xl, T node_xh, T bin_xl, T bin_xh) {
  return OPENPARF_STD_NAMESPACE::min(node_xh, bin_xh) - OPENPARF_STD_NAMESPACE::max(node_xl, bin_xl);
}

/// @brief Non-negative
template<typename T>
inline T OPENPARF_HOST_DEVICE_FUNCTION exactDensity(T node_xl, T node_xh, T bin_xl, T bin_xh) {
  return OPENPARF_STD_NAMESPACE::max((T) 0.0, triangleDensity(node_xl, node_xh, bin_xl, bin_xh));
}

OPENPARF_END_NAMESPACE

#endif
