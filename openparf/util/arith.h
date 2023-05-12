#ifndef OPENPARF_ARITH_H
#define OPENPARF_ARITH_H

#include "util/namespace.h"
#include <cmath>

OPENPARF_BEGIN_NAMESPACE

namespace arithmetic {

    // ============================================================================
    // [Global Constants]
    // ============================================================================

    static constexpr double OPENPARF_M_PI = 3.14159265358979323846;      //!< pi.
    static constexpr double OPENPARF_M_1p5_PI = 4.71238898038468985769;  //!< pi * 1.5.
    static constexpr double OPENPARF_M_2_PI = 6.28318530717958647692;    //!< pi * 2.
    static constexpr double OPENPARF_M_PI_DIV_2 = 1.57079632679489661923;//!< pi / 2.
    static constexpr double OPENPARF_M_PI_DIV_3 = 1.04719755119659774615;//!< pi / 3.
    static constexpr double OPENPARF_M_PI_DIV_4 = 0.78539816339744830962;//!< pi / 4.
    static constexpr double OPENPARF_M_SQRT_0p5 = 0.70710678118654746172;//!< sqrt(0.5).
    static constexpr double OPENPARF_M_SQRT_2 = 1.41421356237309504880;  //!< sqrt(2).
    static constexpr double OPENPARF_M_SQRT_3 = 1.73205080756887729353;  //!< sqrt(3).

    /**
     * @brief Compute the definite integral of Gaussian distribution function.
     * Reference: https://en.wikipedia.org/wiki/Normal_distribution
     *
     * @param [in] mu expected value of Gaussian distribution function
     * @param [in] sigma standard deviation of Gaussian distribution
     * @param [in] lo lower bound of definite interval range
     * @param [in] hi upper bound of definite interval range
     *
     * @return the definite integral of Gaussian distribution function with
     * parameters (mu, sigma) within the interval [lo, hi).
     *
     */
    template<typename T>
    inline T ComputeGaussianAUC(T mu, T sigma, T lo, T hi) {
        T temp = 1.0 / (OPENPARF_M_SQRT_2 * sigma);
        return 0.5 * (OPENPARF_STD_NAMESPACE::erfc((mu - hi) * temp) - OPENPARF_STD_NAMESPACE::erfc((mu - lo) * temp));
    }

};// namespace arithmetic
// namespace arithmetic

OPENPARF_END_NAMESPACE

#endif//OPENPARF_ARITH_H
