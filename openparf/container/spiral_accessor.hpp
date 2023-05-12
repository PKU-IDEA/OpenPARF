/**
 * File              : spiral_accessor.h
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 06.03.2020
 * Last Modified Date: 06.03.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#pragma once

// C++ standard library headers
#include <vector>

// project headers
#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace container {
/// Class to spirally accesses (x, y) coordinates sequence
/// Coordinates will be enumerated in the following order
//                       16
///                   17  7 15
///                18  8  2  6 14
///             19  9  3  0  1  5 13
///                20 10  4 12 24
///                   21 11 23
///                      22
class SpiralAccessor {
  public:
  using IntType = int32_t;
  template<typename T>
  using XY = geometry::Point<T, 2>;

  explicit SpiralAccessor(IntType maxR) : _maxR(maxR) {
    // Calculate the maximum number of points (XYs) in maxR
    // There are 4 * k points on the ring of radius == k (k > 0).
    // So the max number of points is
    //     4 * (1 + maxR) * maxR / 2 + 1
    //     = 2 * (1 + maxR) * maxR + 1
    std::size_t maxNumPoints = 2 * (1 + maxR) * maxR + 1;
    _seq.reserve(maxNumPoints);

    // Initalize the coordinate sequence
    _seq.emplace_back(0, 0);
    for (IntType r = 1; r <= maxR; ++r) {
      // The 1st quadrant
      for (IntType x = r, y = 0; y < r; --x, ++y) {
        _seq.emplace_back(x, y);
      }
      // The 2nd quadrant
      for (IntType x = 0, y = r; y > 0; --x, --y) {
        _seq.emplace_back(x, y);
      }
      // The 3rd quadrant
      for (IntType x = -r, y = 0; y > -r; ++x, --y) {
        _seq.emplace_back(x, y);
      }
      // The 4th quadrant
      for (IntType x = 0, y = -r; y < 0; ++x, ++y) {
        _seq.emplace_back(x, y);
      }
    }
  }

  class iterator {
private:
    using InternalIterator = std::vector<XY<IntType>>::const_iterator;

public:
    explicit iterator(InternalIterator it) : _it(it) {}

    const XY<IntType>& operator*() const { return *_it; }
    InternalIterator   operator->() const { return _it; }
    iterator&          operator++() {
      ++_it;
      return *this;
    }
    bool operator==(iterator rhs) const { return _it == rhs._it; }
    bool operator!=(iterator rhs) const { return !(*this == rhs); }

private:
    InternalIterator _it;
  };

  /// Get the begin(end) iterator points to the first(one after the last) XY with Manhattan distance to the origin of
  /// 'r' Note that the number of points that has their distance less than 'r' should be
  ///   1) If r == 0:  0
  ///   2) If r != 0:  4 * r * (r - 1) / 2 + 1 == 2 * r * (r - 1) + 1
  iterator begin(IntType r) const { return iterator(_seq.begin() + (r ? 2 * r * (r - 1) + 1 : 0)); }
  iterator end(IntType r) const { return iterator(_seq.begin() + (r ? 2 * (r + 1) * r + 1 : 1)); }
  iterator begin() const { return begin(0); }
  iterator end() const { return end(_maxR); }
  IntType  maxRadius() const { return _maxR; }
  IntType  radius(iterator it) const { return std::abs(it->x()) + std::abs(it->y()); }

  private:
  std::vector<XY<IntType>> _seq;        // The sequence of the coordinate
  IntType                  _maxR = 0;   // The maximum Manhattan radius
};

}   // namespace container
OPENPARF_END_NAMESPACE
