/**
 * File              : vector_2d.hpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.20.2020
 * Last Modified Date: 05.20.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "geometry/geometry.hpp"

OPENPARF_BEGIN_NAMESPACE

namespace container {

template<typename T>
using XY = geometry::Point<T, 2>;

template<typename T>
using Box = geometry::Box<T>;

/// Class for 2D vectors
/// This is a Y-major implementation, that is elements in the
/// same column (with the same X) are in a piece of consecutive memory.
/// Thus, when iterate through this 2D vector using a two-level for loop,
/// the following pattern is better from the memory/cache hit prespective
/// for (x ...)
/// {
///     for (y ....)
///     {
///         ....
///     }
/// }
template<typename T>
class Vector2D {
  public:
  using Byte      = std::int8_t;
  using IndexType = std::int32_t;
  enum class InitListType
  {
    XMajor,
    YMajor
  };

  explicit Vector2D() : _sizes(0, 0) {}
  explicit Vector2D(IndexType xs, IndexType ys) : _vec(xs * ys), _sizes(xs, ys) {}
  explicit Vector2D(IndexType xs, IndexType ys, const T &v) : _vec(xs * ys, v), _sizes(xs, ys) {}
  explicit Vector2D(IndexType xs, IndexType ys, InitListType t, std::initializer_list<T> l);
  explicit Vector2D(const XY<IndexType> &sizes) : _vec(sizes.x() * sizes.y()), _sizes(sizes) {}

  void clear() {
    _vec.clear();
    _sizes.set(0, 0);
  }
  void resize(IndexType xs, IndexType ys) {
    _vec.resize(xs * ys);
    _sizes.set(xs, ys);
  }
  void resize(IndexType xs, IndexType ys, const T &v) {
    _vec.resize(xs * ys, v);
    _sizes.set(xs, ys);
  }
  void resize(const XY<IndexType> &sizes) {
    _vec.resize(sizes.x() * sizes.y());
    _sizes = sizes;
  }
  void resize(const XY<IndexType> &sizes, const T &v) {
    _vec.resize(sizes.x() * sizes.y(), v);
    _sizes = sizes;
  }

  IndexType            xSize() const { return _sizes.x(); }
  IndexType            ySize() const { return _sizes.y(); }
  IndexType            size() const { return _vec.size(); }
  const XY<IndexType> &sizes() const { return _sizes; }
  T &                  at(IndexType x, IndexType y) { return _vec.at(xyToIndex(x, y)); }
  const T &            at(IndexType x, IndexType y) const { return _vec.at(xyToIndex(x, y)); }
  T &                  at(IndexType i) { return _vec.at(i); }
  const T &            at(IndexType i) const { return _vec.at(i); }
  T &                  at(const XY<IndexType> &xy) { return _vec.at(xyToIndex(xy.x(), xy.y())); }
  const T &            at(const XY<IndexType> &xy) const { return _vec.at(xyToIndex(xy.x(), xy.y())); }
  T &                  operator[](IndexType i) { return _vec[i]; }
  const T &            operator[](IndexType i) const { return _vec[i]; }
  T &                  operator[](const XY<IndexType> &xy) { return _vec[xyToIndex(xy.x(), xy.y())]; }
  const T &            operator[](const XY<IndexType> &xy) const { return _vec[xyToIndex(xy.x(), xy.y())]; }
  T &                  operator()(IndexType i) { return _vec[i]; }
  const T &            operator()(IndexType i) const { return _vec[i]; }
  T &                  operator()(IndexType x, IndexType y) { return _vec[xyToIndex(x, y)]; }
  const T &            operator()(IndexType x, IndexType y) const { return _vec[xyToIndex(x, y)]; }
  T &                  operator()(const XY<IndexType> &xy) { return _vec[xyToIndex(xy.x(), xy.y())]; }
  const T &            operator()(const XY<IndexType> &xy) const { return _vec[xyToIndex(xy.x(), xy.y())]; }
  typename std::vector<T>::iterator       begin() { return _vec.begin(); }
  typename std::vector<T>::const_iterator begin() const { return _vec.begin(); }
  typename std::vector<T>::iterator       end() { return _vec.end(); }
  typename std::vector<T>::const_iterator end() const { return _vec.end(); }
  T *                                     data() { return _vec.data(); }
  const T *                               data() const { return _vec.data(); }

  // Conversion between XY and index
  IndexType                               indexToX(IndexType i) const { return i / _sizes.y(); }
  IndexType                               indexToY(IndexType i) const { return i % _sizes.y(); }
  XY<IndexType> indexToXY(IndexType i) const { return XY<IndexType>(indexToX(i), indexToY(i)); }
  IndexType     xyToIndex(IndexType x, IndexType y) const { return _sizes.y() * x + y; }

  bool          operator==(const Vector2D<T> &rhs) const { return _vec == rhs._vec && _sizes == rhs._sizes; }
  bool          operator!=(const Vector2D<T> &rhs) const { return !(*this == rhs); }

  private:
  std::vector<T> _vec;
  XY<IndexType>  _sizes;
};
}   // namespace container
OPENPARF_END_NAMESPACE
