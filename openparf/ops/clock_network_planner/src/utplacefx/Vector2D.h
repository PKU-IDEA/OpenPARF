#ifndef __VECTOR2D_H__
#define __VECTOR2D_H__

#include <vector>
#include <initializer_list>

#include "util/namespace.h"
#include "ops/clock_network_planner/src/utplacefx/Types.h"


OPENPARF_BEGIN_NAMESPACE
namespace utplacefx {
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
template <typename T>
class Vector2D
{
public:
    /// Class for initializer_list type
    enum class InitListType : Byte
    {
        XMajor,
        YMajor
    };

public:
    explicit Vector2D() = default;
    explicit Vector2D(IndexType xs, IndexType ys) : _vec(xs * ys), _xSize(xs), _ySize(ys) {}
    explicit Vector2D(IndexType xs, IndexType ys, const T &v) : _vec(xs * ys, v), _xSize(xs), _ySize(ys) {}
    explicit Vector2D(IndexType xs, IndexType ys, InitListType t, std::initializer_list<T> l);

    void                                       clear()                                         { _vec.clear(); _xSize = _ySize = 0; }
    void                                       resize(IndexType xs, IndexType ys)              { _vec.resize(xs * ys); _xSize = xs; _ySize = ys; }
    void                                       resize(IndexType xs, IndexType ys, const T &v)  { _vec.resize(xs * ys, v); _xSize = xs; _ySize = ys; }

    IndexType                                  xSize() const                                   { return _xSize; }
    IndexType                                  ySize() const                                   { return _ySize; }
    IndexType                                  size() const                                    { return _vec.size(); }
    T &                                        at(IndexType x, IndexType y)                    { return _vec.at(xyToIndex(x, y)); }
    const T &                                  at(IndexType x, IndexType y) const              { return _vec.at(xyToIndex(x, y)); }
    T &                                        at(IndexType i)                                 { return _vec.at(i); }
    const T &                                  at(IndexType i) const                           { return _vec.at(i); }
    T &                                        at(const XY<IndexType> &xy)                     { return _vec.at(xyToIndex(xy.x(), xy.y())); }
    const T &                                  at(const XY<IndexType> &xy) const               { return _vec.at(xyToIndex(xy.x(), xy.y())); }
    T &                                        at(const XY<IntType> &xy)                       { return _vec.at(xyToIndex(xy.x(), xy.y())); }
    const T &                                  at(const XY<IntType> &xy) const                 { return _vec.at(xyToIndex(xy.x(), xy.y())); }
    typename std::vector<T>::iterator          begin()                                         { return _vec.begin(); }
    typename std::vector<T>::const_iterator    begin() const                                   { return _vec.begin(); }
    typename std::vector<T>::iterator          end()                                           { return _vec.end(); }
    typename std::vector<T>::const_iterator    end() const                                     { return _vec.end(); }

    // Conversion between XY and index
    IndexType                                  indexToX(IndexType i) const                     { return i / _ySize; }
    IndexType                                  indexToY(IndexType i) const                     { return i % _ySize; }
    XY<IndexType>                              indexToXY(IndexType i) const                    { return XY<IndexType>(indexToX(i), indexToY(i)); }
    IndexType                                  xyToIndex(IndexType x, IndexType y) const       { return _ySize * x + y; }

    bool operator==(const Vector2D<T> &rhs) const { return _vec == rhs._vec && _xSize == rhs._xSize && _ySize == rhs._ySize; }
    bool operator!=(const Vector2D<T> &rhs) const { return ! (*this == rhs); }

private:
    std::vector<T>  _vec;
    IndexType       _xSize = 0;
    IndexType       _ySize = 0;
};

/// Ctor using initializer_list
template <typename T>
inline Vector2D<T>::Vector2D(IndexType xs, IndexType ys, InitListType t, std::initializer_list<T> l)
    : _vec(xs * ys), _xSize(xs), _ySize(ys)
{
    openparfAssert(l.size() == size());
    if (t == InitListType::XMajor)
    {
        // Data in Vector2D is stored in X-major order
        // So we just need to copy the initializer_list to the _vec
        std::copy(l.begin(), l.end(), _vec.begin());
    }
    else // t == InitListType::YMajor
    {
        for (IndexType idx = 0; idx < size(); ++idx)
        {
            IndexType x = idx % _xSize;
            IndexType y = idx / _xSize;
            at(x, y) = *(l.begin() + idx);
        }
    }
}
}
OPENPARF_END_NAMESPACE

#endif // __VECTOR2D_H__
