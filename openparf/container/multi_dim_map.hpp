/**
 * @file   multi_dim_map.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_UTIL_MULTI_DIM_MAP_HPP_
#define OPENPARF_UTIL_MULTI_DIM_MAP_HPP_

#include "util/util.h"
#include <array>
#include <numeric>
#include <vector>

OPENPARF_BEGIN_NAMESPACE

namespace container {

template <typename T, std::size_t Dims, bool SetMultiDimId> class MultiDimMap;

/// @brief Help control to store 1D or MD index in element
template <typename T, std::size_t Dims, bool SetMultiDimId>
struct MultiDimMapTraits;

template <typename T, std::size_t Dims>
struct MultiDimMapTraits<T, Dims, false> {
  using MapType = MultiDimMap<T, Dims, false>;
  using IndexType = std::size_t;

  static void setElementId(MapType &, T &element, IndexType const &id) {
    element.setId(id);
  }
};

template <typename T, std::size_t Dims>
struct MultiDimMapTraits<T, Dims, true> {
  using MapType = MultiDimMap<T, Dims, true>;
  using IndexType = std::size_t;

  static void setElementId(MapType &map, T &element, IndexType const &id) {
    setElementIdKernel(element, map.indexMD(id),
                       std::make_index_sequence<Dims>{});
  }

  /// @brief kernel function for setting the index of an element
  template <IndexType... Args>
  static void setElementIdKernel(T &element,
                                 typename MapType::DimType const &md,
                                 std::index_sequence<Args...>) {
    element.setId(md[Args]...);
  }
};

/// @brief A general implementation of multi-dimensional map,
/// which stores the index within each element.
/// The element type must provide a setId function.
template <typename T, std::size_t Dims, bool SetMultiDimId> class MultiDimMap {
public:
  using ValueType = T;
  using IndexType = std::size_t;
  using DimType = std::array<IndexType, Dims>;
  using ConstIteratorType = typename std::vector<ValueType>::const_iterator;
  using IteratorType = typename std::vector<ValueType>::iterator;
  using TraitsType = MultiDimMapTraits<ValueType, Dims, SetMultiDimId>;

  static_assert(Dims > 1, "MultiDimMap only supports Dims > 1");

  /// @brief default constructor
  MultiDimMap() { dims_.fill(0); }

  /// @brief constructor
  template <class... Args> MultiDimMap(IndexType first, Args &&... args) {
    reshape(first, (IndexType)args...);
  }

  /// @brief copy constructor
  MultiDimMap(MultiDimMap const &rhs) { copy(rhs); }

  /// @brief move constructor
  MultiDimMap(MultiDimMap &&rhs) { move(std::move(rhs)); }

  /// @brief copy constructor
  MultiDimMap &operator=(MultiDimMap const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move constructor
  MultiDimMap &operator=(MultiDimMap &&rhs) {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief getter for an element
  template <class... Args> T const &operator()(Args &&... args) const {
    IndexType id = this->index1D((IndexType)args...);
    return this->elements_[id];
  }

  /// @brief getter for an element
  template <class... Args> T &operator()(Args &&... args) {
    IndexType id = this->index1D((IndexType)args...);
    return this->elements_[id];
  }

  /// @brief getter for an element
  template <class... Args> T const &at(IndexType first, Args &&... args) const {
    IndexType id = this->index1D(first, (IndexType)args...);
    return this->elements_.at(id);
  }

  /// @brief getter for an element
  template <class... Args> T &at(IndexType first, Args &&... args) {
    IndexType id = this->index1D(first, (IndexType)args...);
    return this->elements_.at(id);
  }

  /// @brief getter for an element given a dimension
  T const &at(DimType const &md) const {
    IndexType id = this->index1D(md);
    return this->elements_.at(id);
  }

  /// @brief getter for an element given a dimension
  T &at(DimType const &md) {
    IndexType id = this->index1D(md);
    return this->elements_.at(id);
  }

  /// @brief getter for an element with 1D index
  T const &at(IndexType id) const { return this->elements_.at(id); }

  /// @brief getter for an element with 1D index
  T &at(IndexType id) { return this->elements_.at(id); }

  /// @brief convert multi-dimension index to 1D index
  template <class... Args> IndexType index1D(Args &&... args) const {
    static_assert(sizeof...(Args) == Dims,
                  "#number of variadic arguments mismatch");
    return this->recurseIndex((IndexType)args...);
  }

  /// @brief convert multi-dimension index to 1D index
  IndexType index1D(DimType const &md) const {
    IndexType id = md[0];
    for (IndexType i = 1; i < Dims; ++i) {
      id = id * dims_[i] + md[i];
    }
    return id;
  }

  /// @brief convert 1D index to multi-dimension index
  DimType indexMD(IndexType id) const {
    DimType md;
    for (IndexType i = 0; i < Dims - 1; ++i) {
      auto divisor = std::accumulate(dims_.begin() + i + 1, dims_.end(), 1,
                                     std::multiplies<IndexType>());
      md[i] = id / divisor;
      id = id % divisor;
    }
    md[Dims - 1] = id;
    return md;
  }

  /// @brief reshape the map
  template <class... Args> void reshape(Args &&... args) {
    static_assert(sizeof...(Args) == Dims,
                  "#number of variadic arguments mismatch");
    dims_ = {(IndexType)args...};
    IndexType numel = recurseMultiply((IndexType)args...);
    elements_.resize(numel);
    for (IndexType i = 0; i < numel; ++i) {
      TraitsType::setElementId(*this, elements_[i], i);
    }
  }

  /// @brief shape of the map
  DimType shape() const { return dims_; }

  /// @brief shape of the map
  IndexType shape(IndexType dim) const { return dims_[dim]; }

  /// @brief flat size of the map
  IndexType size() const {
    return std::accumulate(dims_.begin(), dims_.end(), 1,
                           std::multiplies<IndexType>());
  }

  /// @brief capacity of the map
  IndexType capacity() const { return elements_.capacity(); }

  ConstIteratorType begin() const { return elements_.begin(); }

  IteratorType begin() { return elements_.begin(); }

  ConstIteratorType end() const { return elements_.end(); }

  IteratorType end() { return elements_.end(); }

protected:
  /// @brief copy the object
  void copy(MultiDimMap const &rhs) {
    dims_ = rhs.dims_;
    elements_ = rhs.elements_;
  }

  /// @brief move the object
  void move(MultiDimMap &&rhs) {
    dims_ = std::move(rhs.dims_);
    elements_ = std::move(rhs.elements_);
  }

  /// @brief kernel function for recursing the index computation
  template <class V>
  inline void recurseIndexKernel(IndexType &partial, V &&first) const {
    partial += first;
  }

  /// @brief kernel function for recursing the index computation
  template <class V, class... Args>
  inline void recurseIndexKernel(IndexType &partial, V &&first,
                                 Args &&... rest) const {
    auto dim = Dims - sizeof...(Args);
    partial += first;
    partial *= dims_[dim];
    recurseIndexKernel(partial, rest...);
  }

  /// @brief kernel function for recursing the index computation
  template <class... Args> auto recurseIndex(Args &&... args) const {
    IndexType res = 0;
    recurseIndexKernel(res, args...);
    return res;
  }

  /// @brief kernel function for recursing the multiplication
  template <class V> inline auto recurseMultiply(V &&first) const {
    return first;
  }

  /// @brief kernel function for recursing the multiplication
  template <class V, class... Args>
  inline auto recurseMultiply(V &&first, Args &&... rest) const {
    return first * recurseMultiply(rest...);
  }

  /// @brief a helper function for recursive print of multi-dimensional array
  friend void print(std::ostream &os, MultiDimMap const &rhs,
                    std::array<IndexType, Dims> md, IndexType dim) {
    if (dim + 1 == Dims) {
      os << "[";
      for (IndexType i = 0; i < rhs.shape(dim); ++i) {
        md[dim] = i;
        os << " " << rhs.at(md);
      }
      os << " ]";
    } else {
      os << "[";
      for (IndexType i = 0; i < rhs.shape(dim); ++i) {
        md[dim] = i;
        print(os, rhs, md, dim + 1);
      }
      os << "]";
    }
  }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os, MultiDimMap const &rhs) {
    os << "MultiDimMap"
       << "("
       << "[" << rhs.size() << " entries";
    os << ", dims ";
    for (IndexType i = 0; i < rhs.shape().size(); ++i) {
      if (i) {
        os << "x";
      }
      os << rhs.shape(i);
    }
    os << "]";
    DimType dims;
    dims.fill(0);
    print(os, rhs, dims, 0);
    os << ")";
    return os;
  }

  DimType dims_;            ///< dimension of the map
  std::vector<T> elements_; ///< elements in the map
};

} // namespace container

OPENPARF_END_NAMESPACE

#endif
