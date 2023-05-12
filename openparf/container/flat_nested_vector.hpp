/**
 * @file   flat_nested_vector.hpp
 * @author Yibo Lin
 * @date   Mar 2020
 */

#ifndef OPENPARF_CONTAINERS_FLAT_NESTED_VECTOR_HPP_
#define OPENPARF_CONTAINERS_FLAT_NESTED_VECTOR_HPP_

#include <vector>

#include "util/util.h"

OPENPARF_BEGIN_NAMESPACE

namespace container {

/// @brief A flatten representation for array of array.
/// It consists of two arrays, a data array and an index array indicating
/// the start of each row (the second dimension).
template <typename T, typename V>
class FlatNestedVector {
 public:
  using ValueType = T;
  using IndexType = V;
  using ConstIteratorType = typename std::vector<ValueType>::const_iterator;
  using IteratorType = typename std::vector<ValueType>::iterator;

  /// @brief constant proxy
  class ConstVectorProxy {
   public:
    /// @brief constructor
    ConstVectorProxy(ConstIteratorType b, ConstIteratorType e)
        : begin_(b), end_(e) {}
    /// @brief begin iterator
    ConstIteratorType begin() const { return begin_; }
    /// @brief end iterator
    ConstIteratorType end() const { return end_; }
    IndexType size() const { return std::distance(begin_, end_); }
    ValueType const &operator()(IndexType i) const { return *(begin_ + i); }
    ValueType const &at(IndexType i) const {
      openparfAssert(i < size());
      return this->operator()(i);
    }

   protected:
    ConstIteratorType begin_;
    ConstIteratorType end_;
  };

  /// @brief constant proxiy
  class VectorProxy {
   public:
    /// @brief constructor
    VectorProxy(IteratorType b, IteratorType e) : begin_(b), end_(e) {}
    /// @brief begin iterator
    IteratorType begin() { return begin_; }
    /// @brief end iterator
    IteratorType end() { return end_; }
    IndexType size() const { return std::distance(begin_, end_); }
    ValueType &operator()(IndexType i) { return *(begin_ + i); }
    ValueType &at(IndexType i) {
      openparfAssert(i < size());
      return this->operator()(i);
    }

   protected:
    IteratorType begin_;
    IteratorType end_;
  };

  /// @brief default constructor
  FlatNestedVector() { dim2_start_ = {0}; }

  /// @brief constructor
  FlatNestedVector(std::vector<std::vector<ValueType>> const &rhs) {
    dim2_start_.assign(rhs.size() + 1, 0);
    for (IndexType i = 0, ie = rhs.size(); i < ie; ++i) {
      dim2_start_[i + 1] += dim2_start_[i] + rhs[i].size();
    }
    data_.resize(dim2_start_.back());
    IndexType count = 0;
    for (auto const &array : rhs) {
      for (auto const &v : array) {
        data_[count++] = v;
      }
    }
  }

  /// @brief copy constructor
  FlatNestedVector(FlatNestedVector const &rhs) { copy(rhs); }

  /// @brief move constructor
  FlatNestedVector(FlatNestedVector &&rhs) { move(std::move(rhs)); }

  /// @brief copy assignment
  FlatNestedVector &operator=(FlatNestedVector const &rhs) {
    if (this != &rhs) {
      copy(rhs);
    }
    return *this;
  }

  /// @brief move assignment
  FlatNestedVector &operator=(FlatNestedVector &&rhs) {
    if (this != &rhs) {
      move(std::move(rhs));
    }
    return *this;
  }

  /// @brief number of all elements
  IndexType size() const { return data_.size(); }

  /// @brief size of the first dimension
  IndexType size1() const { return dim2_start_.size() - 1; }

  /// @brief size of the second dimension
  IndexType size2(IndexType i) const { return indexEndAt(i) - indexBeginAt(i); }

  /// @brief capacity of the data array
  IndexType capacity() const { return data_.capacity(); }

  /// @brief capacity of the first dimension
  IndexType capacity1() const { return dim2_start_.capacity() - 1; }

  /// @brief reserve capacity
  /// @param s1 size of the first dimension
  /// @param s total number of elements
  void reserve(IndexType s1, IndexType s) {
    data_.reserve(s);
    dim2_start_.reserve(s1 + 1);
  }

  /// @brief getter the range of data at the first dimension
  ConstVectorProxy at(IndexType i) const {
    return ConstVectorProxy(data_.begin() + indexBeginAt(i),
                            data_.begin() + indexEndAt(i));
  }

  /// @brief getter the range of data at the first dimension
  VectorProxy at(IndexType i) {
    openparfAssert(i < size1());
    return VectorProxy(data_.begin() + indexBeginAt(i),
                       data_.begin() + indexEndAt(i));
  }

  /// @brief getter for one element
  ValueType const &at(IndexType i, IndexType j) const {
    openparfAssert(j < size2(i));
    return data_[indexBeginAt(i) + j];
  }

  /// @brief getter for all data
  std::vector<ValueType> const &data() const { return data_; }

  /// @brief getter for all data
  std::vector<ValueType> &data() { return data_; }

  /// @brief getter for all index starts
  std::vector<IndexType> const &indexBeginData() const { return dim2_start_; }

  /// @brief getter for all index starts
  std::vector<IndexType> &indexBeginData() { return dim2_start_; }

  /// @brief begin iterator for the data array
  ConstIteratorType begin() const { return data_.begin(); }

  /// @brief begin iterator for the data array
  IteratorType begin() { return data_.begin(); }

  /// @brief end iterator for the data array
  ConstIteratorType end() const { return data_.end(); }

  /// @brief end iterator for the data array
  IteratorType end() { return data_.end(); }

  /// @brief getter for one element
  ValueType &at(IndexType i, IndexType j) {
    openparfAssertMsg(j < size2(i), "j = %lu, size2(%lu) = %lu", j, i,
                      size2(i));
    return data_[indexBeginAt(i) + j];
  }

  /// @brief index of the start of the ith row
  IndexType indexBeginAt(IndexType i) const {
    openparfAssertMsg(i < size1(), "i = %lu, size1() = %lu", i, size1());
    return dim2_start_[i];
  }

  /// @brief index of the end of the ith row
  IndexType indexEndAt(IndexType i) const {
    openparfAssertMsg(i < size1(), "i = %lu, size1() = %lu", i, size1());
    return dim2_start_[i + 1];
  }

  /// @brief set the index of the start of the ith row
  /// Be aware that there is always a 0 at the first entry
  void pushBackIndexBegin(IndexType v) { dim2_start_.push_back(v); }

  /// @brief append at the last row
  void pushBack(ValueType const &v) {
    data_.push_back(v);
    dim2_start_.back() += 1;
  }

  /// @brief append at the last row
  void pushBack(ValueType &&v) {
    data_.push_back(std::move(v));
    dim2_start_.back() += 1;
  }

  /// @brief append at the last row
  template <class... Args>
  void emplaceBack(Args &&... args) {
    data_.emplace_back(std::forward<Args>(args)...);
    dim2_start_.back() += 1;
  }

  /// @brief clear data
  void clear() {
    data_.clear();
    dim2_start_.clear();
    dim2_start_.push_back(0);
  }

  /// @brief summarize memory usage of the object in bytes
  IndexType memory() const {
    return sizeof(data_) + sizeof(dim2_start_) +
           data_.capacity() * sizeof(ValueType) +
           dim2_start_.capacity() * sizeof(IndexType);
  }

 protected:
  /// @brief copy the object
  void copy(FlatNestedVector const &rhs) {
    data_ = rhs.data_;
    dim2_start_ = rhs.dim2_start_;
  }

  /// @brief move the object
  void move(FlatNestedVector &&rhs) {
    data_ = std::move(rhs.data_);
    dim2_start_ = std::move(rhs.dim2_start_);
  }

  /// @brief overload output stream
  friend std::ostream &operator<<(std::ostream &os,
                                  FlatNestedVector const &rhs) {
    os << "FlatNestedVector"
       << "("
       << "[" << rhs.size() << " entries, " << rhs.size1() << " rows]";
    for (IndexType i = 1; i < (IndexType)rhs.dim2_start_.size(); ++i) {
      os << " [" << rhs.size2(i - 1) << "][";
      for (IndexType j = rhs.dim2_start_[i - 1], je = rhs.dim2_start_[i];
           j < je; ++j) {
        os << " " << rhs.data_[j];
      }
      os << " ]";
    }
    os << ")";
    return os;
  }

  std::vector<ValueType> data_;  ///< all data
  std::vector<IndexType>
      dim2_start_;  ///< length of (number of size1 + 1),
                    ///< record the start of the second dimension
};

}  // namespace container

OPENPARF_END_NAMESPACE

#endif
