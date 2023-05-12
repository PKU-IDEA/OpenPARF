/**
 * @file   multi_dim_map_unittest.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */
#include "container/container.hpp"
#include "util/util.h"
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

OPENPARF_BEGIN_NAMESPACE

using container::MultiDimMap;

namespace unittest {

/// GTest class for fft module testing
class MultiDimMapTest : public ::testing::Test {
public:
  using IndexType = std::size_t;

  template <IndexType Dims> class Element {
  public:
    IndexType id(IndexType dim) const { return id_[dim]; }

    template <class... Args> void setId(Args... args) { id_ = {args...}; }

    IndexType value() const { return value_; }

    void setValue(IndexType v) { value_ = v; }

  protected:
    friend std::ostream &operator<<(std::ostream &os, Element const &rhs) {
      os << "("
         << "(";
      for (IndexType i = 0; i < Dims; ++i) {
        if (i) {
          os << ", ";
        }
        os << rhs.id_[i];
      }
      os << "), " << rhs.value_;
      os << ")";
      return os;
    }

    std::array<IndexType, Dims> id_;
    IndexType value_;
  };

  void test2D() const {
    // test constructor
    MultiDimMap<Element<2>, 2, true> map2d_1;
    MultiDimMap<Element<2>, 2, true> map2d(2, 3);

    // test size
    decltype(map2d_1)::DimType dim2d_1({0, 0});
    ASSERT_EQ(map2d_1.shape(), dim2d_1);
    decltype(map2d)::DimType dim2d({2, 3});
    ASSERT_EQ(map2d.shape(), dim2d);

    // test contents
    for (IndexType i = 0; i < map2d.shape(0); ++i) {
      for (IndexType j = 0; j < map2d.shape(1); ++j) {
        map2d(i, j).setValue(i + j);
      }
    }
    for (IndexType i = 0; i < map2d.shape(0); ++i) {
      for (IndexType j = 0; j < map2d.shape(1); ++j) {
        ASSERT_EQ(map2d.at(i, j).value(), i + j);
      }
    }
  }

  void test3D() const {
    // test constructor
    MultiDimMap<Element<3>, 3, true> map3d(2, 3, 4);

    // test size
    decltype(map3d)::DimType dim3d({2, 3, 4});
    ASSERT_EQ(map3d.shape(), dim3d);

    // test contents
    for (IndexType i = 0; i < map3d.shape(0); ++i) {
      for (IndexType j = 0; j < map3d.shape(1); ++j) {
        for (IndexType k = 0; k < map3d.shape(2); ++k) {
          map3d(i, j, k).setValue(i + j + k);
          ASSERT_EQ(map3d(i, j, k).value(), i + j + k);
        }
      }
    }
    for (IndexType i = 0; i < map3d.shape(0); ++i) {
      for (IndexType j = 0; j < map3d.shape(1); ++j) {
        for (IndexType k = 0; k < map3d.shape(2); ++k) {
          ASSERT_EQ(map3d(i, j, k).value(), i + j + k);
        }
      }
    }

    // test iterator
    for (auto const &e : map3d) {
      ASSERT_EQ(e.value(), e.id(0) + e.id(1) + e.id(2));
    }
  }

  void testOutput() const {
    // test constructor
    MultiDimMap<Element<2>, 2, true> map2d(2, 3);
    MultiDimMap<Element<3>, 3, true> map3d(2, 3, 4);

    for (IndexType i = 0; i < map2d.shape(0); ++i) {
      for (IndexType j = 0; j < map2d.shape(1); ++j) {
        map2d(i, j).setValue(i + j);
      }
    }
    for (IndexType i = 0; i < map3d.shape(0); ++i) {
      for (IndexType j = 0; j < map3d.shape(1); ++j) {
        for (IndexType k = 0; k < map3d.shape(2); ++k) {
          map3d(i, j, k).setValue(i + j + k);
        }
      }
    }

    // test output
    {
      std::ostringstream oss;
      oss << map2d;
      ASSERT_EQ(oss.str(), "MultiDimMap([6 entries, dims 2x3][[ ((0, 0), 0) "
                           "((0, 1), 1) ((0, 2), 2) ][ ((1, 0), 1) ((1, 1), 2) "
                           "((1, 2), 3) ]])");
      std::cout << oss.str() << std::endl;
    }
    {
      std::ostringstream oss;
      oss << map3d;
      ASSERT_EQ(oss.str(), "MultiDimMap([24 entries, dims 2x3x4][[[ ((0, 0, "
                           "0), 0) ((0, 0, 1), 1) ((0, 0, 2), 2) ((0, 0, 3), "
                           "3) ][ ((0, 1, 0), 1) ((0, 1, 1), 2) ((0, 1, 2), 3) "
                           "((0, 1, 3), 4) ][ ((0, 2, 0), 2) ((0, 2, 1), 3) "
                           "((0, 2, 2), 4) ((0, 2, 3), 5) ]][[ ((1, 0, 0), 1) "
                           "((1, 0, 1), 2) ((1, 0, 2), 3) ((1, 0, 3), 4) ][ "
                           "((1, 1, 0), 2) ((1, 1, 1), 3) ((1, 1, 2), 4) ((1, "
                           "1, 3), 5) ][ ((1, 2, 0), 3) ((1, 2, 1), 4) ((1, 2, "
                           "2), 5) ((1, 2, 3), 6) ]]])");
      std::cout << oss.str() << std::endl;
    }

    // test copy/move
    decltype(map3d) map3d_copy = map3d;
    decltype(map3d) map3d_move = std::move(map3d);

    for (IndexType i = 0; i < map3d.shape(0); ++i) {
      for (IndexType j = 0; j < map3d.shape(1); ++j) {
        for (IndexType k = 0; k < map3d.shape(2); ++k) {
          ASSERT_EQ(map3d_copy(i, j, k).value(), i + j + k);
          ASSERT_EQ(map3d_move(i, j, k).value(), i + j + k);
        }
      }
    }
  }
};

TEST_F(MultiDimMapTest, 2D) { test2D(); }

TEST_F(MultiDimMapTest, 3D) { test3D(); }

TEST_F(MultiDimMapTest, Output) { testOutput(); }

} // namespace unittest

OPENPARF_END_NAMESPACE
