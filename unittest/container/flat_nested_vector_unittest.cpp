/**
 * @file   flat_nested_vector_unittest.cpp
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

using container::FlatNestedVector;

namespace unittest {

/// GTest class for fft module testing
class FlatNestedVectorTest : public ::testing::Test {
public:
  using IndexType = std::size_t;

  void testConstructor() const {
    // test constructor
    std::vector<std::vector<int>> values = {
        {0, 1, 2, 3}, {4, 5}, {6}, {7, 8, 9}};
    FlatNestedVector<int, IndexType> fn(values);

    // test size
    ASSERT_EQ(fn.size1(), values.size());
    for (IndexType i = 0; i < fn.size1(); ++i) {
      ASSERT_EQ(fn.size2(i), values[i].size());
    }

    // test contents
    for (IndexType i = 0; i < values.size(); ++i) {
      for (IndexType j = 0; j < values[i].size(); ++j) {
        ASSERT_EQ(fn.at(i, j), values[i][j]);
      }
    }

    // test iterator
    for (IndexType i = 0; i < values.size(); ++i) {
      IndexType j = 0;
      for (auto const &v : fn.at(i)) {
        ASSERT_EQ(v, values[i][j]);
        ++j;
      }
      for (j = fn.indexBeginAt(i); j < fn.indexEndAt(i); ++j) {
        ASSERT_EQ(fn.at(i, j - fn.indexBeginAt(i)),
                  values[i][j - fn.indexBeginAt(i)]);
      }
    }
  }

  void testModifier() const {
    // test constructor
    std::vector<std::vector<int>> values = {
        {0, 1, 2, 3}, {4, 5}, {6}, {7, 8, 9}};
    FlatNestedVector<int, IndexType> fn(values);

    // test iterator
    for (IndexType i = 0; i < values.size(); ++i) {
      IndexType j = 0;
      for (auto &v : fn.at(i)) {
        v *= 10;
        ++j;
      }
    }

    // test iterator
    for (auto &v : fn) {
      v /= 10;
    }

    // test contents
    for (IndexType i = 0; i < values.size(); ++i) {
      for (IndexType j = 0; j < values[i].size(); ++j) {
        ASSERT_EQ(fn.at(i, j), values[i][j]);
      }
    }
  }

  void testSize() const {
    FlatNestedVector<int, IndexType> fn;

    fn.reserve(10, 20);

    ASSERT_EQ(fn.capacity(), 20U);
    ASSERT_EQ(fn.capacity1(), 10U);
  }

  void testOutput() const {
    // test constructor
    std::vector<std::vector<int>> values = {
        {0, 1, 2, 3}, {4, 5}, {6}, {7, 8, 9}};
    FlatNestedVector<int, IndexType> fn(values);

    std::ostringstream oss;
    oss << fn;
    ASSERT_EQ(oss.str(), "FlatNestedVector([10 entries, 4 rows] [4][ 0 1 2 3 ] "
                         "[2][ 4 5 ] [1][ 6 ] [3][ 7 8 9 ])");
    std::cout << oss.str() << std::endl;
  }
};

TEST_F(FlatNestedVectorTest, Constructor) { testConstructor(); }

TEST_F(FlatNestedVectorTest, Modifier) { testModifier(); }

TEST_F(FlatNestedVectorTest, Size) { testSize(); }

TEST_F(FlatNestedVectorTest, Output) { testOutput(); }

} // namespace unittest

OPENPARF_END_NAMESPACE
