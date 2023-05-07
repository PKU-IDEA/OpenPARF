/**
 * @file   main.cpp
 * @author Yibo Lin
 * @date   Mar 2020
 */
#include <gtest/gtest.h>

std::string test_dir;

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  assert(argc == 2);
  test_dir = argv[1];
  return RUN_ALL_TESTS();
}
