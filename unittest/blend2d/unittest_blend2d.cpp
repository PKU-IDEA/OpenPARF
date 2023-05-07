/**
 * File              : unittest_blend2d.cpp
 * Author            : Yibo Lin <yibolin@pku.edu.cn>
 * Date              : 05.03.2020
 * Last Modified Date: 05.03.2020
 * Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
 */

#include "util/util.h"
#include <blend2d.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

OPENPARF_BEGIN_NAMESPACE

namespace unitest {

/// GTest class for fft module testing
class Blend2DTest : public ::testing::Test {
public:
  void testSimple() {
  BLImage img(480, 480, BL_FORMAT_PRGB32);

  // Attach a rendering context into `img`.
  BLContext ctx(img);

  // Clear the image.
  ctx.setCompOp(BL_COMP_OP_SRC_COPY);
  ctx.fillAll();

  // Fill some path.
  BLPath path;
  path.moveTo(26, 31);
  path.cubicTo(642, 132, 587, -136, 25, 464);
  path.cubicTo(882, 404, 144, 267, 27, 31);

  ctx.setCompOp(BL_COMP_OP_SRC_OVER);
  ctx.setFillStyle(BLRgba32(0xFFFFFFFF));
  ctx.fillPath(path);

  // Detach the rendering context from `img`.
  ctx.end();

  // Let's use some built-in codecs provided by Blend2D.
  BLImageCodec codec;
  img.writeToFile("blend2d_test_simple.bmp");

  }
};

TEST_F(Blend2DTest, Simple) { testSimple(); }

} // namespace unitest

OPENPARF_END_NAMESPACE
