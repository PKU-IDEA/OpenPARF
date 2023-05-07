#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : unittest_blend2d.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 05.03.2020
# Last Modified Date: 05.03.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>

import pdb
import os
import sys
import copy
import unittest

if len(sys.argv) < 2:
  print("usage: python script.py [project_dir]")
  project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
else:
  project_dir = sys.argv[1]
print("use project_dir = %s" % (project_dir))

sys.path.append(project_dir)
from openparf import openparf as of
sys.path.pop()

class Blend2DTest (unittest.TestCase):
  def testSingle(self):
      image = of.Image(128, 128, 4, 4, 24, 24)
      image.setFillColor(255, 255, 255, 1.0)
      image.fillRect(4, 4, 2, 2)
      #image.setStrokeColor(255, 255, 255, 1.0)
      #image.strokeRect(4, 4, 2, 2)

      image.end()
      filename = "blend2d_test_single.bmp"
      print("write to %s" % (filename))
      image.write(filename)

  def testBatch(self):
      image = of.Image(128, 128, 4, 4, 24, 24)
      image.setFillColor(100, 100, 100, 1.0)
      image.fillRects([[4, 4], [8, 8]], [[2, 2], [3, 4]])
      image.setStrokeColor(0xFFFFFFFF)
      image.strokeRects([[4, 4], [8, 8]], [[2, 2], [3, 4]])

      # this is not working yet
      image.text(10, 10, "test text", 10)

      image.end()
      filename = "blend2d_test_batch.bmp"
      print("write to %s" % (filename))
      image.write(filename)

if __name__ == '__main__':
  if len(sys.argv) < 2:
    pass
  else:
    sys.argv.pop()
  unittest.main()
