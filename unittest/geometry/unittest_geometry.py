##
# @file   unittest_geometry.py
# @author Yibo Lin
# @date   Mar 2020
#

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

class GeometryTest (unittest.TestCase):
  def testPoint2D(self):
    point_int = of.geometry.Point2DInt(1, 2)
    point_int_copy = of.geometry.Point2DInt()
    point_int_copy = copy.copy(point_int)
    self.assertEqual(point_int.x(), point_int_copy.x())
    self.assertEqual(point_int.y(), point_int_copy.y())

    point_uint = of.geometry.Point2DUint()
    point_uint.setX(1)
    point_uint.setY(2)
    self.assertEqual(point_int.x(), point_uint.x())
    self.assertEqual(point_int.y(), point_uint.y())
    point_uint.set(3, 4)
    self.assertEqual(point_uint.x(), 3)
    self.assertEqual(point_uint.y(), 4)

  def testPoint3D(self):
    point_float = of.geometry.Point3DFloat(1.0, 2, 3)
    point_float_copy = of.geometry.Point3DFloat()
    point_float_copy = point_float
    self.assertEqual(point_float.x(), point_float_copy.x())
    self.assertEqual(point_float.y(), point_float_copy.y())
    self.assertEqual(point_float.z(), point_float_copy.z())

    point_double = of.geometry.Point3DUint()
    point_double.setX(1)
    point_double.setY(2)
    point_double.setZ(3)
    self.assertEqual(point_float.x(), point_double.x())
    self.assertEqual(point_float.y(), point_double.y())
    self.assertEqual(point_float.z(), point_double.z())
    point_double.set(3, 4, 5)
    self.assertEqual(point_double.x(), 3)
    self.assertEqual(point_double.y(), 4)
    self.assertEqual(point_double.z(), 5)

  def testBox(self):
    box_int = of.geometry.BoxInt(1, 2, 3, 4)
    box_uint = copy.copy(box_int)
    self.assertEqual(box_int.xl(), box_uint.xl())
    self.assertEqual(box_int.yl(), box_uint.yl())
    self.assertEqual(box_int.xh(), box_uint.xh())
    self.assertEqual(box_int.yh(), box_uint.yh())

    box_float = of.geometry.BoxFloat()
    box_float.set(6, 7, 8, 9)
    self.assertEqual(box_int.xl() + 5, box_float.xl())
    self.assertEqual(box_int.yl() + 5, box_float.yl())
    self.assertEqual(box_int.xh() + 5, box_float.xh())
    self.assertEqual(box_int.yh() + 5, box_float.yh())

    box_double = of.geometry.BoxFloat(1, 2, 3, 4)
    self.assertEqual(box_double.xl() + 5, box_float.xl())
    self.assertEqual(box_double.yl() + 5, box_float.yl())
    self.assertEqual(box_double.xh() + 5, box_float.xh())
    self.assertEqual(box_double.yh() + 5, box_float.yh())

  def testSize2D(self):
    size_int = of.geometry.Size2DInt(1, 2)
    self.assertEqual(size_int.x(), 1)
    self.assertEqual(size_int.y(), 2)

    size_int_copy = copy.copy(size_int)
    size_int_copy.set(3, 4)
    self.assertEqual(size_int.x() + 2, size_int_copy.x())
    self.assertEqual(size_int.y() + 2, size_int_copy.y())

    size_float = of.geometry.Size2DFloat(1.2, 3.4)
    size_double = of.geometry.Size2DDouble(1.4, 3.6)
    self.assertAlmostEqual(size_float.product(), 1.2 * 3.4, places=6)
    self.assertAlmostEqual(size_double.product(), 1.4 * 3.6, places=6)
    self.assertAlmostEqual(size_float.x() + 0.2, size_double.x(), places=6)
    self.assertAlmostEqual(size_float.y() + 0.2, size_double.y(), places=6)

  def testSize3D(self):
    size_int = of.geometry.Size3DInt(1, 2, 3)
    self.assertEqual(size_int.x(), 1)
    self.assertEqual(size_int.y(), 2)
    self.assertEqual(size_int.z(), 3)

    size_int_copy = copy.copy(size_int)
    size_int_copy.set(3, 4, 5)
    self.assertEqual(size_int.x() + 2, size_int_copy.x())
    self.assertEqual(size_int.y() + 2, size_int_copy.y())
    self.assertEqual(size_int.z() + 2, size_int_copy.z())

    size_float = of.geometry.Size3DFloat(1.2, 3.4, 5.6)
    size_double = of.geometry.Size3DDouble(1.4, 3.6, 5.8)
    self.assertAlmostEqual(size_float.x() + 0.2, size_double.x(), places=6)
    self.assertAlmostEqual(size_float.y() + 0.2, size_double.y(), places=6)
    self.assertAlmostEqual(size_float.z() + 0.2, size_double.z(), places=6)

if __name__ == '__main__':
  if len(sys.argv) < 2:
    pass
  else:
    sys.argv.pop()
  unittest.main()
