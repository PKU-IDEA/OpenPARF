#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : unittest_stopwatch.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 08.13.2020
# Last Modified Date: 08.13.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
import time
import unittest

import openparf.py_utils.stopwatch as stopwatch


class MyTestCase(unittest.TestCase):
    def test_something(self):
        sw = stopwatch.Stopwatch()
        sw.start()
        time.sleep(1)
        print("elapsed time = {} ns ".format(sw.elapsed(stopwatch.Stopwatch.TimeFormat.kNanoSecond)))
        print("elapsed time = {} us ".format(sw.elapsed(stopwatch.Stopwatch.TimeFormat.kMicroSecond)))
        print("elapsed time = {} ms ".format(sw.elapsed()))
        print("elapsed time = {} s ".format(sw.elapsed(stopwatch.Stopwatch.TimeFormat.kSecond)))
        self.assertEqual(sw.elapsed(stopwatch.Stopwatch.TimeFormat.kSecond), 1)
        sw.lap()
        time.sleep(2)
        sw.lap()
        gross_elapsed_time, internals = sw.elapsed_lap(stopwatch.Stopwatch.TimeFormat.kSecond)
        self.assertEqual(gross_elapsed_time, 3)


if __name__ == '__main__':
    unittest.main()
