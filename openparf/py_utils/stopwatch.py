#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : stopwatch.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 08.13.2020
# Last Modified Date: 08.13.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
from enum import Enum
import time

from typing import List


class Stopwatch(object):
    class TimeFormat(Enum):
        kNanoSecond = 0
        kMicroSecond = 1
        kMillSecond = 2
        kSecond = 3

    def __init__(self):
        self._start_time_ns = None
        self._laps = None

    def start(self):
        """Start the stopwatch.
        """
        self._start_time_ns = time.time_ns()
        self._laps = [self._start_time_ns]

    def elapsed(self, time_format: TimeFormat = TimeFormat.kMillSecond) -> int:
        current_time_ns = time.time_ns()
        return self._ticks(self._start_time_ns, current_time_ns, time_format)

    def lap(self, time_format: TimeFormat = TimeFormat.kMillSecond) -> int:
        current_time_ns = time.time_ns()
        last_record_ns = self._laps[-1]
        self._laps.append(current_time_ns)
        return self._ticks(last_record_ns, current_time_ns, time_format)

    def elapsed_lap(self, time_format: TimeFormat = TimeFormat.kMillSecond) -> (int, List[int]):
        lap_times = []
        for i in range(len(self._laps) - 1):
            lap_end = self._laps[i + 1]
            lap_start = self._laps[i]
            lap_times.append(self._ticks(lap_start, lap_end, time_format))
        gross_elapsed_time = self._ticks(self._start_time_ns, self._laps[-1], time_format)
        return gross_elapsed_time, lap_times

    @staticmethod
    def _format(ns_count: int, time_format: TimeFormat = TimeFormat.kMillSecond) -> int:
        if time_format is Stopwatch.TimeFormat.kNanoSecond:
            return ns_count
        elif time_format is Stopwatch.TimeFormat.kMicroSecond:
            up = 1 if (ns_count // 100) % 10 >= 5 else 0
            return (ns_count // 1000) + up
        elif time_format is Stopwatch.TimeFormat.kMillSecond:
            up = 1 if (ns_count // 100000) % 10 >= 5 else 0
            return (ns_count // 1000000) + up
        elif time_format is Stopwatch.TimeFormat.kSecond:
            up = 1 if (ns_count // 100000000) % 10 >= 5 else 0
            return (ns_count // 1000000000) + up
        else:
            raise TypeError("The type of parameter 'time_format' should be Stopwatch.TimeFormat.")

    @staticmethod
    def _ticks(start_time_ns: int, end_time_ns: int, time_format: TimeFormat = TimeFormat.kMillSecond) -> int:
        ns_count = end_time_ns - start_time_ns
        return Stopwatch._format(ns_count, time_format)


class IntervalStopwatch(Stopwatch):
    class TimeInternal(object):
        def __init__(self, gross_elapsed_time_ns, laps):
            self._gross_elapsed_time_ns = gross_elapsed_time_ns
            self._laps = laps

        def get_gross_elapsed_time(self, time_format: Stopwatch.TimeFormat = Stopwatch.TimeFormat.kMillSecond):
            return Stopwatch._format(self._gross_elapsed_time_ns, time_format)

    def __init__(self):
        super(IntervalStopwatch, self).__init__()
        _internals = []

    def start(self):
        if self._start_time_ns is not None:
            raise RuntimeError("Stopwatch has started.")
        super(IntervalStopwatch, self).start()

    def stop(self, time_format: Stopwatch.TimeFormat = Stopwatch.TimeFormat.kMillSecond) -> (int, int):
        if self._start_time_ns is None:
            raise RuntimeError("Stopwatch hasn't started.")
        current_time_ns = time.time_ns()
        self._laps.append(current_time_ns)
        gross_elapsed_time_ns = current_time_ns - self._start_time_ns
        self._internals.append(IntervalStopwatch.TimeInternal(gross_elapsed_time_ns, self._laps))
        self._start_time_ns, self._laps = None, None
        return current_time_ns, self._format(gross_elapsed_time_ns, time_format)

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_value, tb):
        self.stop()

    # The getters
    @property
    def internals(self):
        return self._internals

    # Protected members
    _internals = []
