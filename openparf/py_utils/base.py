#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : base.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 08.13.2020
# Last Modified Date: 08.13.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>

from typing import Callable
from contextlib import ExitStack
from functools import partial


class DeferredAction(ExitStack):
    def __call__(self, func: Callable, *args, **kwargs):
        self.callback(partial(func, *args, **kwargs))


def log_dict(log: Callable, dic: dict):
    d = max(map(len, dic.keys()))
    content_list = [k.ljust(d) + ":" + str(dic[k]) for k in dic]
    log("\n" + "\n".join(content_list))


def foo():
    with DeferredAction() as defer:
        defer(print, 2)
        print(1)
        with DeferredAction() as defer2:
            defer2(print, 3)
            defer2(print, 4)
            print(5)
        defer(print, 7)
        defer(print, 6)


if __name__ == "__main__":
    foo()
