#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : openparf.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.22.2020
# Last Modified Date: 08.26.2021
# Last Modified By  : Jing Mai <jingmai@pku.edu.cn>

import os
import sys
import logging
import argparse
try:
    from loguru import logger
    useLoguru = True
except ModuleNotFoundError:
    useLoguru = False
    logger = logging.getLogger(__name__)
    pass
from openparf.flow import place, route
from openparf.params import Params

import torch


# Intercept standard logging messages towards the Loguru sinks
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


if __name__ == "__main__":
    """
    @brief main function to invoke the entire flow.
    """

    params = Params()

    # TODO(Jing Mai, magic3007@pku.edu.cn): Append `params.printHelp' when `--help` is triggered.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='OpenPARF: A FPGA Placement Framework.')
    parser.add_argument('--log', type=str, default=None, help='path to the logging file')
    parser.add_argument('--config', type=str, required=True, help='path the parameter json file')
    parser.add_argument('--repo', type=str, default=os.getcwd(),
                        help='Full path to parent directory of Aim repo - the .aim directory. '
                             'By default current working directory.')
    parser.add_argument('--expr', type=str, default=None, help="A description to this experiment.")

    args = parser.parse_args()

    log_file_path = args.log
    config_file_path = args.config
    params.repo = args.repo
    params.experiment = args.expr
    if useLoguru:
        logger.remove(handler_id=None)
        logging.basicConfig(handlers=[InterceptHandler()], level=0)
        format_str = "<level>[{level:<7}]</level> <green>{elapsed} sec</green> | {name}:{line} - {message}"
        logger_stderr_handler = logger.add(
            sys.stderr,
            colorize=True,
            format=format_str,
            level="INFO")
        if log_file_path is not None:
            logger_file_handler = logger.add(
                log_file_path,
                mode='w',
                colorize=False,
                format=format_str,
                level='INFO')
    else:
        logging.basicConfig(level=logging.INFO)

    # load parameters
    params.load(config_file_path)
    params.printWelcome()

    logging.info("parameters = %s" % params)
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % params.num_threads

    pl_path = "%s/%s.pl" % (params.result_dir, params.design_name())
    # run placement
    place(params, pl_path)

    # run routing
    if params.route_flag:
        route(params, pl_path)
