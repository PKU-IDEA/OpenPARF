#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : draw_place.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.29.2020
# Last Modified Date: 04.29.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
##

import os
import time
import numpy as np
import logging
import typing
import pdb

from .. import openparf as of


def draw_place(pos,
               inst_sizes,
               area_type_inst_groups,
               die_area,
               movable_range,
               fixed_range,
               filler_range,
               filename,
               iteration=None,
               target_area_types=None):
    """ python implementation of placement drawer.

    :param pos center locations of cells
    :param inst_sizes: array of instance sizes
    :param area_type_inst_groups:
    :param die_area: (xl, yl, xh, yh) of the die area
    :param movable_range: range of movable instances
    :param fixed_range: range of fixed instances
    :param filler_range: range of fillers
    :param filename: output filename
    :param iteration: current optimization step
    :param target_area_types:
    """
    color_map = np.array([[125, 0, 0],
                          [0, 125, 0],
                          [0, 0, 125],
                          [255, 64, 255],
                          [0, 84, 147],
                          [142, 250, 0],
                          [255, 212, 121],
                          [100, 192, 4],
                          [4, 192, 100]])

    layout_xl = xl = die_area[0]
    layout_yl = yl = die_area[1]
    layout_xh = xh = die_area[2]
    layout_yh = yh = die_area[3]

    if target_area_types is None:
        target_area_types = np.arange(len(area_type_inst_groups))
    tt = time.time()
    aspect_ratio = (yh - yl) / (xh - xl)
    height = 800
    width = int(height / aspect_ratio)

    # convert to lower left
    locations = pos - inst_sizes / 2

    for area_type in target_area_types:
        img = of.Image(width, height, layout_xl, layout_yl, layout_xh,
                       layout_yh)

        # draw layout region
        img.setFillColor(0xFFFFFFFF)
        img.setStrokeColor(25, 25, 25, 0.8)
        img.fillRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)
        img.strokeRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)

        at_inst_ids = area_type_inst_groups[area_type].cpu().numpy()
        # draw cells
        # draw fixed macros
        if fixed_range and fixed_range[0] < fixed_range[1]:
            img.setFillColor(255, 0, 0, 0.5)
            img.fillRects(locations[fixed_range[0]:fixed_range[1]],
                          inst_sizes[fixed_range[0]:fixed_range[1]])
            img.setStrokeColor(0, 0, 0, 1)
            img.strokeRects(locations[fixed_range[0]:fixed_range[1]],
                            inst_sizes[fixed_range[0]:fixed_range[1]])
        # draw fillers
        if filler_range and filler_range[0] < filler_range[1]:  # filler is included
            inst_ids = at_inst_ids[np.where(
                np.logical_and(at_inst_ids >= filler_range[0], at_inst_ids < filler_range[1]))]
            img.setFillColor(color_map[area_type, 0], color_map[area_type, 1],
                             color_map[area_type, 2], 0.5)
            img.fillRects(locations[inst_ids], inst_sizes[inst_ids])

        # draw cells
        if movable_range and movable_range[0] < movable_range[1]:
            inst_ids = at_inst_ids[
                np.where(np.logical_and(at_inst_ids >= movable_range[0], at_inst_ids < movable_range[1]))]
            img.setFillColor(color_map[area_type, 0], color_map[area_type, 1],
                             color_map[area_type, 2], 0.5)
            img.fillRects(locations[inst_ids], inst_sizes[inst_ids])

        # show iteration
        if iteration:
            img.setFillColor(0, 0, 0, 1)
            img.text((xl + xh) / 2, (yl + yh) / 2, '{:04}'.format(iteration),
                     32)

        if not os.path.exists(os.path.dirname(filename)):
            os.system("mkdir -p %s" % (os.path.dirname(filename)))
        draw_filename = filename.replace(".bmp", "_at{:02}.bmp".format(area_type))
        img.end()
        img.write(draw_filename)  # Output to file

    logging.info("plotting to %s takes %.3f seconds" %
                 (filename, time.time() - tt))


def draw_place_with_clock_region_assignments(
    pos : np.ndarray,
    inst_sizes: np.ndarray,
    area_type_inst_groups,
    die_area: typing.Tuple[int, int, int, int],
    movable_range,
    fence_region_boxes,
    movable_inst_to_clock_region,
    filename: str,
    target_area_types
):
    """
        @param pos: center locations of cells
        @param inst_sizes: array of instance sizes
        :param area_type_inst_groups:
        :param die_area: (xl, yl, xh, yh) of the die area
        :param movable_range: range of movable instances
        :param filename: output filename
    """
    color_map = np.array([[125, 0, 0],
                        [0, 125, 0],
                        [0, 0, 125],
                        [255, 64, 255],
                        [0, 84, 147],
                        [142, 250, 0],
                        [255, 212, 121]])

    layout_xl = xl = die_area[0]
    layout_yl = yl = die_area[1]
    layout_xh = xh = die_area[2]
    layout_yh = yh = die_area[3]
    aspect_ratio = (yh - yl) / (xh - xl)

    height = 1600
    width = int(height / aspect_ratio)
    img = of.Image(width, height, layout_xl, layout_yl, layout_xh,
                    layout_yh)
    locations = pos - inst_sizes / 2
    if not target_area_types:
        target_area_types = np.arange(len(area_type_inst_groups))

    # draw layout region
    img.setFillColor(0xFFFFFFFF)
    img.setStrokeColor(25, 25, 25, 0.8)
    img.fillRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)
    img.strokeRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)

    img.setStrokeColor(255, 0, 0, 0.5)
    for box in fence_region_boxes:
        img.setStrokeColor(255, 0, 0, 0.5)
        img.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1])

    for area_type in target_area_types:
        at_inst_ids = area_type_inst_groups[area_type].cpu().numpy()
        if movable_range and movable_range[0] < movable_range[1]:
            inst_ids = at_inst_ids[
                np.where(np.logical_and(at_inst_ids >= movable_range[0], at_inst_ids < movable_range[1]))]
            img.setFillColor(color_map[area_type, 0], color_map[area_type, 1],
                                color_map[area_type, 2], 1)
            img.fillRects(locations[inst_ids], inst_sizes[inst_ids])
        img.setStrokeColor(0, 0, 0, 1)
    img.setStrokeWidth(1)
    in_cnt = 0
    for inst_id in range(movable_range[0], movable_range[1]):
        cr_bbox = fence_region_boxes[movable_inst_to_clock_region[inst_id]]
        xy = pos[inst_id]
        if not (xy[0] >= cr_bbox[0]  and xy[0] < cr_bbox[2] and xy[1] >= cr_bbox[1] and xy[1] < cr_bbox[3]):
            # Draw a line from pos to center of new cr_bbox
            img.strokeLine(xy[0], xy[1], (cr_bbox[0] + cr_bbox[2]) / 2, (cr_bbox[1] + cr_bbox[3]) / 2)
        else:
            in_cnt += 1
    if not os.path.exists(os.path.dirname(filename)):
        os.system("mkdir -p %s" % (os.path.dirname(filename)))
    img.end()
    img.write(filename)  # Output to file


def draw_fence_regions(
        pos: np.ndarray,
        inst_sizes: np.ndarray,
        die_area,
        movable_range,
        fixed_range,
        inst_to_regions: np.ndarray,
        fence_region_boxes,
        filename,
        target_fence_regions_indexes=None,
        iteration=None):
    """

    :param pos: Center positions of instances
    :param inst_sizes: Tensor of instance sizes, shape of (#instances, 2)
    :param die_area: Coordinates of the die area, (xl, yl, xh, yh)
    :param movable_range: range of movable instances
    :param fixed_range: range of fixed instances
    :param inst_to_regions: instance to fence regions indexes
    :param fence_region_boxes: coordinates of fence region boxes, shape of (#fence regions, 4)
    :param filename: output file name
    :param target_fence_regions_indexes: fence region indexes that will be dumped to image files
    :param iteration: current iteration numbers
    """

    layout_xl = xl = die_area[0]
    layout_yl = yl = die_area[1]
    layout_xh = xh = die_area[2]
    layout_yh = yh = die_area[3]

    tt = time.time()
    aspect_ratio = (yh - yl) / (xh - xl)
    height = 800
    width = int(height / aspect_ratio)

    # Convert to lower left
    locations = pos - inst_sizes / 2

    if target_fence_regions_indexes is None:
        target_fence_regions_indexes = np.arange(len(fence_region_boxes))

    for fence_region_idx in target_fence_regions_indexes:
        img = of.Image(width, height, layout_xl, layout_yl, layout_xh, layout_yh)

        # Draw layout region
        img.setFillColor(0xFFFFFFFF)
        img.setStrokeColor(25, 25, 25, 0.8)
        img.fillRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)
        img.strokeRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)

        fence_region_xl = fence_region_boxes[fence_region_idx][0]
        fence_region_yl = fence_region_boxes[fence_region_idx][1]
        fence_region_xh = fence_region_boxes[fence_region_idx][2]
        fence_region_yh = fence_region_boxes[fence_region_idx][3]

        # Draw fence regions
        img.setStrokeColor(0, 0, 0, 0.3)
        img.strokeRect(fence_region_xl, fence_region_yl,
                       fence_region_xh - fence_region_xl,
                       fence_region_yh - fence_region_yl)

        inst_ids = np.squeeze((inst_to_regions == fence_region_idx).nonzero())

        # Draw movable instances
        img.setFillColor(255, 0, 0, 0.5)
        movable_inst_ids = inst_ids[
                np.where(np.logical_and(inst_ids >= movable_range[0], inst_ids < movable_range[1]))]
        img.fillRects(locations[movable_inst_ids], inst_sizes[movable_inst_ids])

        # Draw fixed Instances
        img.setFillColor(0, 255, 0, 0.5)
        fixed_inst_ids = inst_ids[
                np.where(np.logical_and(inst_ids >= fixed_range[0], inst_ids < fixed_range[1]))]
        img.fillRects(locations[fixed_inst_ids], inst_sizes[fixed_inst_ids])

        # Show iteration
        if iteration:
            img.setFillColor(0, 0, 0, 1)
            img.text((xl + xh) / 2, (yl + yh) / 2, '{:04}'.format(iteration), 32)

        # Dump to image file
        if not os.path.exists(os.path.dirname(filename)):
            os.system("mkdir -p %s" % (os.path.dirname(filename)))

        draw_filename = filename.replace(".bmp", "_fr{:04}.bmp".format(fence_region_idx))
        img.end()
        img.write(draw_filename)

    logging.info("plotting to %s takes %.3f seconds" %
                 (filename + "(fence_region)", time.time() - tt))
