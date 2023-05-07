#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : params.py
# Author            : Yibo Lin <yibolin@pku.edu.cn>
# Date              : 04.21.2020
# Last Modified Date: 04.21.2020
# Last Modified By  : Yibo Lin <yibolin@pku.edu.cn>
# TODO: this file is copied from openparf/
# Modify the build system for a better way
import os
import sys
import json
import math
import copy
import numexpr
from collections import OrderedDict
import pdb


class Params:
    """
    @brief Parameter class
    """
    def __init__(self):
        """
        @brief initialization
        """
        filename = os.path.join(os.path.dirname(__file__), 'params.json')
        self.__dict__ = {}
        params_dict = {}
        with open(filename, "r") as f:
            params_dict = json.load(f, object_pairs_hook=OrderedDict)
        for key, value in params_dict.items():
            if 'default' in value:
                self.__dict__[key] = value['default']
            else:
                self.__dict__[key] = None
        self.__dict__['params_dict'] = params_dict

    def printWelcome(self):
        """
        @brief print welcome message
        """
        content = """\
========================================================
                       OpenPARF
            Yibo Lin (http://yibolin.com)
========================================================"""
        print(content)

    def printHelp(self):
        """
        @brief print help message for JSON parameters
        """
        content = self.toMarkdownTable()
        print(content)

    def toMarkdownTable(self):
        """
        @brief convert to markdown table
        """
        key_length = len('JSON Parameter')
        key_length_map = []
        default_length = len('Default')
        default_length_map = []
        description_length = len('Description')
        description_length_map = []

        def getDefaultColumn(key, value):
            if sys.version_info.major < 3:  # python 2
                flag = isinstance(value['default'], unicode)
            else:  #python 3
                flag = isinstance(value['default'], str)
            if flag and not value['default'] and 'required' in value:
                return value['required']
            else:
                return value['default']

        for key, value in self.params_dict.items():
            key_length_map.append(len(key))
            default_length_map.append(len(str(getDefaultColumn(key, value))))
            description_length_map.append(len(value['description']))
            key_length = max(key_length, key_length_map[-1])
            default_length = max(default_length, default_length_map[-1])
            description_length = max(description_length,
                                     description_length_map[-1])

        content = "| %s %s| %s %s| %s %s|\n" % (
            'JSON Parameter', " " *
            (key_length - len('JSON Parameter') + 1), 'Default', " " *
            (default_length - len('Default') + 1), 'Description', " " *
            (description_length - len('Description') + 1))
        content += "| %s | %s | %s |\n" % ("-" * (key_length + 1), "-" *
                                           (default_length + 1), "-" *
                                           (description_length + 1))
        count = 0
        for key, value in self.params_dict.items():
            content += "| %s %s| %s %s| %s %s|\n" % (
                key, " " * (key_length - key_length_map[count] + 1),
                str(getDefaultColumn(key, value)), " " *
                (default_length - default_length_map[count] + 1),
                value['description'], " " *
                (description_length - description_length_map[count] + 1))
            count += 1
        return content

    def toJson(self):
        """
        @brief convert to json
        """
        data = {}
        for key, value in self.__dict__.items():
            if key != 'params_dict':
                data[key] = value
        return data

    def fromJson(self, data):
        """
        @brief load form json
        """
        for key, value in data.items():
            self.__dict__[key] = value
        self.evaluate()

    def dump(self, filename):
        """
        @brief dump to json file
        """
        with open(filename, 'w') as f:
            json.dump(self.toJson(), f)

    def load(self, filename):
        """
        @brief load from json file
        """
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))

    def __str__(self):
        """
        @brief string
        """
        return str(self.toJson())

    def __repr__(self):
        """
        @brief print
        """
        return self.__str__()

    def design_name(self):
        """
        @brief speculate the design name for dumping out intermediate solutions
        """
        if self.aux_input:
            design_name = os.path.basename(self.aux_input).replace(
                ".aux", "").replace(".AUX", "")
        elif self.verilog_input:
            design_name = os.path.basename(self.verilog_input).replace(
                ".v", "").replace(".V", "")
        return design_name

    def solution_file_suffix(self):
        """
        @brief speculate placement solution file suffix
        """
        return "pl"

    def evaluate(self):
        """
        @brief evaluate math expressions in JSON
        """
        if 'gp_model2area_types_map' in self.__dict__:
            m = self.__dict__['gp_model2area_types_map']
            clean_map = {}
            for key, value_map in m.items():
                clean_value_map = {}
                for value_key in value_map.keys():
                    value = value_map[value_key]
                    if value_key.startswith('is'):
                        clean_value_map[value_key] = value
                    else:
                        value_list = []
                        if isinstance(value, list):
                            for vv in value:
                                if isinstance(vv, str):
                                    value_list.append(numexpr.evaluate(vv).item())
                                else:
                                    value_list.append(vv)
                        elif isinstance(value, str):
                            value_list.append(numexpr.evaluate(value).item())
                            value_list.append(numexpr.evaluate(value).item())
                        else:
                            value_list.append(value)
                            value_list.append(value)
                        clean_value_map[value_key] = value_list
                if isinstance(key, list):
                    for kk in key:
                        clean_map[kk] = copy.deepcopy(clean_value_map)
                else:
                    clean_map[key] = clean_value_map
            self.__dict__['gp_model2area_types_map'] = clean_map
        if 'gp_resource2area_types_map' in self.__dict__:
            m = self.__dict__['gp_resource2area_types_map']
            clean_map = {}
            for key, value_list in m.items():
                if not isinstance(value_list, list):
                    clean_map[key] = [value_list]
                else:
                    clean_map[key] = value_list
            self.__dict__['gp_resource2area_types_map'] = clean_map
