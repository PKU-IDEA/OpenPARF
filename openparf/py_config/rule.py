#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : rule.py
# Author            : Jing Mai <magic3007@pku.edu.cn>
# Date              : 10.20.2020
# Last Modified Date: 10.20.2020
# Last Modified By  : Jing Mai <magic3007@pku.edu.cn>
def is_single_site_single_resource_type(site_type, db):
    """ Check if a site type is `SSSRT`(single site single resource type), which means this site type only consists of
    exactly one resource type. Note site "IO" is excluded.

    :param site_type: site type
    :return: None if this site type is not `SSSRT`, otherwise return resource type index.
    """
    resource_id = None
    for rid in range(site_type.numResources()):
        if site_type.resourceCapacity(rid) > 0:
            resource_name = db.layout().resourceMap().resource(rid).name()
            if resource_name == "IO":
                return None
            if resource_id is not None:
                return None
            resource_id = rid
    return resource_id
