# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

def _init():
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    _global_dict[key] = value

def get_value(key, defvalue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defvalue