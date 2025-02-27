# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch
from torch import ops


def load(data, flag, is_init=True, shape=(), dtype='float32', offset=0, step=0, uselookup=False, mode=0, group_id=0,
         init_zero_use_data=None,init_one_use_data=None):
    """
    Parameter Description:
    1)	data is const and serves as the initial value.
    2)	flag is a string that represents a variable.
    3)	isinit indicates whether const has an initial value.
    4)	shape is used to indicate the current shape when load does not have a const initial value.
    5)	dtype is used to indicate the current data type when load does not have a const initial value.
    6)	offset is used to determine the starting offset of the data.
    7)	step determines the position of data jumps.
    8)	uselookup determines whether to use a dynamic DDR base address.
    9)  mode is used to specify the data loading position. 0 represents automatic mode, 
        where the compiler decides whether to use DDR or GLB based on requirements. 
        1 represents GLB buffer loading, and 2 represents DDR loading.
    10) Specify the group ID, which is by default set to 0.
    """
    if init_zero_use_data is not None:
        data=torch.zeros_like(init_zero_use_data)
    if init_one_use_data is not None:
        data=torch.ones_like(init_one_use_data)
    if len(data.shape) == 4:
        data = data.permute(0, 2, 3, 1)
    out = ops.custom.load(data, flag, is_init, shape, dtype, offset, step, uselookup, mode, group_id)
    if len(out.shape) == 4:
        out = out.permute(0, 3, 1, 2)

    return out


def save(dataConst, flag, offset=0, step=0, isOutput=False, uselookup=False, mode=0, group_id=0):
    """
    Parameter Description:
    1)	Input dataConst can be const or an input variable.
    2)	The parameter is a string representing the variable.
    3)	offset is used to determine the starting offset of the data.
    4)	step determines the position where the data jumps.
    5)	isOutput specifies whether the current save is an output.
    6)	uselookup determines whether to use a dynamic DDR base address.
    7)	mode specifies the data loading position. 0 represents automatic mode, 
        where the compiler decides whether to use DDR or GLB based on requirements; 
        1 represents GLB buffer loading, and 2 represents DDR loading.
    8)	Specify the group ID, default is 0.
    """
    if len(dataConst.shape) == 4:
        dataConst = dataConst.permute(0, 2, 3, 1)
    out = ops.custom.save(dataConst, flag, offset, step, isOutput, uselookup, mode, group_id)
    if len(out.shape) == 4:
        out = out.permute(0, 3, 1, 2)

    return out


def load_kernel(data, flag, is_init=True, shape=(), dtype='float32', offset=0, step=0, uselookup=False, mode=0,
                group_id=0,init_zero_use_data=None,init_one_use_data=None):
    """
    Parameter Description:
    1)	data is const and serves as the initial value.
    2)	flag is a string that represents a variable.
    3)	isinit indicates whether const has an initial value.
    4)	shape is used to indicate the current shape when load does not have a const initial value.
    5)	dtype is used to indicate the current data type when load does not have a const initial value.
    6)	offset is used to determine the starting offset of the data.
    7)	step determines the position of data jumps.
    8)	uselookup determines whether to use a dynamic DDR base address.
    9)  mode is used to specify the data loading position. 0 represents automatic mode, 
        where the compiler decides whether to use DDR or GLB based on requirements. 
        1 represents GLB buffer loading, and 2 represents DDR loading.
    10) Specify the group ID, which is by default set to 0.
    """
    if init_zero_use_data is not None:
        data=torch.zeros_like(init_zero_use_data)
    if init_one_use_data is not None:
        data=torch.ones_like(init_one_use_data)
    out = ops.custom.load(data, flag, is_init, shape, dtype, offset, step, uselookup, mode, group_id)

    return out


def save_kernel(dataConst, flag, offset=0, step=0, isOutput=False, uselookup=False, mode=0, group_id=0):
    """
    Parameter Description:
    1)	Input dataConst can be const or an input variable.
    2)	The parameter is a string representing the variable.
    3)	offset is used to determine the starting offset of the data.
    4)	step determines the position where the data jumps.
    5)	isOutput specifies whether the current save is an output.
    6)	uselookup determines whether to use a dynamic DDR base address.
    7)	mode specifies the data loading position. 0 represents automatic mode, 
        where the compiler decides whether to use DDR or GLB based on requirements; 
        1 represents GLB buffer loading, and 2 represents DDR loading.
    8)	Specify the group ID, default is 0.
    """
    out = ops.custom.save(dataConst, flag, offset, step, isOutput, uselookup, mode, group_id)

    return out

def rand(data, oshape=(), uInitValue=0, mode=0):   
    uInitValue = 0x4000
    if mode == "HW_RANDOM":
        mode = 0
    elif mode == "FOLLOW_SEED":
        mode = 1
    output = ops.custom.Rand(data, oshape, uInitValue, mode)
    return output
