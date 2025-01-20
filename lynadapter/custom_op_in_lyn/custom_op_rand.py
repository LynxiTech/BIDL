# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


import lyngor as lyn
def Rand(inputs, input_types):
    '''
    Parameters
    inputs      : Input data
    input_types : Types of input data
    '''
    return lyn.apu.random_data(inputs[0], inputs[1].type_annotation.shape, inputs[2], inputs[3])
lyn.pytorch_convert_map.update({'custom::Rand': Rand})