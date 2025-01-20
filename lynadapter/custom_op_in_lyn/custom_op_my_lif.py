# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import lyngor as lyn



def get_outshape_my_permute(inputs, attrs):
    x_shape = inputs[0].shape
    return [x_shape]


def reset_with_decay(inputs, input_types):
    # varZero = lyn.const(0, dtype)
    input_tensor = inputs[0]
    theta = inputs[1]
    v_0 = inputs[2]
    alpha = inputs[3]
    beta = inputs[4]
    params = [theta, v_0, alpha, beta]
    hwType = "apu"

    from tvm import relay
    if isinstance(input_tensor, (relay.expr.Call)):
        return lyn.apu.pfit(
            data=input_tensor,
            th0=-100.,  # 0x8000,
            th1=theta,
            cfg=12,  # 1100

            const0=beta,
            const1=alpha,
            const2=0,
            const3=0,
            else0=v_0 * alpha + beta,
            else1=0
        )
    else:
        raise NotImplementedError
    # return lyn.apu.pfit(
    #      data = input_tensor,
    #      th0 =theta,
    #      th1 = 0x7c00,
    #      cfg = 8,

    #      const0 =v_0,
    #      const1 =0,
    #      const2 = 0,
    #      const3 =  0,
    #      else0 =beta,
    #      else1 = alpha
    # )

    # return lyn.apu.pfit(
    #         data = input_tensor,
    #         th0 = 0xfc00,
    #         th1 = theta,
    #         cfg = 12, # 1100

    #         const0 = beta,
    #         const1 = alpha,
    #         const2 = 0,
    #         const3 = 0,
    #         else0 = v_0 * alpha + beta,
    #         else1 = 0
    #     )
    # return lyn.apu.custom_op(
    #     inputs=(input_tensor),
    #     params=params,
    #     opName="myCustomOpFit_" + hwType,
    #     soPath="./lib/resetwithdacay.so",
    #     soFuncName="reset_with_decay",
    #     inputNum=1,
    #     outputNum=1,
    #     inputType=("float16",),
    #     outputType=("float16",),  # same as input
    #     outputShapeFunc=get_outshape_my_permute,
    #     hwType=hwType
    # )


lyn.pytorch_convert_map.update({'custom::resetwithdecay': reset_with_decay})


def cmp_and_fire(inputs, input_types):
    # varZero = lyn.const(0, dtype)
    input_tensor = inputs[0]
    theta = inputs[1]
    params = [theta]
    hwType = "apu"
    return lyn.apu.pfit(
        data=input_tensor,
        th0=theta,
        th1=0x7c00,
        cfg=12,

        const0=1,
        const1=0,
        const2=0,
        const3=0,
        else0=0,
        else1=0
    )
    return lyn.apu.custom_op(
        inputs=(input_tensor),
        params=params,
        opName="myCustomOpFire_" + hwType,
        soPath="./lib/cmpandfire.so",
        soFuncName="cmp_and_fire",
        inputNum=1,
        outputNum=1,
        inputType=("float16",),
        outputType=("float16",),  # same as input
        outputShapeFunc=get_outshape_my_permute,
        hwType=hwType
    )


lyn.pytorch_convert_map.update({'custom::cmpandfire': cmp_and_fire})


def load(inputs, input_types):
    '''
    Parameters
    inputs      : Input data
    input_types : Types of input data
    '''
    # check
    assert isinstance(inputs[1], str)
    data = inputs[0]
    flag = inputs[1]
    is_init = inputs[2]
    shape = inputs[3]
    dtype = inputs[4]
    offset = inputs[5]
    step = inputs[6]
    uselookup = inputs[7]
    mode = inputs[8]
    group_id = inputs[9]
    out = lyn.apu.load(data=data,
                       flag=flag,
                       is_initial=is_init,
                       shape=shape,
                       dtype=dtype,
                       offset=offset,
                       step=step,
                       uselookup=uselookup,
                       mode=mode,
                       group_id=group_id)
    return out


# op save definition
def save(inputs, input_types):
    '''
    Parameters
    inputs      : Input data
    input_types : Types of input data
    '''
    # check
    assert isinstance(inputs[1], str)
    data = inputs[0]
    flag = inputs[1]

    if len(inputs) > 2:
        offset = inputs[2]
        step = inputs[3]
        isOutput = inputs[4]
        uselookup = inputs[5]
        mode = inputs[6]
        group_id = inputs[7]
        out = lyn.apu.save(data, flag, offset, step, isOutput, uselookup, mode, group_id)
    else:
        out = lyn.apu.save(data, flag)

    return out


# register operators load save
lyn.pytorch_convert_map.update({'custom::load': load})
lyn.pytorch_convert_map.update({'custom::save': save})
