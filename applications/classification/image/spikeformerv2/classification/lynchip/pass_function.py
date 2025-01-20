import lyngor
import tvm.apu_api
import tvm.relay as relay
from tvm.relay import *
import tvm.relay
import tvm.relay
import tvm.relay
import tvm.relay
import tvm.relay
from tvm.relay.dataflow_pattern import is_op, is_tuple, wildcard
from tvm.relay.transform.transform import function_pass
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.transform import *


def skip_generic_op(expr):
    skip_op = ["reshape", "transpose", "cast", "layout_transform", "squeeze"]
    if isinstance(expr, tvm.relay.expr.Call) and expr.op.name in skip_op:
        return skip_generic_op(expr.args[0])

    return expr

def get_args1_end(expr):
    op_args = []
    assert isinstance(expr, tvm.relay.expr.Call)
    for elem in range(1, len(expr.args)):
        op_args.append(expr.args[elem])
    return op_args

def get_zeros(expr):
    shape = relay.frontend.common.infer_shape(expr)
    zeros = np.zeros(shape[-1], 'float32')
    return relay.const(zeros, zeros.dtype)

def get_split_const(expr, split, axis=0):
    constData = expr.data.asnumpy()
    data_list = np.split(constData, split, axis=axis)

    return data_list

def copy_const(expr):
    constData = expr.data.asnumpy()
    return relay.const(constData, constData.dtype)

def copy_conv(expr,weight,conv):
    conv2d = relay.nn.conv2d(expr, weight,
                strides=conv.attrs['strides'],
                padding=conv.attrs['padding'],
                dilation=conv.attrs['dilation'],
                groups=conv.attrs['groups'],
                channels=conv.attrs['channels'],
                kernel_size=conv.attrs['kernel_size'],
                data_layout=conv.attrs['data_layout'],
                kernel_layout=conv.attrs['kernel_layout'],
                out_layout=conv.attrs['out_layout'],
                out_dtype=conv.attrs['out_dtype'],
                is_depthwise=conv.attrs['is_depthwise'])

    return conv2d

def infer_shape(expr):
    return relay.frontend.common.infer_shape(expr)


def getBeginLif(expr):
    flag = False
    mulCall = None
    if isinstance(expr, tvm.relay.expr.Call) and expr.op.name == 'cast':
        ge = expr.args[0]
        if isinstance(ge, tvm.relay.expr.Call) and ge.op.name == 'greater_equal':
            add = ge.args[0]
            if isinstance(add, tvm.relay.expr.Call) and add.op.name == 'add':
                mul0 = add.args[0]
                if isinstance(mul0, tvm.relay.expr.Call) and mul0.op.name == 'multiply':
                    flag = True
                    mulCall = mul0
    return flag, mulCall


def getEndLif(expr):
    flag = False
    subCall = None
    if isinstance(expr, tvm.relay.expr.Call) and expr.op.name == 'cast':
        ge = expr.args[0]
        if isinstance(ge, tvm.relay.expr.Call) and ge.op.name == 'greater_equal':
            add0 = ge.args[0]
            if isinstance(add0, tvm.relay.expr.Call) and add0.op.name == 'add':
                add1 = add0.args[0]
                if isinstance(add1, tvm.relay.expr.Call) and add1.op.name == 'add':
                    mul0 = skip_generic_op(add1.args[1])
                    if isinstance(mul0, tvm.relay.expr.Call) and mul0.op.name == 'multiply':
                        sub = mul0.args[0]
                        if isinstance(sub, tvm.relay.expr.Call) and sub.op.name == 'subtract':
                            flag = True
                            subCall = sub
    return flag, subCall


def getMidLif(expr):
    flag = False
    subCall = None
    if isinstance(expr, tvm.relay.expr.Call) and expr.op.name == 'cast':
        ge = expr.args[0]
        if isinstance(ge, tvm.relay.expr.Call) and ge.op.name == 'greater_equal':
            add0 = ge.args[0]
            if isinstance(add0, tvm.relay.expr.Call) and add0.op.name == 'add':
                add1 = add0.args[0]
                if isinstance(add1, tvm.relay.expr.Call) and add1.op.name == 'add':
                    mul0 = skip_generic_op(add1.args[1])
                    if isinstance(mul0, tvm.relay.expr.Call) and mul0.op.name == 'multiply':
                        sub = mul0.args[0]
                        if isinstance(sub, tvm.relay.expr.Call) and sub.op.name == 'subtract':
                            flag = True
                            subCall = sub
    return flag, subCall

def getMultiLif(expr):
    if isinstance(expr, tvm.relay.expr.Call) and expr.op.name == 'concatenate':
        tup0 = expr.args[0]
        if isinstance(tup0, tvm.relay.Tuple):
            timeSteps = len(tup0)
            inputCalls = []
            for i in range(timeSteps):
                if i == 0:
                    flag, multCall = getBeginLif(tup0[i])
                    if flag is False:
                        return False, []
                    inputCalls.append(multCall)
                elif i == timeSteps - 1:
                    flag, subCall = getEndLif(tup0[i])
                    if flag is False:
                        return False, []
                    inputCalls.append(subCall)
                else:
                    flag, subCall = getMidLif(tup0[i])
                    if flag is False:
                        return False, []
                    inputCalls.append(subCall)

            return True, inputCalls
    return False, []


def reMultiLif(inputs, times):
    v = 0
    tau = 2.0
    tau = 1 / tau
    spike_list = []
    # lif
    for i in range(times):
        x = inputs[i]
        if i == 0:
            v = relay.multiply(x, relay.const(tau, 'float32'))
        else:
            add = relay.add(x,v)
            v = relay.multiply(add, relay.const(tau, 'float32'))
        spike_d = relay.greater_equal(v, relay.const(1, 'float32'))
        spike_d = relay.cast(spike_d, 'float32')
        spike_t = relay.less_equal(v, relay.const(1, 'float32'))
        spike_t = relay.cast(spike_t, 'float32')
        v = relay.multiply(spike_t, v)
        spike_list.append(spike_d)
    return spike_list


@function_pass(opt_level=1, required=["InferType"])
class optSpikeTransformer:
    def __init__(self):
        pass
    def transform_function(self, func, mod, ctx):
        class reMSConvBlock0(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    add0 = call.args[0]
                    add1 = call.args[1]
                    if isinstance(add0, tvm.relay.expr.Call) and add0.op.name == 'add' \
                    and isinstance(add1, tvm.relay.expr.Call) and add1.op.name == 'add':
                        conv3 = add0.args[0]
                        bias3 = add0.args[1]
                        if isinstance(conv3, tvm.relay.expr.Call) and conv3.op.name == 'nn.conv2d':
                            conv2 = conv3.args[0]
                            weight3 = conv3.args[1]
                            if isinstance(conv2, tvm.relay.expr.Call) and conv2.op.name == 'nn.conv2d':
                                lif1 = conv2.args[0]
                                weight2 = conv2.args[1]
                                flag, inputCalls1 = getMultiLif(lif1)
                                if flag is True:
                                    split0 = inputCalls1[0].args[0].tuple_value
                                    add2 = split0.args[0]
                                    if isinstance(add2, tvm.relay.expr.Call) and add2.op.name == 'add':
                                        conv1 = add2.args[0]
                                        bias1 = add2.args[1]
                                        if isinstance(conv1, tvm.relay.expr.Call) and conv1.op.name == 'nn.conv2d':
                                            lif0 = conv1.args[0]
                                            weight1 = conv1.args[1]
                                            flag, inputCalls0 = getMultiLif(lif0)
                                            if flag:
                                                split1 = inputCalls0[0].args[0].tuple_value
                                                add3 = split1.args[0]
                                                if isinstance(add3, tvm.relay.expr.Call) and add3.op.name == 'add' and add3 == add1:
                                                    conv0 = add3.args[0]
                                                    bias0 = add3.args[1]
                                                    if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                                                        layTrans = conv0.args[0]
                                                        weight0 = conv0.args[1]
                                                        if isinstance(layTrans, tvm.relay.expr.Call) and layTrans.op.name == 'layout_transform':
                                                            tile = layTrans.args[0]
                                                            if isinstance(tile, tvm.relay.expr.Call) and tile.op.name == 'tile':
                                                                data = tile.args[0]
                                                                times = int(relay.frontend.common.infer_shape(tile)[0])
                                                                data = relay.layout_transform(data, src_layout='NCHW', dst_layout='NHWC')
                                                                temp0 = []
                                                                for i in range(times):
                                                                    conv = copy_conv(data,copy_const(weight0),conv0)
                                                                    add = relay.add(conv, copy_const(bias0))
                                                                    temp0.append(add)

                                                                spike_list = reMultiLif(temp0, times)

                                                                temp1 = []
                                                                for i in range(times):
                                                                    cur = spike_list[i]
                                                                    conv = copy_conv(cur,copy_const(weight1),conv1)
                                                                    add = relay.add(conv, copy_const(bias1))
                                                                    temp1.append(add)

                                                                spike_list = reMultiLif(temp1, times)

                                                                temp2 = []
                                                                for i in range(times):
                                                                    cur = spike_list[i]
                                                                    conv = copy_conv(cur,copy_const(weight2),conv2)
                                                                    conv = copy_conv(conv,copy_const(weight3),conv3)
                                                                    add0 = relay.add(conv, copy_const(bias3))
                                                                    add1 = relay.add(temp0[i], add0)
                                                                    temp2.append(add1)
                                                                out = relay.concatenate(temp2, 0)
                                                                return self.visit(out)
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)

        class reMSConvBlock1(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    add0 = call.args[1]
                    concat0 = call.args[0]
                    if isinstance(add0, tvm.relay.expr.Call) and add0.op.name == 'add' \
                    and isinstance(concat0, tvm.relay.expr.Call) and concat0.op.name == 'concatenate':
                        conv1 = add0.args[0]
                        bias1 = add0.args[1]
                        if isinstance(conv1, tvm.relay.expr.Call) and conv1.op.name == 'nn.conv2d':
                            lif1 = conv1.args[0]
                            weight1 = conv1.args[1]
                            flag, inputCalls1 = getMultiLif(lif1)
                            if flag:
                                split0 = inputCalls1[0].args[0].tuple_value
                                add1 = split0.args[0]
                                if isinstance(add1, tvm.relay.expr.Call) and add1.op.name == 'add':
                                    conv0 = add1.args[0]
                                    bias0 = add1.args[1]
                                    if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                                        lif0 = conv0.args[0]
                                        weight0 = conv0.args[1]
                                        flag, inputCalls0 = getMultiLif(lif0)
                                        if flag:
                                            split1 = inputCalls0[0].args[0].tuple_value
                                            concat1 = split1.args[0]
                                            if isinstance(concat1, tvm.relay.expr.Call) and concat1.op.name == 'concatenate' and concat0 == concat1:
                                                print('reMSConvBlock1--')
                                                temp0 = concat0.args[0]
                                                times = int(relay.frontend.common.infer_shape(concat0)[0])
                                                spike_list = reMultiLif(temp0, times)
                                                temp1 = []
                                                for i in range(times):
                                                    cur = spike_list[i]
                                                    conv = copy_conv(cur, copy_const(weight0), conv0)
                                                    add = relay.add(conv, copy_const(bias0))
                                                    temp1.append(add)

                                                spike_list = reMultiLif(temp1, times)
                                                temp2 = []
                                                for i in range(times):
                                                    cur = spike_list[i]
                                                    conv = copy_conv(cur, copy_const(weight1), conv1)
                                                    add = relay.add(conv, copy_const(bias1))
                                                    out = relay.add(temp0[i], add)
                                                    temp2.append(out)
                                                out = relay.concatenate(temp2, 0)
                                                return self.visit(out)
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)

        class reMSConvBlock2(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    conv0 = call.args[0]
                    bias0 = call.args[1]
                    if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                        lif0 = conv0.args[0]
                        weight0 = conv0.args[1]
                        flag, inputCalls0 = getMultiLif(lif0)
                        if flag:
                            split0 = inputCalls0[0].args[0].tuple_value
                            concat0 = split0.args[0]
                            if isinstance(concat0, tvm.relay.expr.Call) and concat0.op.name == 'concatenate':
                                temp0 = concat0.args[0]
                                times = int(relay.frontend.common.infer_shape(concat0)[0])
                                spike_list = reMultiLif(temp0, times)

                                temp1 = []
                                for i in range(times):
                                    cur = spike_list[i]
                                    conv = copy_conv(cur, copy_const(weight0), conv0)
                                    add = relay.add(conv, copy_const(bias0))
                                    temp1.append(add)
                                out = relay.concatenate(temp1, 0)
                                return self.visit(out)
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)

        class reMSConvBlock3(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    add0 = call.args[0]
                    concat0 = call.args[1]
                    if isinstance(add0, tvm.relay.expr.Call) and add0.op.name == 'add' \
                    and isinstance(concat0, tvm.relay.expr.Call) and concat0.op.name == 'concatenate':
                        conv2 = add0.args[0]
                        bias2 = add0.args[1]
                        if isinstance(conv2, tvm.relay.expr.Call) and conv2.op.name == 'nn.conv2d':
                            conv1 = conv2.args[0]
                            weight2 = conv2.args[1]
                            if isinstance(conv1, tvm.relay.expr.Call) and conv1.op.name == 'nn.conv2d':
                                lif1 = conv1.args[0]
                                weight1 = conv1.args[1]
                                flag, inputCalls1 = getMultiLif(lif1)
                                if flag:
                                    split0 = inputCalls1[0].args[0].tuple_value
                                    add2 = split0.args[0]
                                    if isinstance(add2, tvm.relay.expr.Call) and add2.op.name == 'add':
                                        conv0 = add2.args[0]
                                        bias0 = add2.args[1]
                                        if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                                            lif0 = conv0.args[0]
                                            weight0 = conv0.args[1]
                                            flag, inputCalls0 = getMultiLif(lif0)
                                            if flag:
                                                split1 = inputCalls0[0].args[0].tuple_value
                                                concat1 = split1.args[0]
                                                if isinstance(concat1, tvm.relay.expr.Call) and concat1.op.name == 'concatenate' and concat1 == concat0:
                                                    temp0 = concat1.args[0]
                                                    times = int(relay.frontend.common.infer_shape(concat1)[0])

                                                    spike_list = reMultiLif(temp0, times)

                                                    temp1 = []
                                                    for i in range(times):
                                                        cur = spike_list[i]
                                                        conv = copy_conv(cur, copy_const(weight0), conv0)
                                                        add = relay.add(conv, copy_const(bias0))
                                                        temp1.append(add)

                                                    spike_list = reMultiLif(temp1, times)

                                                    temp2 = []
                                                    for i in range(times):
                                                        cur = spike_list[i]
                                                        conv = copy_conv(cur, copy_const(weight1), conv1)
                                                        conv = copy_conv(conv, copy_const(weight2), conv2)
                                                        add0 = relay.add(conv, bias2)
                                                        add1 = relay.add(temp0[i], add0)
                                                        temp2.append(add1)

                                                    out = relay.concatenate(temp2, 0)
                                                    return self.visit(out)
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)


        class reTransformer(relay.dataflow_pattern.DFPatternCallback):
            def __init__(self):
                super(reTransformer, self).__init__()
                # Data
                self.q = wildcard()  # T, 14, 14, 512
                self.k = wildcard()  # T, 14, 14, 512
                self.v = wildcard()  # T, 14, 14, 512

                self.scale = is_constant()

                # for q
                self.layTrans_q = is_op('layout_transform')(self.q)
                self.reshape_q = is_op('reshape')(self.layTrans_q)
                self.trans_q = is_op('transpose')(self.reshape_q)
                self.reshape_q = is_op('reshape')(self.trans_q)
                self.q_out = is_op('transpose')(self.reshape_q)

                # for k
                self.layTrans_k = is_op('layout_transform')(self.k)
                self.reshape_k = is_op('reshape')(self.layTrans_k)
                self.trans_k = is_op('transpose')(self.reshape_k)
                self.reshape_k = is_op('reshape')(self.trans_k)
                self.k_out = is_op('transpose')(self.reshape_k)

                # for v
                self.layTrans_v = is_op('layout_transform')(self.v)
                self.reshape_v = is_op('reshape')(self.layTrans_v)
                self.trans_v = is_op('transpose')(self.reshape_v)
                self.reshape_v = is_op('reshape')(self.trans_v)
                self.v_out = is_op('transpose')(self.reshape_v)

                self.bmm = is_op('nn.batch_matmul')(self.k_out, self.v_out)
                self.bmm_t = is_op('transpose')(self.bmm)

                self.qkv = is_op('nn.batch_matmul')(self.q_out, self.bmm_t)
                self.mul = is_op('multiply')(self.qkv, self.scale)

                self.out = is_op('transpose')(self.mul)

                self.gout = self.out
                self.pattern=self.gout

            def callback(self, pre, post, node_map):
                input_q = node_map[self.q][0]
                input_k = node_map[self.k][0]
                input_v = node_map[self.v][0]
                scale = node_map[self.scale][0]

                inputqOutShape = relay.frontend.common.infer_shape(node_map[self.q_out][0])
                inputkOutShape = relay.frontend.common.infer_shape(node_map[self.k_out][0])
                inputvOutShape = relay.frontend.common.infer_shape(node_map[self.v_out][0])

                input_q = relay.reshape(input_q, (inputqOutShape[0],inputqOutShape[2],inputqOutShape[1],inputqOutShape[3])) # b,196,8,64
                input_k = relay.reshape(input_k, (inputkOutShape[0],inputkOutShape[3],inputkOutShape[1],inputkOutShape[2])) # b,196,8,64
                input_v = relay.reshape(input_v, (inputvOutShape[0],inputvOutShape[3],inputvOutShape[1],inputvOutShape[2])) # b,196,8,64

                input_q = relay.transpose(input_q, [0,2,1,3]) # b,8,196,64
                input_k = relay.transpose(input_k, [0,2,3,1]) # b,8,64,196
                input_v = relay.transpose(input_v, [0,2,3,1]) # b,8,64,196

                bmm = relay.nn.batch_matmul(input_v,input_k) # b,8,64,64

                qkv = relay.nn.batch_matmul(bmm,input_q) # b,8,64,196

                scale = scale.data.asnumpy()[0] # 0.125
                new_scale = np.repeat(scale, inputkOutShape[3]) # 196

                out = relay.multiply(qkv, relay.const(new_scale, new_scale.dtype))

                return out

        class reBackBone(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    conv2 = call.args[0]
                    bias2 = call.args[1]
                    if isinstance(conv2, tvm.relay.expr.Call) and conv2.op.name == 'nn.conv2d':
                        lif2 = conv2.args[0]
                        weight2 = conv2.args[1]

                        flag, inputCalls2 = getMultiLif(lif2)

                        if flag:
                            split2 = inputCalls2[0].args[0].tuple_value
                            add0 = split2.args[0]
                            if isinstance(add0, tvm.relay.expr.Call) and add0.op.name == 'add':
                                layTrans0 = add0.args[1]
                                res = add0.args[0]
                                if isinstance(layTrans0, tvm.relay.expr.Call) and layTrans0.op.name == 'layout_transform' \
                                and isinstance(res, tvm.relay.expr.Call) and res.op.name == 'add':
                                    lyn_broad_mac = skip_generic_op(layTrans0.args[0])
                                    if isinstance(lyn_broad_mac, tvm.relay.expr.Call) and lyn_broad_mac.op.name == 'lyn_broad_mac':
                                        add1 = skip_generic_op(lyn_broad_mac.args[0])
                                        if isinstance(add1, tvm.relay.expr.Call) and add1.op.name == 'add':
                                            conv1 = add1.args[0]
                                            bias1 = add1.args[1]
                                            if isinstance(conv1, tvm.relay.expr.Call) and conv1.op.name == 'nn.conv2d':
                                                lif1 = skip_generic_op(conv1.args[0])
                                                weight1 = conv1.args[1]

                                                flag, inputCalls1 = getMultiLif(lif1)

                                                if flag:
                                                    split1 = inputCalls1[0].args[0].tuple_value
                                                    add2 = split1.args[0]
                                                    if isinstance(add2, tvm.relay.expr.Call) and add2.op.name == 'add':
                                                        scale1 = add2.args[1]
                                                        mul = add2.args[0]
                                                        if isinstance(mul, tvm.relay.expr.Call) and mul.op.name == 'multiply':
                                                            scale0 = mul.args[1]
                                                            add3 = skip_generic_op(mul.args[0])
                                                            if isinstance(add3, tvm.relay.expr.Call) and add3.op.name == 'add':
                                                                conv0 = add3.args[0]
                                                                bias0 = add3.args[1]
                                                                if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                                                                    lif0 = skip_generic_op(conv0.args[0])
                                                                    weight0 = conv0.args[1]

                                                                    flag, inputCalls0 = getMultiLif(lif0)
                                                                    if flag:
                                                                        split0 = inputCalls0[0].args[0].tuple_value
                                                                        add4 = skip_generic_op(split0.args[0])

                                                                        if add4 == res:
                                                                            data = add4
                                                                            times = int(relay.frontend.common.infer_shape(data)[0])
                                                                            input_shape = relay.frontend.common.infer_shape(data)
                                                                            temp0 = relay.split(data, times, 0)

                                                                            spike_list = reMultiLif(temp0, times)

                                                                            weight0 = weight0.data.asnumpy().transpose(3,1,2,0)
                                                                            bias0 = bias0.data.asnumpy()

                                                                            scale0 = scale0.data.asnumpy().reshape(-1)
                                                                            scale1 = scale1.data.asnumpy().reshape(-1)

                                                                            new_weight0 = (scale0*weight0).transpose(3,1,2,0)
                                                                            new_bias0 = scale0*bias0 + scale1

                                                                            temp1 = []
                                                                            for i in range(times):
                                                                                cur = spike_list[i]
                                                                                conv = copy_conv(cur, relay.const(new_weight0,'float32'), conv0)
                                                                                add = relay.add(conv, relay.const(new_bias0, 'float32'))
                                                                                temp1.append(add)

                                                                            spike_list = reMultiLif(temp1, times)

                                                                            temp2 = []
                                                                            scale2 = lyn_broad_mac.args[1].data.asnumpy().transpose(1,0).reshape(1, input_shape[1], input_shape[2], input_shape[3])
                                                                            sclae3 = lyn_broad_mac.args[2].data.asnumpy().transpose(1,0).reshape(1, input_shape[1], input_shape[2], input_shape[3])

                                                                            scale2 = relay.const(scale2, scale2.dtype)
                                                                            sclae3 = relay.const(sclae3, sclae3.dtype)

                                                                            for i in range(times):
                                                                                cur = spike_list[i]
                                                                                conv = copy_conv(cur, copy_const(weight1), conv1)
                                                                                add = relay.add(conv, copy_const(bias1))
                                                                                mul = relay.multiply(add, copy_const(scale2))
                                                                                add = relay.add(mul, copy_const(sclae3))
                                                                                out = relay.add(temp0[i], add)
                                                                                temp2.append(out)

                                                                            spike_list = reMultiLif(temp2, times)

                                                                            temp3 = []
                                                                            for i in range(times):
                                                                                cur = spike_list[i]
                                                                                conv = copy_conv(cur, copy_const(weight2), conv2)
                                                                                add = relay.add(conv, copy_const(bias2))
                                                                                temp3.append(add)
                                                                            out = relay.concatenate(temp3, 0)
                                                                            return self.visit(out)


                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)

        class optBackBone(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    scale1 = call.args[1]
                    mul = call.args[0]
                    if isinstance(mul, tvm.relay.expr.Call) and mul.op.name == 'multiply':
                        scale0 = mul.args[1]
                        add3 = skip_generic_op(mul.args[0])
                        if isinstance(add3, tvm.relay.expr.Call) and add3.op.name == 'add':
                            conv0 = add3.args[0]
                            bias0 = add3.args[1]
                            if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                                layout = conv0.args[0]
                                weight0 = conv0.args[1]
                                if isinstance(layout, tvm.relay.expr.Call) and layout.op.name == 'layout_transform':
                                    weight0 = weight0.data.asnumpy().transpose(3,1,2,0)
                                    bias0 = bias0.data.asnumpy()

                                    scale0 = scale0.data.asnumpy().reshape(-1)
                                    scale1 = scale1.data.asnumpy().reshape(-1)

                                    new_weight0 = (scale0*weight0).transpose(3,1,2,0)
                                    new_bias0 = scale0*bias0 + scale1

                                    curShape = infer_shape(layout)
                                    outShape = infer_shape(call)
                                    cur = relay.reshape(layout,(1, curShape[0]*curShape[1], curShape[2],curShape[3]))

                                    conv = copy_conv(cur, relay.const(new_weight0,'float32'), conv0)
                                    add = relay.add(conv, relay.const(new_bias0, 'float32'))
                                    out = relay.transpose(add, [0,1,3,2])
                                    out = relay.reshape(out, outShape)

                                    return self.visit(out)

                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)


        class reLastBackBone(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'concatenate':
                    lif2 = call
                    flag, inputCalls2 = getMultiLif(lif2)
                    if flag:
                        split2 = inputCalls2[0].args[0].tuple_value
                        reshape = split2.args[0]
                        if isinstance(reshape, tvm.relay.expr.Call) and reshape.op.name == 'reshape' \
                        and isinstance(reshape.args[0], tvm.relay.expr.Call) and reshape.args[0].op.name == 'mean':
                            mean = reshape.args[0]
                            add0 = skip_generic_op(mean.args[0])
                            if isinstance(add0, tvm.relay.expr.Call) and add0.op.name == 'add':
                                layTrans0 = add0.args[1]
                                res = add0.args[0]
                                if isinstance(layTrans0, tvm.relay.expr.Call) and layTrans0.op.name == 'layout_transform' \
                                and isinstance(res, tvm.relay.expr.Call) and res.op.name == 'add':
                                    lyn_broad_mac = skip_generic_op(layTrans0.args[0])
                                    if isinstance(lyn_broad_mac, tvm.relay.expr.Call) and lyn_broad_mac.op.name == 'lyn_broad_mac':
                                        add1 = skip_generic_op(lyn_broad_mac.args[0])
                                        if isinstance(add1, tvm.relay.expr.Call) and add1.op.name == 'add':
                                            conv1 = add1.args[0]
                                            bias1 = add1.args[1]
                                            if isinstance(conv1, tvm.relay.expr.Call) and conv1.op.name == 'nn.conv2d':
                                                lif1 = skip_generic_op(conv1.args[0])
                                                weight1 = conv1.args[1]
                                                flag, inputCalls1 = getMultiLif(lif1)

                                                if flag:
                                                    split1 = inputCalls1[0].args[0].tuple_value
                                                    add2 = split1.args[0]
                                                    if isinstance(add2, tvm.relay.expr.Call) and add2.op.name == 'add':
                                                        scale1 = add2.args[1]
                                                        mul = add2.args[0]
                                                        if isinstance(mul, tvm.relay.expr.Call) and mul.op.name == 'multiply':
                                                            scale0 = mul.args[1]
                                                            add3 = skip_generic_op(mul.args[0])
                                                            if isinstance(add3, tvm.relay.expr.Call) and add3.op.name == 'add':
                                                                conv0 = add3.args[0]
                                                                bias0 = add3.args[1]
                                                                if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                                                                    lif0 = skip_generic_op(conv0.args[0])
                                                                    weight0 = conv0.args[1]

                                                                    flag, inputCalls0 = getMultiLif(lif0)
                                                                    if flag:
                                                                        split0 = inputCalls0[0].args[0].tuple_value
                                                                        add4 = skip_generic_op(split0.args[0])

                                                                        if add4 == res:
                                                                            data = add4
                                                                            times = int(relay.frontend.common.infer_shape(data)[0])
                                                                            input_shape = relay.frontend.common.infer_shape(data)
                                                                            temp0 = relay.split(data, times, 0)

                                                                            spike_list = reMultiLif(temp0,times)

                                                                            weight0 = weight0.data.asnumpy().transpose(3,1,2,0)
                                                                            bias0 = bias0.data.asnumpy()

                                                                            scale0 = scale0.data.asnumpy().reshape(-1)
                                                                            scale1 = scale1.data.asnumpy().reshape(-1)

                                                                            new_weight0 = (scale0*weight0).transpose(3,1,2,0)
                                                                            new_bias0 = scale0*bias0 + scale1

                                                                            temp1 = []
                                                                            for i in range(times):
                                                                                cur = spike_list[i]
                                                                                conv = copy_conv(cur, relay.const(new_weight0,'float32'),conv0)
                                                                                add = relay.add(conv, relay.const(new_bias0, 'float32'))
                                                                                temp1.append(add)

                                                                            spike_list = reMultiLif(temp1,times)

                                                                            temp2 = []
                                                                            scale2 = lyn_broad_mac.args[1].data.asnumpy().transpose(1,0).reshape(1, input_shape[1], input_shape[2], input_shape[3])
                                                                            sclae3 = lyn_broad_mac.args[2].data.asnumpy().transpose(1,0).reshape(1, input_shape[1], input_shape[2], input_shape[3])

                                                                            scale2 = relay.const(scale2, scale2.dtype)
                                                                            sclae3 = relay.const(sclae3, sclae3.dtype)

                                                                            for i in range(times):
                                                                                cur = spike_list[i]
                                                                                conv = copy_conv(cur, copy_const(weight1), conv1)
                                                                                add = relay.add(conv, copy_const(bias1))
                                                                                mul = relay.multiply(add, copy_const(scale2))
                                                                                add = relay.add(mul, copy_const(sclae3))
                                                                                add = relay.add(temp0[i], add)
                                                                                reshape = relay.reshape(add, (1,-1,input_shape[3]))
                                                                                mean = relay.mean(reshape, 1)
                                                                                out = relay.reshape(mean, (1,1,1,input_shape[3]))
                                                                                temp2.append(out)

                                                                            spike_list = reMultiLif(temp2,times)
                                                                            out = relay.concatenate(spike_list, 0)
                                                                            return self.visit(out)


                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)


        class reBuildMultiLif(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'concatenate':
                    lif = call
                    flag, inputCalls2 = getMultiLif(lif)
                    if flag:
                        split = inputCalls2[0].args[0].tuple_value
                        data = split.args[0]
                        times = relay.frontend.common.infer_shape(lif)[0]
                        temp = relay.split(data, times, axis=0)
                        spike_list = reMultiLif(temp,times)
                        out = relay.concatenate(spike_list, 0)
                        return self.visit(out)
                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)

        class optLifConv2d(tvm.relay.ExprMutator):
            def visit_call(self, call):
                if call.op.name == 'add':
                    conv0 = call.args[0]
                    bias0 = call.args[1]
                    if isinstance(conv0, tvm.relay.expr.Call) and conv0.op.name == 'nn.conv2d':
                        concat0 = conv0.args[0]
                        weight0 = conv0.args[1]
                        skip_concat0 = skip_generic_op(conv0.args[0])
                        layout = conv0.args[0]
                        conv1 = conv0.args[0]
                        if isinstance(concat0, tvm.relay.expr.Call) and concat0.op.name == 'concatenate':
                            tup = concat0.args[0]
                            if isinstance(tup, tvm.relay.Tuple):
                                times = int(len(tup))
                                if relay.frontend.common.infer_shape(weight0)[-1] > 512:
                                    # 切 inc
                                    split = 4
                                    weight0List = get_split_const(weight0, split, axis=-1)
                                    outList = []
                                    for i in range(times):
                                        cur = tup[i]
                                        inputList = relay.split(cur, split, axis=-1)
                                        out = None
                                        for j in range(split):
                                            data = inputList[j]
                                            conv = copy_conv(data, relay.const(weight0List[j],'float32'), conv0)
                                            if j == 0:
                                                out = conv
                                            else:
                                                out = relay.add(out, conv)
                                        outList.append(out)
                                    gout = relay.concatenate(outList, 0)
                                    gout = relay.add(gout, bias0)
                                    return self.visit(gout)

                        # if isinstance(skip_concat0, tvm.relay.expr.Call) and skip_concat0.op.name == 'concatenate':
                        #     tup = skip_concat0.args[0]
                        #     if isinstance(tup, tvm.relay.Tuple):
                        #         times = int(len(tup))
                        #         if relay.frontend.common.infer_shape(weight0)[-1] == 2048:
                        #             # 切 inc
                        #             split = 1
                        #             weight0List = get_split_const(weight0, split, axis=-1)
                        #             outList = []
                        #             for i in range(times):
                        #                 cur = tup[i]
                        #                 curShape = infer_shape(cur)
                        #                 cur = relay.reshape(cur, (curShape[0], curShape[1], 1, curShape[2]))
                        #                 cur = relay.layout_transform(cur, src_layout='NCHW', dst_layout='NHWC')
                        #                 inputList = relay.split(cur, split, axis=-1)
                        #                 out = None
                        #                 for j in range(split):
                        #                     data = inputList[j]
                        #                     conv = copy_conv(data, relay.const(weight0List[j],'float32'), conv0)
                        #                     if j == 0:
                        #                         out = conv
                        #                     else:
                        #                         out = relay.add(out, conv)
                        #                 outList.append(out)
                        #             gout = relay.concatenate(outList, 0)
                        #             gout = relay.add(gout, bias0)
                        #             return self.visit(gout)

                        if isinstance(layout, tvm.relay.expr.Call) and layout.op.name == 'layout_transform':
                            if relay.frontend.common.infer_shape(weight0)[-1] == 2048 or relay.frontend.common.infer_shape(weight0)[-1] == 2560:
                                curShape = infer_shape(layout)
                                outShape = infer_shape(call)
                                cur = relay.reshape(layout,(1, curShape[0], curShape[2],curShape[3]))

                                conv = copy_conv(cur, weight0, conv0)
                                out = relay.add(conv, bias0)

                                out = relay.reshape(out, outShape)
                                return self.visit(out)
                            if list(relay.frontend.common.infer_shape(weight0))== [640, 1, 1, 640]:
                                curShape = infer_shape(layout)
                                outShape = infer_shape(call)
                                cur = relay.reshape(layout,(1, curShape[0]*curShape[1], curShape[2],curShape[3]))

                                conv = copy_conv(cur, weight0, conv0)
                                out = relay.add(conv, bias0)

                                out = relay.reshape(out, outShape)
                                return self.visit(out)

                            if list(relay.frontend.common.infer_shape(weight0))== [512, 1, 1, 512]:
                                curShape = infer_shape(layout)
                                outShape = infer_shape(call)
                                cur = relay.reshape(layout,(1, curShape[0]*curShape[1], curShape[2],curShape[3]))

                                conv = copy_conv(cur, weight0, conv0)
                                out = relay.add(conv, bias0)

                                out = relay.reshape(out, outShape)
                                return self.visit(out)

                        if isinstance(conv1, tvm.relay.expr.Call) and conv1.op.name == 'nn.conv2d':
                            if list(relay.frontend.common.infer_shape(weight0)) == [640, 1, 1, 640]:
                                curShape = infer_shape(conv1)
                                outShape = infer_shape(call)
                                cur = relay.reshape(conv1,(1, curShape[0]*curShape[1], curShape[2],curShape[3]))

                                conv = copy_conv(cur, weight0, conv0)
                                out = relay.add(conv, bias0)

                                out = relay.reshape(out, outShape)
                                return self.visit(out)

                            if list(relay.frontend.common.infer_shape(weight0)) == [512, 1, 1, 512]:
                                curShape = infer_shape(conv1)
                                outShape = infer_shape(call)
                                cur = relay.reshape(conv1,(1, curShape[0]*curShape[1], curShape[2],curShape[3]))

                                conv = copy_conv(cur, weight0, conv0)
                                out = relay.add(conv, bias0)

                                out = relay.reshape(out, outShape)
                                return self.visit(out)


                new_fn = self.visit(call.op)
                new_args = [self.visit(arg) for arg in call.args]
                return tvm.relay.Call(new_fn, new_args, call.attrs)

        class splitConv2d0(relay.dataflow_pattern.DFPatternCallback):
            def __init__(self):
                super(splitConv2d0, self).__init__()
                # Data
                self.var = wildcard()  # b, 3, 224, 224
                self.weight0 = is_constant()

                self.conv2d0 = is_op('nn.conv2d')(self.var, self.weight0)

                self.pattern=self.conv2d0

            def callback(self, pre, post, node_map):
                input0 = node_map[self.var][0]

                conv0 = node_map[self.conv2d0][0]
                weight0 = node_map[self.weight0][0]

                input_shape = relay.frontend.common.infer_shape(input0)
                weight_shape = relay.frontend.common.infer_shape(weight0)
                split = 4

                if weight_shape[-1] == 1024:
                    # 切inc
                    inputList = relay.split(input0, split, axis=-1)
                    weightList = get_split_const(weight0,split,-1)
                    out = None
                    for i in range(split):
                        cur = inputList[i]
                        conv2d0 = relay.nn.conv2d(cur, relay.const(weightList[i], 'float32'),
                                                  strides=conv0.attrs['strides'],
                                                  padding=conv0.attrs['padding'],
                                                  dilation=conv0.attrs['dilation'],
                                                  groups=conv0.attrs['groups'],
                                                  channels=conv0.attrs['channels'],
                                                  kernel_size=conv0.attrs['kernel_size'],
                                                  data_layout=conv0.attrs['data_layout'],
                                                  kernel_layout=conv0.attrs['kernel_layout'],
                                                  out_layout=conv0.attrs['out_layout'],
                                                  out_dtype=conv0.attrs['out_dtype'],
                                                  is_depthwise=conv0.attrs['is_depthwise'])
                        if i == 0:
                            out = conv2d0
                        else:
                            out = relay.add(out, conv2d0)
                    gout = out
                    return gout
                else:
                    return post

        func = relay.Function(func.params, func.body, None, func.type_params, func.attrs)

        # INFO for spikeTransformer0
        # INFO 1th MS_DownSampling + MS_ConvBlock
        func = reMSConvBlock0().visit(func)

        func = reMSConvBlock1().visit(func)

        func = reMSConvBlock2().visit(func)

        # INFO 2sd MS_DownSampling + MS_ConvBlock
        func = reMSConvBlock3().visit(func)

        func = reMSConvBlock1().visit(func)

        func = reMSConvBlock2().visit(func)


        # INFO for spikeTransformer1
        func = reTransformer().rewrite(func)
        # func = reLastBackBone().visit(func)
        # func = optBackBone().visit(func)

        # # INFO for all
        func = reBuildMultiLif().visit(func)
        func = optLifConv2d().visit(func)
        func = splitConv2d0().rewrite(func)

        return func