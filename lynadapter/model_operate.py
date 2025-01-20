# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import lyngor as lyn
import torch
import numpy as np


def debug():
    import os;
    os.environ["DMLC_LOG_DEBUG"] = "1"
    lyn.logger.setLevel('DEBUG')
    lyn.logger_time.setLevel('DEBUG')


def run_custom_op_in_model_by_pytorch(model, dict_data):
    graph = model
    input_data = dict_data['data']  # .astype("float32")
    input = torch.tensor(input_data)
    ref_out = graph.forward(input)
    if isinstance(ref_out, tuple):
        ref_out_numpy = [r.detach().numpy() for r in ref_out]
    else:
        ref_out_numpy = [ref_out.detach().numpy()]
    return ref_out_numpy


# compare result between lyngor and pytorch.
def compare_result(ref_result, tvm_result, error_value):
    if type(ref_result) != list:
        ref_result = [ref_result]
    if type(tvm_result) != list:
        tvm_result = [tvm_result]

    mse_errror = ((tvm_result[0] - ref_result[0]) ** 2).mean(axis=None)

    print("[INFO] error:", mse_errror)
    assert mse_errror < error_value, "tvm and ref result not match, error is: %f" % mse_errror


# Registering Operator to lyngor.
# from examples.custom_op.pytorch_load_save.custom_op_in_lyn.custom_op_load_save import *
# from custom_op_in_lyn.custom_op_load_save import *

def run_custom_op_in_model_by_lyn(in_size, model, dict_data, out_path, target="apu",input_type="float32",version=0,batch_size=1,
                                  post_mode=None,profiler=False,core_mem_mode=None):
    lyn.op_graph_batch_limit = 25   # default set
    dict_inshape = {}
    # print(len(in_size[0]))
    for i in range(len(in_size[0])):
        dict_inshape.update({f'data_{i}': in_size[0][i]})    
    # *1.DLmodel load
    lyn_model = lyn.DLModel()
    model_type = 'Pytorch'
    lyn_model.load(model, model_type, inputs_dict= dict_inshape,in_type=input_type)
    # *2.DLmodel build
    # lyn_module = lyn.Builder(target=target, is_map=False, cpu_arch='x86', cc="g++")
    lyn_module = lyn.Builder(target=target, is_map=True)
    opt_level = 3
    if batch_size==1:
        if post_mode is not None:
            module_path = lyn_module.build(lyn_model.mod, lyn_model.params, opt_level, out_path=out_path,version=version,post_mode=post_mode,profiler=profiler)
        else:
            if core_mem_mode is not None:
                
                module_path = lyn_module.build(lyn_model.mod, lyn_model.params, opt_level, out_path=out_path,version=version,profiler=profiler,core_mem_mode=core_mem_mode)
            else:
                module_path = lyn_module.build(lyn_model.mod, lyn_model.params, opt_level, out_path=out_path,version=version,profiler=profiler)
    else:
        if post_mode is not None:
            module_path = lyn_module.build(lyn_model.mod, lyn_model.params, opt_level, out_path=out_path,version=1,run_batch=batch_size,post_mode=post_mode,profiler=profiler)
        else:
            module_path = lyn_module.build(lyn_model.mod, lyn_model.params, opt_level, out_path=out_path,version=1,run_batch=batch_size,profiler=profiler)

    # *3.module load
    # module = lyn.loader.load(path=module_path+"/Net_0")
    # *4.module run
    # module.run(data_format='numpy', **dict_data)
    # return module.get_output()


# run Module and compare result.
def run(model, in_size, out_path, target="apu",input_type="float32",version=0,batch_size=1,post_mode=None,profiler=False,core_mem_mode=None):
    data_type = 'float16'
    # * generater input data
    shape = in_size[0][0]
    data = np.random.randint(0, 2, shape).astype(data_type)
    dict_data = {}
    dict_data.update({'data': data})
    # * run by lyngor
    lyn_result = run_custom_op_in_model_by_lyn(in_size, model, dict_data, out_path, target,input_type=input_type,version=version,batch_size=batch_size,
                                               post_mode = post_mode,profiler=profiler,core_mem_mode=core_mem_mode)
    # * run by pytorch
    # pytorch_result = run_custom_op_in_model_by_pytorch( model, dict_data, )
    # print(pytorch_result[0].shape)

    # * compare result
    # print("[INFO] lyn_result:",lyn_result)
    # print("[INFO] pytorch_result:",pytorch_result)
    # compare_result(pytorch_result, lyn_result, 0.1)
