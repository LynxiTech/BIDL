# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
import lyngor as lyn
from torch import ops
from cann import CANN, CANN_apu
import cann
import os

root = os.getcwd()
for _ in range(3):
    root = os.path.dirname(root)   

if __name__ == '__main__':

    target = 'apu'
    cann.ON_APU = True
    pt_model = CANN()

    inputs_dict = {'net_in': (3360,)}
    out_path = root + "/model_files/videotracking/cann"
    
    model = lyn.DLModel()
    model.load(pt_model.eval(), model_type="Pytorch", inputs_dict=inputs_dict)

    builder = lyn.Builder(target=target, is_map=True)
    # builder = lyn.Builder(target=target, is_map=False, cpu_arch='x86', cc="g++")  # m

    out_path = builder.build(model.mod, model.params, opt_level=3, out_path=out_path, version=0)
    # out_path = builder.build(model.mod, model.params, opt_level=3, out_path="./demo_m/", save_graph=True, version=0)  # m
