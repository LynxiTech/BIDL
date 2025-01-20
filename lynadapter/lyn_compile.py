# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


from torch import ops
import os
import sys

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]



def load_library():
    # load libcustom_op.so
    library_path = os.path.join(ROOT,"./custom_op_in_pytorch/build/libcustom_ops.so")
    ops.load_library(library_path)
    
    
def model_compile(model,_base_,in_size,version=0,batch_size=1,input_type = "uint8",post_mode=None,profiler=False,core_mem_mode=None):
    #checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)      

    #sys.path.append(ROOT)
    from . import model_operate 
    
    from  .custom_op_in_lyn import custom_op_my_lif,custom_op_rand
    
    load_library()   
    
    model_operate.run(model, in_size, out_path=f'{_base_}',input_type=input_type,version=version,batch_size=batch_size,post_mode=post_mode,profiler=profiler,
                      core_mem_mode=core_mem_mode)  