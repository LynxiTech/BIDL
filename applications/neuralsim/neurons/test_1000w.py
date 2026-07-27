'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch
import numpy as np
from torch import ops, nn
import time
from neurontypes import lif_neuron
import pickle as pk
import random
import json
import sys






def parse_args():
    parser = argparse.ArgumentParser(description='neuron model simulation')
    parser.add_argument('--use_lyngor', type=int, help='use lyngor flag: 1 or 0, means if use lyngor or not')
    parser.add_argument('--use_legacy', type=int, help='use legacy flag: 1 or 0, means if use legacy or not')
    parser.add_argument('--use_gpu', type=int, help='use gpu flag: 1 or 0, means if use gpu or not')
    parser.add_argument('--device', default=0, type=int,help='apu device id')

    args = parser.parse_args()
    return args


class lifSnn(nn.Module):
    def __init__(self, on_apu=True) -> None:
        super(lifSnn, self).__init__()
        self.node = lif_neuron(on_apu=on_apu)
        self.on_apu = on_apu

    def forward(self, xi):
        if self.on_apu:
            assert len(xi.shape) == 4
            self.node.reset(xi)
            s = self.node(xi)
            return s
        else:
            s = self.node(xi)
            return s





def main():
    args = parse_args()    
    assert args.use_lyngor in (0, 1), 'use_lyngor must in (0, 1)'
    assert args.use_legacy in (0, 1), 'use_legacy must in (0, 1)'
    assert args.use_gpu in (0, 1), 'use_gpu must in (0, 1)'    
    assert isinstance(args.device, int), 'apu chip id must be int'
    chip_id = args.device
    USE_LYNGOR = True if args.use_lyngor == 1 else False
    USE_LEGACY = True if args.use_legacy == 1 else False
    USE_GPU = True if args.use_gpu == 1 else False
   

   

    # b, n, t, c, h, w = 1, 1, 50, 2, 50, 10
    b, n, t, c, h, w = 1, 1, 100, 200, 500, 100

    input_data = torch.randn(b, n, t, c, h, w)
    print("t: ", t)
   
        
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]    

    if USE_LYNGOR:

        
        model = lifSnn(on_apu=True)
      
        model_name = model.__class__.__name__
        opath = os.path.join(os.path.join(ROOT, "../../../model_files/neuralsim"),  model_name)        
        from lynadapter.lyn_sdk_model import ApuRun,ApuRun_Single
       
        in_size = [((1, c, h, w),)]  
        if not USE_LEGACY:
            from lynadapter.lyn_compile import model_compile
            input_type = "float32"
            model_compile(model,opath,in_size,input_type=input_type)           
            
            
        model_path = os.path.join(opath, "Net_0")

        
       
        arun = ApuRun_Single(chip_id, model_path)
        
        t0 = time.time()
       
        apu_result =[]
        for bi in range(n):
           
            data = input_data[:, bi,  ...].numpy().astype(np.float32)
            
            
            for i in range(t): 
                output = arun.run(data[:,i,...],i)
                
                apu_result.append(output[0])
             

             
        #apu_result = np.concatenate(outputs, axis=0)
        arun.apu_unload()
        t1 = time.time()
        
        print("apu: timesteps/s: ", n * t / (t1 - t0))
       

    if USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        random.seed(456)
        torch.manual_seed(1234)
        rng = np.random.default_rng(seed=1234)
        model = lifSnn(on_apu=False).to(device)
        
        gpu_result = []
        t0 = time.time()
        for bi in range(n):
            for i in range(t):
               
                xi = input_data[:, bi, i, ...].to(device)
                if i == 0:
                    model.node.reset(xi)

                s = model(xi)

                if torch.cuda.is_available():
                    gpu_result.append(s.cpu().numpy())
                else:
                    gpu_result.append( s.numpy())
        t1 = time.time()
        print("gpu: timesteps/s: ", n * t / (t1 - t0))

    if USE_GPU and USE_LYNGOR:
        fire_apu = 0
        fire_gpu = 0
        for bi in range(n * t):
            apu_b = apu_result[bi]
            gpu_b = gpu_result[bi]
            fire_apu += np.sum(apu_b == 1)
            fire_gpu += np.sum(gpu_b == 1)
       
        
        fire_total_num = n * t * c * h * w

        if fire_apu == 0 and fire_gpu == 0:
            fire_rate_acc = 0.0
        elif fire_gpu == 0:
            fire_rate_acc = (abs(fire_apu - fire_gpu) / fire_total_num) / (fire_apu / fire_total_num)
        else:
            fire_rate_acc = (abs(fire_apu - fire_gpu) / fire_total_num) / (fire_gpu / fire_total_num)
        print(f"fire rate difference: {round(fire_rate_acc * 100., 3)}%")
    

    
       




if __name__ == '__main__':
    main()
