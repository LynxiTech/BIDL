import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../../")
from lynadapter.warp_load_save import load,save

class STDPLearner(nn.Module):
    def __init__(self, tau_pre=10., tau_post=10.,nu = None,reduction = None,
               weight_decay = 0.0,
               traces_additive = False,trace_scale =0.01,on_apu=False):
        super().__init__()               
        self.tau_pre = torch.tensor(tau_pre)
        self.tau_post = torch.tensor(tau_post) 
        if nu is None:
            self.nu = torch.tensor([0.01, 0.01], dtype=torch.float)
        else:
            self.nu = torch.tensor(nu) 
        if reduction is None:
            self.reduction = torch.sum
        else:
            self.reduction = reduction
        self.dt = 1.0
        self.trace_scale = trace_scale
        self.traces_additive = traces_additive
        self.compute_decays()
        self.on_apu = on_apu
        self.source_x = None
        self.target_x = None

    def compute_decays(self):
        self.pre_trace_decay = torch.exp(-self.dt / self.tau_pre) 
        self.post_trace_decay = torch.exp(-self.dt / self.tau_post)  


    def update(self, source_s, target_s):
        if self.on_apu:
            source_x = load(self.source_x.clone(), f'sx')
            target_x = load(self.target_x.clone(), f'tx')

        else:
            source_x = self.source_x.clone()
            target_x = self.target_x.clone()
 
        source_x = source_x * self.pre_trace_decay

        target_x = target_x * self.post_trace_decay
        
        
        if self.traces_additive:
            source_x = source_x + self.trace_scale*source_s#.float()
            target_x = target_x + self.trace_scale*target_s#.float()
        else:

            tem = source_x + source_s*self.trace_scale
            tem1 = source_x * source_s
            source_x = (tem - tem1)
       
        
            tem = target_x + target_s*self.trace_scale
            
            tem1 = target_x * target_s

            target_x = (tem - tem1)
        
  
        source_s_temp  = source_s.t()

        source_x_temp =  source_x.t()
        #target_x_temp = target_x#.unsqueeze(1)
    
        update1 = torch.mm(source_s_temp,target_x)

        update2 = torch.mm(source_x_temp,target_s) 
        

        delta_w = ((self.nu[1]*update2) - (self.nu[0]*update1))
        #delta_w = ((self.nu[1]*update2) - (self.nu[0]*update1))
        #print(torch.min(delta_w),torch.max(delta_w))
        
       


        if self.on_apu:
            self.source_x = source_x.clone()
            self.target_x = target_x.clone()
            save(self.source_x, f'sx')
            save(self.target_x, f'tx')
        else:
            self.source_x = source_x
            self.target_x = target_x


        
        return delta_w

    def reset(self,source_num,target_num):
        self.source_x = torch.zeros(1,source_num,dtype=torch.float32,requires_grad=False)#,device=source_num.device,requires_grad=False)
        self.target_x = torch.zeros(1,target_num,dtype=torch.float32,requires_grad=False)#,device=target_num.device,requires_grad=False)
