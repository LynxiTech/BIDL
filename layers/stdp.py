# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch
import torch.nn as nn
from torch import ops
import math
import numpy as np
from typing import Optional, Sequence, Union


import sys
sys.path.append("../../")
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel


class RSTDPLeanrner(nn.Module):
    def __init__(self,tau_pre=20.0, tau_post=20.0,weight =None,
               nu = None,
               class_num=0,
               neurons_num=0,
               reduction = None,
               weight_decay = 0.0,
               traces_additive = True,
               reward_fn= None,
                weight_min = -np.inf,
                weight_max = np.inf,
                norm= None,
                trace_scale = 1.0,
                momentum_reward = 0.95,
                ON_APU=False,
               **kwargs,
              ):        
        super(RSTDPLeanrner,self).__init__()
        self.tau_pre = torch.tensor(tau_pre)
        self.tau_post = torch.tensor(tau_post)  

        if nu is None:
            self.nu = torch.tensor([0.0, 0.0], dtype=torch.float)
        else:
            self.nu = torch.tensor(nu)
        self.class_num = class_num
        self.neurons_num = neurons_num

        if reduction is None:
            self.reduction = torch.sum
        else:
            self.reduction = reduction
        self.dt = 1.0
        self.trace_scale = trace_scale
        self.traces_additive = traces_additive
        self.source_x = None
        self.target_x = None
        self.reward = 0.
        self.compute_decays()
        self.distance = np.inf
        self.momentum_reward = momentum_reward
        self.ON_APU = ON_APU
        

    def reward_fn(self,gt_label,target_s): 
        target_s_temp = target_s.reshape(self.class_num,self.neurons_num)
        target_s_temp = torch.mean(target_s_temp,dim=1)   
        reward = gt_label-target_s_temp
        reward = reward.repeat(self.neurons_num,1).t().reshape(-1)

        return reward


    def compute_decays(self):
        self.pre_trace_decay = torch.exp(-self.dt / self.tau_pre) 
        self.post_trace_decay = torch.exp(-self.dt / self.tau_post)  
        

    def update(self, source_s, target_s,gt_label):
        if self.ON_APU:
            source_x = load(self.source_x.clone(), f'sx')
            target_x = load(self.target_x.clone(), f'tx')

        else:
            source_x = self.source_x.clone()
            target_x = self.target_x.clone()

        reward_new = self.reward_fn(gt_label, target_s)

        source_x = source_x * self.pre_trace_decay
        target_x = target_x * self.post_trace_decay

        if self.traces_additive:
            source_x = source_x + self.trace_scale*source_s#.float()
            target_x = target_x + self.trace_scale*target_s#.float()
        else:
            tem = source_x + source_s*self.trace_scale
            tem1 = source_x * source_s
            source_x = (tem - tem1)
        
            tem = self.target_x + target_s*self.trace_scale
            tem1 = self.target_x * target_s
            target_x = (tem - tem1)


        source_s_temp  = source_s.permute(1,0)

        source_x_temp = source_x.permute(1,0)
        target_x_temp = target_x#.unsqueeze(1)
        
        update1 = torch.mm(source_s_temp,target_x_temp)
        update2 = torch.mm(source_x_temp,target_s) 

        delta_w = ((self.nu[1]*update2) - (self.nu[0]*update1))
        delta_w = delta_w*reward_new       


        if self.ON_APU:
            self.source_x = source_x.clone()
            self.target_x = target_x.clone()
            save(self.source_x, f'sx')
            save(self.target_x, f'tx')
        else:
            self.source_x = source_x
            self.target_x = target_x



        return delta_w

    def update_scale(self, source_s, target_s,gt_label):
        if self.ON_APU:
            source_x = load(self.source_x.clone(), f'sx')
            target_x = load(self.target_x.clone(), f'tx')

        else:
            source_x = self.source_x.clone()
            target_x = self.target_x.clone()

        reward_new = self.reward_fn(gt_label, target_s)

        source_x = source_x * self.pre_trace_decay
        target_x = target_x * self.post_trace_decay

        if self.traces_additive:
            source_x = source_x + self.trace_scale*source_s#.float()
            target_x = target_x + self.trace_scale*target_s#.float()
        else:
            tem = source_x + source_s*self.trace_scale
            tem1 = source_x * source_s
            source_x = (tem - tem1)
        
            tem = self.target_x + target_s*self.trace_scale
            tem1 = self.target_x * target_s
            target_x = (tem - tem1)


        source_s_temp  = source_s.permute(1,0)

        source_x_temp = source_x.permute(1,0)
        target_x_temp = target_x#.unsqueeze(1)
        
        update1 = torch.mm(source_s_temp,target_x_temp)
        update2 = torch.mm(source_x_temp,target_s) 

        delta_w = ((self.nu[1]*5000.*update2) - (self.nu[0]*5000.*update1))
        delta_w = delta_w*reward_new       


        if self.ON_APU:
            self.source_x = source_x.clone()
            self.target_x = target_x.clone()
            save(self.source_x, f'sx')
            save(self.target_x, f'tx')
        else:
            self.source_x = source_x
            self.target_x = target_x



        return delta_w

            

    def reset(self,source_s,target_shape):
        self.source_x = torch.zeros_like(source_s,dtype=source_s.dtype,device=source_s.device,requires_grad=False)
        self.target_x = torch.zeros(target_shape,dtype=source_s.dtype,device=source_s.device,requires_grad=False)




class Hebbian(nn.Module):
    def __init__(self,tau_pre=10.0, tau_post=10.0,weight=None,
               nu = None,
               reduction = None,
               weight_decay = 0.0,
               traces_additive = True,
               norm = None,
                weight_min = -np.inf,
                weight_max = np.inf,
                trace_scale = 1.0,
                ON_APU=False,
               **kwargs, 
              ):
        super(Hebbian,self).__init__()
        self.weight = weight
        self.tau_pre = torch.tensor(tau_pre)
        self.tau_post = torch.tensor(tau_post)  

        if nu is None:
            self.nu = torch.tensor([0.0, 0.0], dtype=torch.float)
        else:
            self.nu = torch.tensor(nu)


        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            self.reduction = torch.sum
        else:
            self.reduction = reduction
        self.dt = 1.0
        self.trace_scale = trace_scale
        self.traces_additive = True
        self.norm = None
        self.source_x = None
        self.target_x = None       
        self.compute_decays()
        self.ON_APU = ON_APU

    def compute_decays(self):
        self.pre_trace_decay = torch.exp(-self.dt / self.tau_pre)  
        self.post_trace_decay = torch.exp(-self.dt / self.tau_post)     
        

    def update(self, source_s, target_s,gt_label,finetune):

        self.source_x = self.source_x * self.pre_trace_decay
        self.target_x = self.target_x * self.post_trace_decay
        if self.traces_additive:
            self.source_x += self.trace_scale*source_s.float()
            self.target_x += self.trace_scale*target_s.float()
        else:
            tem = self.source_x + source_s
            tem1 = self.source_x * source_s
            self.source_x = (tem - tem1)
        
            tem = self.target_x + target_s
            tem1 = self.target_x * target_s
            self.target_x = (tem - tem1)

        source_s  = source_s.unsqueeze(-1)
        source_x = self.source_x.unsqueeze(-1)
        target_s = target_s.unsqueeze(1)
        target_x = self.target_x.unsqueeze(1)
        update1 = self.reduction(torch.bmm(source_s,target_x),dim=0)     
 
        update2 = self.reduction(torch.bmm(source_x,target_s),dim=0)        
        
        delta_w = (self.nu[1]*update2+ self.nu[0]*update1)*finetune

     

        w = w + delta_w
        super().update()

        return self.weight

            

    def reset(self,source_s,target_s):
        self.source_x = torch.zeros_like(source_s,dtype=source_s.dtype,device=source_s.device,requires_grad=False)
        self.target_x = torch.zeros_like(target_s,dtype=target_s.dtype,device=target_s.device,requires_grad=False)
        self.weight = self.weight.to(source_s.device)

       