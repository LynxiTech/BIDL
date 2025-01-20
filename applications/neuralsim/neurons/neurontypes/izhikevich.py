'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import torch
from torch import ops, nn
import numpy
import sys
sys.path.append("../../../")
from lynadapter.warp_load_save import load,save

gain_coef = 100.

class izhikevich(nn.Module):
    def __init__(self,
                 random_seed=123,
                 a: float = 0.02,
                 b: float = 0.2,
                 c: float = -65.0,
                 d: float = 8.0,
                 I_e: float = 100.0,
                 V_th: float = 30.0,
                 consistent_integrathion=True,
                 on_apu=True):
        super(izhikevich, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.I_e = I_e
        self.V_th = V_th
        self.consistent_integrathion = consistent_integrathion
        self.h = 0.1

        self.on_apu = on_apu
        self.v = None
        self.u = None
        self.rng = numpy.random.default_rng(random_seed)

    def reset(self, xi):
        self.v = torch.ones_like(xi)  # init v
        #self.v = torch.tensor(self.rng.random(xi.shape), device=xi.device)  # init v
        self.u = torch.zeros_like(xi)  # init u

    def forward(self, x: torch.Tensor):
        out = []
        self.I = gain_coef * x

        scale = 10.0  #to avoid self.a * self.b be too small for fp16
        if self.on_apu:
            v_temp = load(self.v.clone(), "v")
            u_temp = load(self.u.clone(), "u")
        else:
            v_temp = self.v.clone()
            u_temp = self.u.clone()

        if self.consistent_integrathion:
            v_new = v_temp + self.h * (0.04 * v_temp * v_temp + 5.0 * v_temp + 140.0 - u_temp + self.I + self.I_e)
            u_new = u_temp + scale * self.h * self.a * (self.b * v_temp - u_temp)
        else:
            I_syn = 0.0
            v_new = v_temp + self.h * 0.5 * (
                    0.04 * v_temp * v_temp + 5.0 * v_temp + 140.0 - u_temp + self.I + self.I_e + I_syn)
            v_new += self.h * 0.5 * (0.04 * v_new * v_new + 5.0 * v_new + 140.0 - u_temp + self.I + self.I_e + I_syn)
            u_new = u_temp + scale * self.h * self.a * (self.b * v_new - u_temp)

        if self.on_apu:
            self.v = v_new.clone()
            spike = ops.custom.cmpandfire(self.v.clone(), self.V_th)
        else:
            spike = v_new.gt(self.V_th).float()

        v_new_temp = (1. - spike) * v_new + spike * self.c
        u_new_temp = (1. - spike) * (u_new / scale) + spike * (u_new / scale + self.d)
        self.v = v_new_temp.clone()
        self.u = u_new_temp.clone()

        if self.on_apu:
            save(self.v, "v")
            save(self.u, "u")
        

        
        return spike

