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


class Adex(nn.Module):
    def __init__(self,
                 random_seed=123,
                 a: float = 0.0,
                 b: float = 60.0,
                 R: float = 0.5,
                 tau_m: float = 30.0,
                 tau_w: float = 30.0,
                 delta: float = 10.,
                 v_th: float = -50.0,
                 v_peak: float = 5.0,
                 v_rest: float = -70.0,
                 v_reset: float = 35.0,
                 dt: float = 0.1,
                 on_apu=True):
        super(Adex, self).__init__()
        self.a = a
        self.b = b
        self.R = R
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.delta = delta
        self.v_th = v_th
        self.v_peak = v_peak
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.dt = dt

        self.on_apu = on_apu
        self.v = None
        self.w = None
        self.rng = numpy.random.default_rng(random_seed)

    def reset(self, xi):
        self.v = torch.zeros_like(xi)  # init v
        #self.v = torch.tensor(self.rng.random(xi.shape), device=xi.device)  # init v
        self.w = torch.zeros_like(xi)  # init w

    def forward(self, x: torch.Tensor):
        inpt = x
        if self.on_apu:
            v_temp = load(self.v.clone(), "v")
            w_temp = load(self.w.clone(), "w")
        else:
            v_temp = self.v.clone()
            w_temp = self.w.clone()

        temp1 = self.v_rest - v_temp
        temp2 = torch.exp((v_temp - self.v_th) / self.delta)
        temp3 = temp1 + self.delta * temp2 - self.R * w_temp + self.R * inpt

        wtemp = self.a * (v_temp - self.v_rest) - w_temp
        w_new = w_temp + self.dt * wtemp / self.tau_w

        v_new = v_temp + self.dt * temp3 / self.tau_m

        # spiking
        if self.on_apu:
            self.v = v_new.clone()
            spike = ops.custom.cmpandfire(self.v.clone(), self.v_peak)
        else:
            spike = v_new.gt(self.v_peak).float()
        spike_inv = 1. - spike

        v_new_temp = spike * self.v_reset + spike_inv * v_new
        w_new_temp = spike * (w_new + self.b) + spike_inv * w_new
        self.v = v_new_temp.clone()
        self.w = w_new_temp.clone()
        if self.on_apu:
            save(self.v, "v")
            save(self.w, "w")

        return spike
