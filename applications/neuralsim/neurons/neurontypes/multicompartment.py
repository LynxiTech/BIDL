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

compartment_params = {'V_rest': 0.0, 'V_thre': 20.0, 'g_L_s': 0.1, 'g_L_d1': 0.1, 'g_L_d2': 0.1, 'g_d1_s': 0.1,
                      'g_s_d1': 0.1, 'g_d2_d1': 0.1, 'g_d1_d2': 0.1}

gain_coef = 10.

# Multi compartment LIF neuron
class MultiCompartment(nn.Module):
    def __init__(self, random_seed=123,params=compartment_params, dt=1, on_apu=False):
        super(MultiCompartment, self).__init__()

        self.dt = dt
        self.V_rest = params['V_rest']
        self.V_thre = params['V_thre']
        self.g_L_s = params['g_L_s']
        self.g_L_d1 = params['g_L_d1']
        self.g_L_d2 = params['g_L_d2']
        self.g_d1_s = params['g_d1_s']
        self.g_s_d1 = params['g_s_d1']
        self.g_d2_d1 = params['g_d2_d1']
        self.g_d1_d2 = params['g_d1_d2']

        self.on_apu = on_apu
        self.V_s = None
        self.V_d1 = None
        self.V_d2 = None
        self.rng = numpy.random.default_rng(random_seed)

    def reset(self, xi):
        self.V_s = torch.zeros_like(xi)
        #self.V_s = torch.tensor(self.rng.random(xi.shape), device=xi.device)  # init V_s
        self.V_d1 = torch.zeros_like(xi)
        #self.V_d1 = torch.tensor(self.rng.random(xi.shape), device=xi.device)  # init V_d1
        self.V_d2 = torch.zeros_like(xi)
        #self.V_d2 = torch.tensor(self.rng.random(xi.shape), device=xi.device)  # init V_d2

    def forward(self, Iinj_s, Iinj_d1, Iinj_d2):
        if self.on_apu:
            V_s_temp = load(self.V_s.clone(), "V_s")
            V_d1_temp = load(self.V_d1.clone(), "V_d1")
            V_d2_temp = load(self.V_d2.clone(), "V_d2")
        else:
            V_s_temp = self.V_s.clone()
            V_d1_temp = self.V_d1.clone()
            V_d2_temp = self.V_d2.clone()

        # soma
        s_tmp1 = self.g_L_s * (V_s_temp - self.V_rest)
        s_tmp2 = self.g_d1_s * (V_s_temp - V_d1_temp)
        dVsdt = s_tmp1 + s_tmp2 - gain_coef * Iinj_s

        # dendrite1
        d1_tmp1 = self.g_L_d1 * (V_d1_temp - self.V_rest)
        d1_tmp2 = self.g_s_d1 * (V_d1_temp - V_s_temp)
        d1_tmp3 = self.g_d2_d1 * (V_d1_temp - V_d2_temp)
        dVd1dt = d1_tmp1 + d1_tmp2 + d1_tmp3 - Iinj_d1

        # dendrite2
        d2_tmp1 = self.g_L_d2 * (V_d2_temp - self.V_rest)
        d2_tmp2 = self.g_d1_d2 * (V_d2_temp - V_d1_temp)
        dVd2dt = d2_tmp1 + d2_tmp2 - Iinj_d2

        V_s_temp = V_s_temp - self.dt * dVsdt
        V_d1_temp = V_d1_temp - self.dt * dVd1dt
        V_d2_temp = V_d2_temp - self.dt * dVd2dt

        # spiking
        if self.on_apu:
            self.V_s = V_s_temp.clone()
            self.V_d1 = V_d1_temp.clone()
            self.V_d2 = V_d2_temp.clone()
            spike = ops.custom.cmpandfire(self.V_s.clone(), self.V_thre)
            spike_d1 = ops.custom.cmpandfire(self.V_d1.clone(), self.V_thre)
            spike_d2 = ops.custom.cmpandfire(self.V_d2.clone(), self.V_thre)
        else:
            spike = V_s_temp.gt(self.V_thre).float()
            spike_d1 = V_d1_temp.gt(self.V_thre).float()
            spike_d2 = V_d2_temp.gt(self.V_thre).float()
        spike_inv = 1. - spike
        spike_d1_inv = 1. - spike_d1
        spike_d2_inv = 1. - spike_d2

        V_s_temp = spike * self.V_rest + spike_inv * V_s_temp
        V_d1_temp = spike_d1 * self.V_rest + spike_d1_inv * V_d1_temp
        V_d2_temp = spike_d2 * self.V_rest + spike_d2_inv * V_d2_temp
        self.V_s = V_s_temp.clone()
        self.V_d1 = V_d1_temp.clone()
        self.V_d2 = V_d2_temp.clone()
        if self.on_apu:
            save(self.V_s, "V_s")
            save(self.V_d1, "V_d1")
            save(self.V_d2, "V_d2")

        return spike
