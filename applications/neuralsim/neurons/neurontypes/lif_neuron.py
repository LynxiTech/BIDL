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

lif_params = {'alpha': 0.3, 'beta': 0.0, 'theta': 0.5, 'v_0': 0.0, 'I_e': 0.6, 'v_init': 0.0}


class lif_neuron(nn.Module):
    def __init__(self, random_seed=123, lif_params=lif_params, on_apu=True):
        super(lif_neuron, self).__init__()
        self.alpha = lif_params['alpha']
        self.beta = lif_params['beta']
        self.theta = lif_params['theta']  # thresh
        self.v_0 = lif_params['v_0']  # reset
        self.I_e = lif_params['I_e']  # dc

        self.on_apu = on_apu
        self.v = None
        self.rng = numpy.random.default_rng(random_seed)

    def reset(self, xi):
        self.v = torch.zeros_like(xi)  # init v
        #self.v = torch.tensor(self.rng.random(xi.shape), device=xi.device)  # init v

    def forward(self, x: torch.Tensor):
        if self.on_apu:
            v_temp = load(self.v.clone(), 'v')
        else:
            v_temp = self.v.clone()
        v_temp += x
        v_temp += self.I_e
        if self.on_apu:
            # reset & leakage
            self.v = v_temp.clone()
            fire = ops.custom.cmpandfire(self.v.clone(), self.theta)
            v_temp = ops.custom.resetwithdecay(self.v.clone(), self.theta, self.v_0, self.alpha, self.beta)
        else:
            # reset
            v_ = v_temp - self.theta
            fire = v_.gt(0.).float()
            fire_inv = 1. - fire
            v_temp = fire * self.v_0 + fire_inv * v_temp.clone()
            # leakage
            v_temp = self.alpha * v_temp + self.beta
        self.v = v_temp.clone()
        if self.on_apu:
            save(self.v, 'v')

        return fire
