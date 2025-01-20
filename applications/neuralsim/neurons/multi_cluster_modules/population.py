'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import torch
import numpy as np
from torch import ops
import sys
sys.path.append("../../../")
from lynadapter.warp_load_save import load,save


class Parameters(object):
    def __init__(self, neuron_param={}, V_th=15., V_reset=0., decay=0.9):
        self.V_th = neuron_param.get("V_th", V_th)
        self.V_reset = neuron_param.get("V_reset", V_reset)
        self.decay = neuron_param.get("decay", decay)



class State(object):
    def __init__(self, num=1, i_0=0., i_1=0., i_syn_ex=0., i_syn_in=0., V_m=0.):
        self.num = num
        self.i_0 = torch.full((num,), i_0, dtype=torch.float32)
        self.i_1 = torch.full((num,), i_1, dtype=torch.float32)
        self.i_syn_ex = torch.full((num,), i_syn_ex, dtype=torch.float32)
        self.i_syn_in = torch.full((num,), i_syn_in, dtype=torch.float32)
        self.V_m = torch.full((1, num), V_m, dtype=torch.float32)


class Population(torch.nn.Module):
    def __init__(self,
                 num=1,
                 resolution=0.1,
                 delay=0.1,
                 pop_index=0,
                 neuron_id=0,
                 neuron_param={},
                 ex_inh_type='excitatory',
                 on_apu=False):
        super(Population, self).__init__()
        self.on_apu = on_apu
        self.ex_inh_type = ex_inh_type
        self.neuron_param = neuron_param
        self.num = num
        self.id = np.array(range(1, self.num + 1)) + neuron_id
        self.pop_index = pop_index
        self.h = resolution
        self.delay = delay
        self.min_step = int(round(self.delay / self.h)) if self.delay >= self.h else int(round(1. / self.h))   #assert self.delay >= self.h

        # initialize state variables
        self.P_ = Parameters(self.neuron_param)
        self.S_ = State(self.num, V_m=neuron_param.get("V_m"))


        self.spikes_ex = torch.zeros(self.min_step, self.num, dtype=torch.float32)
        self.spikes_in = torch.zeros(self.min_step, self.num, dtype=torch.float32)
        self.currents = torch.zeros(self.min_step, 2, self.num, dtype=torch.float32)
        self.weighted_spikes = torch.zeros(self.min_step, self.num, dtype=torch.float32)

    def reset(self, xi):
        # self.S_.V_m = 20. * torch.ones_like(xi)
        self.S_.V_m = torch.zeros_like(xi)

    def reset_spike(self):
        self.spikes_ex = torch.zeros(self.min_step, self.num, dtype=torch.float32)
        self.spikes_in = torch.zeros(self.min_step, self.num, dtype=torch.float32)

    def update(self):
        if self.on_apu:
            v_temp = load(self.S_.V_m, f'v_{self.pop_index}')
        else:
            v_temp = self.S_.V_m

        self.S_.i_syn_ex = self.spikes_ex[0].clone()
        self.S_.i_syn_in = self.spikes_in[0].clone()

        v_temp += self.S_.i_syn_ex + self.S_.i_syn_in

        self.S_.i_0 = self.currents[0][0]
        self.S_.i_1 = self.currents[0][1]
        if v_temp.device != self.S_.i_0.device:
            v_temp_device = torch.device(v_temp.device)
            v_temp = v_temp.to(torch.device(self.S_.i_0.device))
            v_temp += self.S_.i_0 + self.S_.i_1
            v_temp = v_temp.to(v_temp_device)
        else:
            v_temp += self.S_.i_0 + self.S_.i_1

        cond = v_temp.gt(self.P_.V_th).float()
        v_temp = cond * self.P_.V_reset + (1. - cond) * v_temp
        v_temp *= self.P_.decay

        self.weighted_spikes = cond.clone()
        if self.on_apu:
            save(self.weighted_spikes, f'spike_{self.pop_index}')

        self.S_.V_m = v_temp.clone()
        if self.on_apu:
            save(self.S_.V_m, f'v_{self.pop_index}')

        return cond.clone()

    def forward(self, currents=None):
        self.currents = currents
        res = self.update()
        self.reset_spike()
        return res
