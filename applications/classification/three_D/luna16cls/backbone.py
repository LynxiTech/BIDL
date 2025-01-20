# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
from torch.nn import Flatten
import sys
sys.path.append("../")
from layers.lif import Conv2dLif
from layers.lifplus import Conv2dLifPlus
from layers.gradient_approx import *
sys.path.append("../../")
from lynadapter.warp_load_save import load_kernel,save_kernel
from utils import globals
globals._init()

class SeqClif3Fc3LcItout(nn.Module):


    def __init__(self,
                 timestep=8, input_channels=1, h=32, w=32, nclass=2, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None,spike_func=None,use_inner_loop=False
                 ):
        super(SeqClif3Fc3LcItout, self).__init__()
        neuron=neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLif(input_channels, 32, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp1 = nn.MaxPool2d(2, stride=2)       
        self.clif2 = Conv2dLif(32, 64, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.clif3 = Conv2dLif(64, 128, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        assert amode == 'mean'       # assemble model, optional:'sum', 'pick'
        self.flat = Flatten(1, -1)
        self.head = nn.Sequential(
            nn.Linear(h // 8 * w // 8 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, nclass)
        )
        self.tempAdd = None
        self.timestep = timestep
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        self.MODE = globals.get_value('MODE')

    def reset(self, xi):
        self.tempAdd = pt.zeros_like(xi)

    def forward(self, xis: pt.Tensor) -> pt.Tensor:
        if self.ON_APU:
            assert len(xis.shape) == 4
            x0 = xis
            x1 = self.mp1(self.clif1(x0))
            x2 = self.mp2(self.clif2(x1))
            x3 = self.mp3(self.clif3(x2))
            x4 = self.flat(x3)
            x5 = self.head(x4)
            x5 = x5.unsqueeze(2).unsqueeze(3)
            self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', mode=self.MODE,init_zero_use_data=x5)
            self.tempAdd = self.tempAdd + x5 / self.timestep
            output = self.tempAdd.clone()
            save_kernel(self.tempAdd, f'tempAdd', mode=self.MODE)
            return output.squeeze(-1).squeeze(-1)

        else:
            t = xis.size(1)
            xo = 0
            for i in range(t):
                x0 = xis[:, i, ...]
                if i == 0: self.clif1.reset(x0)
                x1 = self.mp1(self.clif1(x0))
                if i == 0: self.clif2.reset(x1)
                x2 = self.mp2(self.clif2(x1))
                if i == 0: self.clif3.reset(x2)
                x3 = self.mp3(self.clif3(x2))
                x4 = self.flat(x3)
                x5 = self.head(x4)
                xo = xo + x5 / self.timestep
            return xo
        
class SeqClifplus3Fc3LcItout(nn.Module):


    def __init__(self,
                 timestep=8, input_channels=1, h=32, w=32, nclass=2, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None
                 ):
        super(SeqClifplus3Fc3LcItout, self).__init__()
        neuron=neuron.lower()
        assert neuron in ['lifplus']
        if neuron == 'lifplus' and neuron_config != None:
            input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay = neuron_config

        self.clif1 = Conv2dLifPlus(input_channels, 32, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, input_accum=input_accum,
                                rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay,
                                noise=noise,spike_func=None,use_inner_loop=False)
        self.mp1 = nn.MaxPool2d(2, stride=2)

        self.clif2 = Conv2dLifPlus(32, 64, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, input_accum=input_accum,
                                rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay, 
                                noise=noise,spike_func=None,use_inner_loop=False)
        self.mp2 = nn.MaxPool2d(2, stride=2)

        self.clif3 = Conv2dLifPlus(64, 128, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, input_accum=input_accum,
                                rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay,
                                noise=noise,spike_func=None,use_inner_loop=False)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        assert amode == 'mean'       # assemble model, optional:'sum', 'pick'
        self.flat = Flatten(1, -1)
        self.head = nn.Sequential(
            nn.Linear(h // 8 * w // 8 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, nclass)
        )
        self.tempAdd = None
        self.timestep = timestep
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')

    def reset(self, xi):
        self.tempAdd = pt.zeros_like(xi)

    def forward(self, xis: pt.Tensor) -> pt.Tensor:
        if self.ON_APU:
            assert len(xis.shape) == 4
            x0 = xis

            x1 = self.mp1(self.clif1(x0))
            x2 = self.mp2(self.clif2(x1))
            x3 = self.mp3(self.clif3(x2))

            x4 = self.flat(x3)
            x5 = self.head(x4)
            x5 = x5.unsqueeze(2).unsqueeze(3)
            self.tempAdd = load_kernel(self.tempAdd, f'tempAdd',init_zero_use_data=x5)
            self.tempAdd = self.tempAdd + x5 / self.timestep
            output = self.tempAdd.clone()
            save_kernel(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)

        else:
            t = xis.size(1)
            xo = 0
            for i in range(t):
                x0 = xis[:, i, ...]
                if i == 0: self.clif1.reset(x0)
                x1 = self.mp1(self.clif1(x0))
                if i == 0: self.clif2.reset(x1)
                x2 = self.mp2(self.clif2(x1))
                if i == 0: self.clif3.reset(x2)
                x3 = self.mp3(self.clif3(x2))
                x4 = self.flat(x3)
                x5 = self.head(x4)
                xo = xo + x5 / self.timestep

            return xo