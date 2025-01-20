# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
from torch.nn import Flatten
import sys
sys.path.append("../")
from layers.lif import Conv2dLif, FcLif
from layers.lifplus import Conv2dLifPlus, FcLifPlus
from layers.gradient_approx import *
sys.path.append("../../")
from lynadapter.warp_load_save import load_kernel,save_kernel
from utils import globals
globals._init()

class SeqClif3Flif2DgItout(nn.Module):

    def __init__(self,
                 timestep=60, input_channels=2, h=40, w=40, nclass=11, cmode='spike', fmode='spike', amode='mean', soma_params='all_share', 
                 noise=1e-3, neuron='lif', neuron_config=None,spike_func=None,use_inner_loop=False):
        super(SeqClif3Flif2DgItout, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.norm = nn.BatchNorm2d(input_channels)
        self.clif1 = Conv2dLif(input_channels, 64, 3, stride=2, padding=1, feed_back=False,mode=cmode, soma_params=soma_params,
                                noise=noise,spike_func=None,use_inner_loop=False)
        self.clif2 = Conv2dLif(64, 128, 3, stride=2, padding=1, feed_back=False, mode=cmode,
                                soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.clif3 = Conv2dLif(128, 256, 3, stride=2, padding=1, feed_back=False,mode=cmode,
                                soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.flat = Flatten(1, -1)
        self.flif1 = FcLif(h // 8 * w // 8 * 256, 256, mode=fmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.flif2 = FcLif(256, nclass, mode=fmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        assert amode == 'mean'
        self.tempAdd = None
        self.timestep = timestep
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        self.MULTINET = globals.get_value('MULTINET')
        self.MODE = globals.get_value('MODE')

    def reset(self, xi):
        self.tempAdd = pt.zeros_like(xi)

    def forward(self, xis: pt.Tensor) -> pt.Tensor:
        if self.ON_APU:
            assert len(xis.shape) == 4
            x0 = xis
            x0 = self.norm(x0)
            x1 = self.clif1(x0)
            x2 = self.clif2(x1)
            x3 = self.clif3(x2)
            x4 = self.flat(x3)
            x5 = self.flif1(x4)
            x6 = self.flif2(x5)
            x6 = x6.unsqueeze(2).unsqueeze(3)
            if self.MULTINET:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE,init_zero_use_data=x6)
            else:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', mode=self.MODE,init_zero_use_data=x6)
            self.tempAdd = self.tempAdd + x6 / self.timestep
            output = self.tempAdd.clone()
            if self.MULTINET:
                save_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE)
            else:
                save_kernel(self.tempAdd, f'tempAdd', mode=self.MODE)
            return output.squeeze(-1).squeeze(-1)
        else:
            t = xis.size(1)
            xo = 0
            for i in range(t):
                x0 = xis[:, i, ...]
                x0 = self.norm(x0)
                if i == 0: self.clif1.reset(x0)
                x1 = self.clif1(x0)
                if i == 0: self.clif2.reset(x1)
                x2 = self.clif2(x1)
                if i == 0: self.clif3.reset(x2)
                x3 = self.clif3(x2)
                x4 = self.flat(x3)
                if i == 0: self.flif1.reset(x4)
                x5 = self.flif1(x4)
                if i == 0: self.flif2.reset(x5)
                x6 = self.flif2(x5)
                xo = xo + x6 / t

            return xo
        
class SeqClifplus3Flifplus2DgItout(nn.Module):

    def __init__(self,
                 timestep=60, input_channels=2, h=40, w=40, nclass=11, cmode='spike', fmode='spike', amode='mean', soma_params='all_share', 
                 noise=1e-3, neuron='lif', neuron_config=None):
        super(SeqClifplus3Flifplus2DgItout, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lifplus']
        if neuron == 'lifplus' and neuron_config != None:
            input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay = neuron_config
        self.norm = nn.BatchNorm2d(input_channels)
       
        self.clif1 = Conv2dLifPlus(input_channels, 64, 3, stride=2, padding=1, feed_back=False, mode=cmode, soma_params=soma_params, input_accum=input_accum,
                                rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay,
                                noise=noise,spike_func=None,use_inner_loop=False)
        
        self.clif2 = Conv2dLifPlus(64, 128, 3, stride=2, padding=1, feed_back=False, mode=cmode, soma_params=soma_params, input_accum=input_accum,
                                rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay,
                                noise=noise,spike_func=None,use_inner_loop=False)
        
        self.clif3 = Conv2dLifPlus(128, 256, 3, stride=2, padding=1, feed_back=False, mode=cmode, soma_params=soma_params, input_accum=input_accum,
                                rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay,
                                noise=noise,spike_func=None,use_inner_loop=False)
        self.flat = Flatten(1, -1)
        
        self.flif1 = FcLifPlus(h // 8 * w // 8 * 256, 256, mode=fmode, soma_params=soma_params, input_accum=input_accum,
                            rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay,
                            noise=noise,spike_func=None,use_inner_loop=False)
        
        self.flif2 = FcLifPlus(256, nclass, mode=fmode, soma_params=soma_params, input_accum=input_accum,
                            rev_volt=rev_volt, fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay,
                            noise=noise,spike_func=None,use_inner_loop=False)
        assert amode == 'mean'
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
            x0 = self.norm(x0)
            x1 = self.clif1(x0)
            x2 = self.clif2(x1)
            x3 = self.clif3(x2)
            x4 = self.flat(x3)
            x5 = self.flif1(x4)
            x6 = self.flif2(x5)
            x6 = x6.unsqueeze(2).unsqueeze(3)
            self.tempAdd = load_kernel(self.tempAdd, f'tempAdd',init_zero_use_data=x6)
            self.tempAdd = self.tempAdd + x6 / self.timestep
            output = self.tempAdd.clone()
            save_kernel(self.tempAdd, f'tempAdd')
            return output.squeeze(-1).squeeze(-1)
        else:
            t = xis.size(1)
            xo = 0
            for i in range(t):
                x0 = xis[:, i, ...]
                x0 = self.norm(x0)
                if i == 0: self.clif1.reset(x0)
                x1 = self.clif1(x0)
                if i == 0: self.clif2.reset(x1)
                x2 = self.clif2(x1)
                if i == 0: self.clif3.reset(x2)
                x3 = self.clif3(x2)
                x4 = self.flat(x3)
                if i == 0: self.flif1.reset(x4)
                x5 = self.flif1(x4)
                if i == 0: self.flif2.reset(x5)
                x6 = self.flif2(x5)
                xo = xo + x6 / t

            return xo
