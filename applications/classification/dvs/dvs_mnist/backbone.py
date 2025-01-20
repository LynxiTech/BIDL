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
from layers.temporal_aggregation import Aggregation
from layers.time_distributed import TimeDistributed
from utils import globals
globals._init()



class SeqClif3Fc3DmItout(nn.Module):

    def __init__(self,
                 timestep=20, input_channels=2, h=40, w=40, nclass=10, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None,spike_func=None,use_inner_loop=False):
        super(SeqClif3Fc3DmItout, self).__init__()
        neuron=neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLif(input_channels, 32, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.clif2 = Conv2dLif(32, 64, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.clif3 = Conv2dLif(64, 128, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        assert amode == 'mean'
        self.flat = Flatten(1, -1)
        self.head = nn.Sequential(
            nn.Linear((h // 8) * (w // 8) * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, nclass)
        )
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
            x1 = self.mp1(self.clif1(x0))           
            x2 = self.mp2(self.clif2(x1))
            x3 = self.mp3(self.clif3(x2))
            x4 = self.flat(x3)
            x5 = self.head(x4)
            x5 = x5.unsqueeze(2).unsqueeze(3)
            self.reset(x5)
            if self.MULTINET:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE,init_zero_use_data=x5)
            else:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', mode=self.MODE,init_zero_use_data=x5)
            self.tempAdd = self.tempAdd + x5 / self.timestep
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
        
class SeqClifplus3Fc3DmItout(nn.Module):

    def __init__(self,
                 timestep=20, input_channels=2, h=40, w=40, nclass=10, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None):
        super(SeqClifplus3Fc3DmItout, self).__init__()
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
        assert amode == 'mean'
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
            xo_list = []
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
        
class SeqClif3Fc3DmIt(nn.Module):

    def __init__(self,
                 timestep=20, input_channels=2, h=40, w=40, nclass=10, cmode='spike', amode='mean',
                 noise=0, neuron='lif', neuron_config=None,it_batch=1):
        super(SeqClif3Fc3DmIt, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLif(input_channels, 32, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp1 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif2 = Conv2dLif(32, 64, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp2 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif3 = Conv2dLif(64, 128, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp3 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        amode = 'mean'
        self.avg = Aggregation(amode,dim=0) 
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
        self.MULTINET = globals.get_value('MULTINET')
        self.MODE = globals.get_value('MODE')
    def forward(self, x: pt.Tensor) -> pt.Tensor:
        if not self.ON_APU:
            x = x.transpose(0,1)
        s0 = x.shape  # t,b,c,h,w
        assert len(s0) in [3, 5]
        s1 = s0[0] * s0[1], *s0[2:]
        if x.is_contiguous():
            x = x.view(s1)
        else:
            x = x.reshape(s1)
        
        x1 = self.mp1(self.clif1(x))           
        x2 = self.mp2(self.clif2(x1))
        x3 = self.mp3(self.clif3(x2))
        if s0[1]!=1:
            x3 = x3.reshape(s0[0],s0[1],x3.shape[1],x3.shape[2],x3.shape[3])
            x3 = self.avg(x3)
        else:
            x3 = self.avg(x3).unsqueeze(0)
        x4 = self.flat(x3)
        x5 = self.head(x4)
        return x5
    
class SeqClif5Fc2DmItout(nn.Module):

    def __init__(self,
                 timestep=20, input_channels=2, h=128, w=128, nclass=10, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None,spike_func=None,use_inner_loop=False):
        super(SeqClif5Fc2DmItout, self).__init__()
        neuron=neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLif(input_channels, 32, 5, stride=2, padding=2, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.clif2 = Conv2dLif(32, 64, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.clif3 = Conv2dLif(64, 128, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        self.clif4 = Conv2dLif(128, 256, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params,noise=noise,spike_func=None,use_inner_loop=False)
        self.mp4 = nn.MaxPool2d(2, stride=2)
        self.clif5 = Conv2dLif(256, 512, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params,noise=noise,spike_func=None,use_inner_loop=False)
        self.mp5 = nn.MaxPool2d(2, stride=2)
        assert amode == 'mean'
        self.flat = Flatten(1, -1)
        self.head = nn.Sequential(
            nn.Linear(h // 64 * w // 64 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, nclass)
        )
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
            x1 = self.mp1(self.clif1(x0))           
            x2 = self.mp2(self.clif2(x1))
            x3 = self.mp3(self.clif3(x2))
            x4 = self.mp4(self.clif4(x3))
            x5 = self.mp5(self.clif5(x4))           
            x6 = self.flat(x5)
            x7 = self.head(x6)
            x7 = x7.unsqueeze(2).unsqueeze(3)
            self.reset(x7)
            if self.MULTINET:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE,init_zero_use_data=x7)
            else:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', mode=self.MODE,init_zero_use_data=x7)
            self.tempAdd = self.tempAdd + x7 / self.timestep
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
                if i == 0: self.clif1.reset(x0)
                x1 = self.mp1(self.clif1(x0))
                if i == 0: self.clif2.reset(x1)
                x2 = self.mp2(self.clif2(x1))
                if i == 0: self.clif3.reset(x2)
                x3 = self.mp3(self.clif3(x2))
                if i == 0: self.clif4.reset(x3)
                x4 = self.mp4(self.clif4(x3))
                if i == 0: self.clif5.reset(x4)
                x5 = self.mp5(self.clif5(x4))
                x6 = self.flat(x5)
                x7 = self.head(x6)
                xo = xo + x7 / self.timestep
            return xo

class SeqClif5Fc2DmIt(nn.Module):

    def __init__(self,
                 timestep=20, input_channels=2, h=128, w=128, nclass=10, cmode='spike', amode='mean',
                 noise=0, neuron='lif', neuron_config=None,it_batch=1):
        super(SeqClif5Fc2DmIt, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLif(input_channels, 32, 5, stride=2, padding=2, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp1 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif2 = Conv2dLif(32, 64, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp2 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif3 = Conv2dLif(64, 128, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp3 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif4 = Conv2dLif(128, 256, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp4 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif5 = Conv2dLif(256, 512, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True,it_batch=it_batch)
        self.mp5 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        amode = 'mean'
        self.avg = Aggregation(amode,dim=0) 
        self.flat = Flatten(1, -1)
        self.head = nn.Sequential(
            nn.Linear(h // 64 * w // 64 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, nclass)
        )
        self.tempAdd = None
        self.timestep = timestep
        
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        self.MULTINET = globals.get_value('MULTINET')
        self.MODE = globals.get_value('MODE')
    def forward(self, x: pt.Tensor) -> pt.Tensor:
        if not self.ON_APU:
            x = x.transpose(0,1)
        s0 = x.shape  # t,b,c,h,w
        assert len(s0) in [3, 5]
        s1 = s0[0] * s0[1], *s0[2:]
        if x.is_contiguous():
            x = x.view(s1)
        else:
            x = x.reshape(s1)
        
        x1 = self.mp1(self.clif1(x))           
        x2 = self.mp2(self.clif2(x1))
        x3 = self.mp3(self.clif3(x2))
        x4 = self.mp4(self.clif4(x3))
        x5 = self.mp5(self.clif5(x4))
        if s0[1]!=1:
            x5 = x5.reshape(s0[0],s0[1],x5.shape[1],x5.shape[2],x5.shape[3])
            x5 = self.avg(x5)
        else:
            x5 = self.avg(x5).unsqueeze(0)
        x6 = self.flat(x5)
        x7 = self.head(x6)
        return x7