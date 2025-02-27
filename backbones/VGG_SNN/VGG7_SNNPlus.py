# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
from torch.nn import Flatten
import sys
sys.path.append("../")
from layers.lifplus import Conv2dLifPlus, FcLifPlus
sys.path.append("../../")
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel
from utils import globals
globals._init()

class VGG7_SNNPlus(nn.Module):

    def __init__(self,
                 timestep=20, input_channels=2, h=128, w=128, nclass=10, cmode='spike', amode='mean', soma_params='all_share',
                 noise=0, neuron='lif', neuron_config=None,spike_func=None,use_inner_loop=False):
        super(VGG7_SNNPlus, self).__init__()
        neuron=neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLifPlus(input_channels, 32, 5, stride=2, padding=2, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.clif2 = Conv2dLifPlus(32, 64, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.clif3 = Conv2dLifPlus(64, 128, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params, noise=noise,spike_func=None,use_inner_loop=False)
        self.mp3 = nn.MaxPool2d(2, stride=2)
        self.clif4 = Conv2dLifPlus(128, 256, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params,noise=noise,spike_func=None,use_inner_loop=False)
        self.mp4 = nn.MaxPool2d(2, stride=2)
        self.clif5 = Conv2dLifPlus(256, 512, 3, stride=1, padding=1, mode=cmode, soma_params=soma_params,noise=noise,spike_func=None,use_inner_loop=False)
        self.mp5 = nn.MaxPool2d(2, stride=2)
        assert amode == 'mean'
        self.flat = Flatten(1, -1)
        self.flif1 = FcLifPlus(h // 64 * w // 64 * 512, 512, mode=cmode, soma_params=soma_params,noise=noise,spike_func=None,use_inner_loop=False)
        self.flif2 = FcLifPlus(512, nclass, mode=cmode, soma_params=soma_params,noise=noise,spike_func=None,use_inner_loop=False)
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
            x7 = self.flif1(x6)
            x8 = self.flif2(x7)
            x8 = x8.unsqueeze(2).unsqueeze(3)
            if self.MULTINET:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', uselookup=True, mode=self.MODE,init_zero_use_data=x8)
            else:
                self.tempAdd = load_kernel(self.tempAdd, f'tempAdd', mode=self.MODE,init_zero_use_data=x8)
            self.tempAdd = self.tempAdd + x8 / self.timestep
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
                if i == 0: self.flif1.reset(x6)
                x7 = self.flif1(x6)
                if i == 0: self.flif2.reset(x7)
                x8 = self.flif2(x7)
                xo = xo + x8 / self.timestep
            return xo


