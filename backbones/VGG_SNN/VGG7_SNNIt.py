# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
from torch.nn import Flatten
import sys
sys.path.append("../")
from layers.lif import Conv2dLif,FcLif
from layers.temporal_aggregation import Aggregation
from layers.time_distributed import TimeDistributed
from utils import globals
globals._init()


class VGG7_SNNIt(nn.Module):

    def __init__(self,
                 timestep=20, input_channels=2, h=128, w=128, nclass=10, cmode='spike', amode='mean',
                 noise=0, neuron='lif', neuron_config=None):
        super(VGG7_SNNIt, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.clif1 = Conv2dLif(input_channels, 32, 5, stride=2, padding=2, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True)
        self.mp1 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif2 = Conv2dLif(32, 64, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True)
        self.mp2 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif3 = Conv2dLif(64, 128, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True)
        self.mp3 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif4 = Conv2dLif(128, 256, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True)
        self.mp4 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        self.clif5 = Conv2dLif(256, 512, 3, stride=1, padding=1, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True)
        self.mp5 = TimeDistributed(nn.MaxPool2d(2, stride=2))
        amode = 'mean'
        self.avg = Aggregation(amode,dim=0) 
        self.flat = Flatten(1, -1)
        self.flif1 = FcLif(h // 64 * w // 64 * 512, 512, mode=cmode,noise=noise,spike_func=None,use_inner_loop=True)
        self.flif2 = FcLif(512, nclass, mode=cmode, noise=noise,spike_func=None,use_inner_loop=True)
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
        x1 = self.mp1(self.clif1(x,s0[1]))           
        x2 = self.mp2(self.clif2(x1,s0[1]))
        x3 = self.mp3(self.clif3(x2,s0[1]))
        x4 = self.mp4(self.clif4(x3,s0[1]))
        x5 = self.mp5(self.clif5(x4,s0[1]))
        if s0[1]!=1:
            x5 = x5.reshape(s0[0],s0[1],x5.shape[1],x5.shape[2],x5.shape[3])
            x5 = self.avg(x5)
        else:
            x5 = self.avg(x5).unsqueeze(0)
        x6 = self.flat(x5)
        x7 = self.flif1(x6)
        x8 = self.flif2(x7)
        return x8