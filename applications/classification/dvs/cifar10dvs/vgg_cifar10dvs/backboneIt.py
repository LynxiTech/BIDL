# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
from torch.nn import Flatten
import sys
sys.path.append("../")
from layers.lif import Lif2d
from layers.time_distributed import TimeDistributed
from layers.attention import TCJA, TemporalWiseAttention
from utils import globals
globals._init()


class Conv3x3Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timestep, mode, noise,soma_params="all_share", it_batch=1):
        super(Conv3x3Block, self).__init__()
        self.time = timestep
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.lif = Lif2d(mode=mode, noise=noise, hidden_channel=out_channels,use_inner_loop=True,soma_params=soma_params,it_batch=it_batch)
        
        self.ON_APU = globals.get_value('ON_APU')
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        # if self.ON_APU:
            # x = self.lif_apu(x, time=self.time)
        # else:
        x = self.lif(x)
        return x


class AttentionVGGNet(nn.Module):
    """For CIFAR10-DVS."""
    def __init__(self,
                 timestep=10, input_channels=2, h=128, w=128, nclass=10, cmode='spike', noise=0):
        super().__init__()       
        
        # self.pool1 = nn.AdaptiveAvgPool2d(48)
        self.pool1 = nn.AvgPool2d(kernel_size=34, stride=2)
        self.conv1 = Conv3x3Block(2, 64, timestep, mode=cmode, noise=noise)
        self.conv2 = Conv3x3Block(64, 128, timestep, mode=cmode, noise=noise)

        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv3 = Conv3x3Block(128, 256, timestep, mode=cmode, noise=noise)
        self.conv4 = Conv3x3Block(256, 256, timestep, mode=cmode, noise=noise)
        self.twa1 = TemporalWiseAttention(T=10)

        self.pool3 = nn.AvgPool2d(2, 2)

        self.conv5 = Conv3x3Block(256, 512, timestep, mode=cmode, noise=noise)
        self.conv6 = Conv3x3Block(512, 512, timestep, mode=cmode, noise=noise)
        self.pool4 = nn.AvgPool2d(2, 2)

        self.conv7 = Conv3x3Block(512, 512, timestep, mode=cmode, noise=noise)
        self.conv8 = Conv3x3Block(512, 512, timestep, mode=cmode, noise=noise)
        self.tcja = TCJA(3, 3, 10, 512)
        self.twa2 = TemporalWiseAttention(T=10)
        self.pool5 = nn.AvgPool2d(2, 2)

        self.fc = nn.Sequential(
            nn.Flatten(1), # nn.Flatten(2),
            nn.Linear(512 * 3 * 3, 10),
        )        
        self.ON_APU = globals.get_value('ON_APU')
        

    def forward(self, x):
        
        if not self.ON_APU:
            x = x.transpose(0,1)
        T, N, C, H, W = x.size()
        if x.is_contiguous():
            x = x.view(N*T, C, H, W)
        else:
            x = x.reshape(N*T, C, H, W)
   
        x = self.pool1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.twa1(x)  
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.tcja(x)  
        x = self.twa2(x)  
        x = self.pool5(x)
        x = self.fc(x)
        
        x = x.view(T, N, -1)  
        x = x.mean(dim=0)  
  
        return x
    
    
