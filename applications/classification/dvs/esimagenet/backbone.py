# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch.nn as nn
from torch.nn import Flatten
import torch
import sys
sys.path.append("../")

from layers.lif import Lif2d

sys.path.append("../../")
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel
from utils import globals
globals._init()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
            inplanes, planes, stride=1, downsample=None, noise=0, mode='spike', soma_params='all_share', hidden_channels=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, planes, 3, stride, 1), nn.BatchNorm2d(planes))
        self.lif1 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[0],spike_func=None,use_inner_loop=False)
        self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, 3, 1, 1), nn.BatchNorm2d(planes))
        self.downsample = downsample
        self.lif2 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[1],spike_func=None,use_inner_loop=False)

    def forward(self, x):
        identity = x
        out = self.lif1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.lif2(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
            inplanes, planes, stride=1, downsample=None, noise=0, mode='spike', soma_params='all_share', hidden_channels=None,
    ):
        super(Bottleneck, self).__init__()
        _mid_ = planes // self.expansion
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, _mid_, 1, 1, 0), nn.BatchNorm2d(_mid_))
        self.lif1 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[0],spike_func=None,use_inner_loop=False)
        self.conv2 = nn.Sequential(nn.Conv2d(_mid_, _mid_, 3, stride, 1), nn.BatchNorm2d(_mid_))
        self.lif2 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[1],spike_func=None,use_inner_loop=False)
        self.conv3 = nn.Sequential(nn.Conv2d(_mid_, planes, 1, 1, 0), nn.BatchNorm2d(planes))
        self.downsample = downsample
        self.lif3 = Lif2d(mode=mode, noise=noise, soma_params=soma_params, hidden_channel=hidden_channels[2],spike_func=None,use_inner_loop=False)

    def forward(self, x):
        identity = x
        out = self.lif1(self.conv1(x))
        out = self.lif2(self.conv2(out))
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.lif3(out)



class ResNetLifItout(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
            depth, nclass, low_resolut=False, timestep=8,h=224,w=224,
            input_channels=3, stem_channels=64, base_channels=64,
            down_t=(4, 'max'),
            zero_init_residual=False,
            noise=1e-3, cmode='spike', amode='mean', soma_params='all_share',norm = None
    ):
        super(ResNetLifItout, self).__init__()
        assert down_t[0] == 1

        if low_resolut:  # for resolution less than 64
            self.conv = nn.Sequential(nn.Conv2d(input_channels, stem_channels, 3, 1, 1),
                nn.BatchNorm2d(stem_channels))
        else:
            self.conv = nn.Sequential(nn.Conv2d(input_channels, stem_channels, 7, 2, 3),
                nn.BatchNorm2d(stem_channels))
        self.lif = Lif2d(mode=cmode, noise=noise, soma_params=soma_params, hidden_channel=stem_channels,spike_func=None,use_inner_loop=False)
        self.pool = nn.MaxPool2d((3, 3), (2, 2), 1)

        assert depth in [10, 18, 34, 50, 101, 152]
        if depth == 50:
            hidden_channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        elif depth in [10, 18, 34]:
            hidden_channels = [[0, 0], [0, 0], [0, 0], [0, 0]]
        elif depth in [50, 101, 152]:
            hidden_channels = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        block, layers = self.arch_settings[depth]
        ci, co = stem_channels, base_channels * block.expansion
        self.layer1 = self._make_layer(block, ci, co, layers[0], 1, noise, mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[0])
        ci, co = co, co * 2
        self.layer2 = self._make_layer(block, ci, co, layers[1], 2, noise, mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[1])
        ci, co = co, co * 2
        self.layer3 = self._make_layer(block, ci, co, layers[2], 2, noise, mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[2])
        ci, co = co, co * 2
        self.layer4 = self._make_layer(block, ci, co, layers[3], 2, noise, mode=cmode, soma_params=soma_params, hidden_channels=hidden_channels[3])
        assert amode == 'mean'
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(co, nclass)
        self.flat = Flatten(1, -1)

        if zero_init_residual:
            self.zero_init_blocks()

        #self._register_lyn_reset_hook()
        self.xout = 0.
        self.timestep = timestep

        self.norm = norm
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        self.MULTINET = globals.get_value('MULTINET')
        self.MODE = globals.get_value('MODE')
        self.tempAdd = torch.tensor([0.])
        self.tempAdd = torch.tensor([0.])
    '''
    def _register_lyn_reset_hook(self):
        for child in self.modules():
            if isinstance(child, Lif2d):  # Lif, Lif1d, Conv2dLif, FcLif...
                assert hasattr(child.lif, 'reset')
                child.register_forward_pre_hook(self.lyn_reset_hook)

    @staticmethod
    def lyn_reset_hook(m, xi: tuple):
        assert isinstance(xi, tuple) and len(xi) == 1
        xi = xi[0]
        if not hasattr(m, 'lyn_cnt'):
            setattr(m, 'lyn_cnt', 0)
        if m.lyn_cnt == 0:
            # print(m)
            m.lif.reset(xi)
            m.lyn_cnt += 1
        else:
            m.lyn_cnt += 1
    '''
    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.module[1].weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.module[1].weight, 0)

    @staticmethod
    def _make_layer(block, ci, co, blocks, stride, noise, mode='spike', soma_params='all_share', hidden_channels=None):
        downsample = None
        if stride != 1 or ci != co:
            downsample = nn.Sequential(nn.AvgPool2d(stride, ceil_mode=True),
                nn.Conv2d(ci, co, 1, 1, 0), nn.BatchNorm2d(co))
        layers = [
            block(ci, co, stride, downsample=downsample, noise=noise, mode=mode, soma_params=soma_params, hidden_channels=hidden_channels)
        ]
        for i in range(1, blocks):
            layers.append(block(co, co, noise=noise, mode=mode, soma_params=soma_params, hidden_channels=hidden_channels))
        return nn.Sequential(*layers)

    def reset(self, xi):
        self.tempAdd = torch.zeros_like(xi)

    def forward(self, xis):
        if self.ON_APU:
            assert len(xis.shape) == 4
            if self.norm:
                xis = xis - torch.tensor(self.norm["mean"])[:,None,None]
                xis = xis / torch.tensor(self.norm["std"])[:,None,None]
            x0 = self.lif(self.conv(xis))   
            x0 = self.pool(x0)      
            x1 = self.layer1(x0)     
 
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)            
            x5 = self.gap(x4)
            x5 = self.flat(x5)
            x6 = self.fc(x5)      
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
            x5s = []
            for t in range(xis.size(1)):
                xi = xis[:, t, ...]
                x0 = self.lif(self.conv(xi))
                x0 = self.pool(x0)

                x1 = self.layer1(x0)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)

                x5 = self.gap(x4)
                x5s.append(x5)

            xo = (sum(x5s) / len(x5s))[:, :, 0, 0]

            xo = self.fc(xo)

            self._reset_lyn_cnt()

            return xo



    def _reset_lyn_cnt(self):
        for child in self.modules():
            if hasattr(child, 'lyn_cnt'):
                child.lyn_cnt = 0
