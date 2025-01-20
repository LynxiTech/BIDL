# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
import sys
sys.path.append("../")
from layers.lif import Lif2d 
from layers.time_distributed import TimeDistributed
from utils import globals
globals._init()


class CompatOp: 

    @staticmethod
    def add(x1, x2): 
        assert type(x1) == type(x2)
        if isinstance(x1, (list, tuple)):
            x3 = [a + b for a, b in zip(x1, x2)]
        elif isinstance(x1, pt.Tensor):
            x3 = x1 + x2
        else:
            raise NotImplementedError
        return x3

    @staticmethod
    def sub(x1, x2):  
        assert type(x1) == type(x2)
        if isinstance(x1, (list, tuple)):
            x3 = [a - b for a, b in zip(x1, x2)]
        elif isinstance(x1, pt.Tensor):
            x3 = x1 - x2
        else:
            raise NotImplementedError
        return x3

    @staticmethod
    def mul(x1, x2):  
        assert type(x1) == type(x2)
        if isinstance(x1, (list, tuple)):
            x3 = [a * b for a, b in zip(x1, x2)]
        elif isinstance(x1, pt.Tensor):
            x3 = x1 * x2
        else:
            raise NotImplementedError
        return x3


class AvgPoolTimeVideo(nn.AvgPool1d):

    def __init__(self, down: int):
        super(AvgPoolTimeVideo, self).__init__(down)

    def forward(self, xis):
        """
        :param xis: (b,t,c,h,w)
        """
        b, t, c, h, w = xis.shape
        assert t % self.kernel_size[0] == 0
        x1s = xis.permute(0, 2, 3, 4, 1).reshape(-1, t)
        x2s = super(AvgPoolTimeVideo, self).forward(x1s)
        xos = x2s.reshape(b, c, h, w, -1).permute(0, 4, 1, 2, 3)
        return xos


class MaxPoolTimeVideo(nn.MaxPool1d):

    def __init__(self, down: int):
        super(MaxPoolTimeVideo, self).__init__(down)

    def forward(self, xis):
        """
        :param xis: (b,t,c,h,w)
        """
        b, t, c, h, w = xis.shape
        assert t % self.kernel_size == 0
        x1s = xis.permute(0, 2, 3, 4, 1).reshape(-1, t)
        x2s = super(MaxPoolTimeVideo, self).forward(x1s)
        xos = x2s.reshape(b, c, h, w, -1).permute(0, 4, 1, 2, 3)
        return xos


class GlobalAvgPoolVideo(nn.AdaptiveAvgPool3d):

    def __init__(self):
        super(GlobalAvgPoolVideo, self).__init__((1, 1, 1))

    def forward(self, xis):
        """
        :param xis: (b,t,c,h,w)
        """
        assert self.output_size == (1, 1, 1)
        x1s = xis.permute(1, 2, 0, 3, 4)  # t,b,c,h,w -> b,c,t,h,w
        x2s = super(GlobalAvgPoolVideo, self).forward(x1s)
        xos = x2s[:, :, 0, 0, 0]
        return xos


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
            inplanes, planes, stride=1, downsample=None, noise=0,mode='spike',hidden_channels=None,soma_params='all_share',it_batch=1
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = TimeDistributed(nn.Sequential(nn.Conv2d(inplanes, planes, 3, stride, 1), nn.BatchNorm2d(planes)))
        self.lif1 = Lif2d(mode=mode, noise=noise,use_inner_loop=True,hidden_channel=hidden_channels[0],soma_params=soma_params,it_batch=it_batch)
        self.conv2 = TimeDistributed(nn.Sequential(nn.Conv2d(planes, planes, 3, 1, 1), nn.BatchNorm2d(planes)))
        self.downsample = downsample
        self.lif2 = Lif2d(mode=mode, noise=noise,use_inner_loop=True,hidden_channel=hidden_channels[1],soma_params=soma_params,it_batch=it_batch)

    def forward(self, x):
        identity = x
        out = self.lif1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = CompatOp.add(out, identity)  # ``ADD``
        return self.lif2(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
            inplanes, planes, stride=1, downsample=None, noise=0,mode='spike', time=1,hidden_channels=None,soma_params="all_share",it_batch=1
    ):
        super(Bottleneck, self).__init__()
        _mid_ = planes // self.expansion
        self.conv1 = TimeDistributed(nn.Sequential(nn.Conv2d(inplanes, _mid_, 1, 1, 0), nn.BatchNorm2d(_mid_)))
        self.lif1 = Lif2d(mode=mode, noise=noise,use_inner_loop=True,hidden_channel=hidden_channels[0],soma_params=soma_params,it_batch=it_batch)
        self.conv2 = TimeDistributed(nn.Sequential(nn.Conv2d(_mid_, _mid_, 3, stride, 1), nn.BatchNorm2d(_mid_)))
        self.lif2 = Lif2d(mode=mode, noise=noise,use_inner_loop=True,hidden_channel=hidden_channels[1],soma_params=soma_params,it_batch=it_batch)
        self.conv3 = TimeDistributed(nn.Sequential(nn.Conv2d(_mid_, planes, 1, 1, 0), nn.BatchNorm2d(planes)))
        self.downsample = downsample
        self.lif3 = Lif2d(mode=mode, noise=noise,use_inner_loop=True,hidden_channel=hidden_channels[2],soma_params=soma_params,it_batch=it_batch)
        self.t = time
        self.ON_APU = globals.get_value('ON_APU')       
    def forward(self, x):
        identity = x
        if not self.ON_APU:
            out = self.lif1(self.conv1(x))
            out = self.lif2(self.conv2(out))
        else:
            out = self.lif1(self.conv1(x))
            out = self.lif2(self.conv2(out))
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = CompatOp.add(out, identity)  # ``ADD``
        if not self.ON_APU:
            return self.lif3(out)  
        else: 
            return self.lif3(out)    



class ResNetLif(nn.Module):
    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    time_pools = {
        'avg': AvgPoolTimeVideo,
        'max': MaxPoolTimeVideo
    }

    def __init__(self,
            depth, nclass, low_resolut=False,
            input_channels=3, stem_channels=64, base_channels=64,
            timestep=10,
            h=224,w=224,
            down_t=(4, 'max'),
            cmode='spike',
            zero_init_residual=False,
            soma_params="all_share",
            it_batch=1,
            noise=1e-3
    ):
        super(ResNetLif, self).__init__()
        self.time = timestep

        if low_resolut:  # for resolution less than 64
            self.conv = TimeDistributed(nn.Sequential(nn.Conv2d(input_channels, stem_channels, 3, 1, 1),
                nn.BatchNorm2d(stem_channels)))
        else:
            self.conv = TimeDistributed(nn.Sequential(nn.Conv2d(input_channels, stem_channels, 7, 2, 3),
                nn.BatchNorm2d(stem_channels)))
        self.lif = Lif2d(mode=cmode, noise=noise,hidden_channel=stem_channels,use_inner_loop=True,soma_params=soma_params,it_batch=it_batch)
        self.pool = nn.Sequential(self.time_pools[down_t[1]](down_t[0]) if down_t[0] > 1 else nn.Identity(),
            TimeDistributed(nn.MaxPool2d((3, 3), (2, 2), 1)))  # ``ceil_mode=True`` not needed with ``padding``
        hidden_channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]

        block, layers = self.arch_settings[depth]
        ci, co = stem_channels, base_channels * block.expansion
        self.layer1 = self._make_layer(block, ci, co, layers[0], 1, noise,cmode, self.time,hidden_channels=hidden_channels[0],soma_params=soma_params,it_batch=it_batch)
        ci, co = co, co * 2
        self.layer2 = self._make_layer(block, ci, co, layers[1], 2, noise,cmode, self.time,hidden_channels=hidden_channels[1],soma_params=soma_params,it_batch=it_batch)
        ci, co = co, co * 2
        self.layer3 = self._make_layer(block, ci, co, layers[2], 2, noise,cmode, self.time,hidden_channels=hidden_channels[2],soma_params=soma_params,it_batch=it_batch)
        ci, co = co, co * 2
        self.layer4 = self._make_layer(block, ci, co, layers[3], 2, noise,cmode, self.time,hidden_channels=hidden_channels[3],soma_params=soma_params,it_batch=it_batch)

        self.gap = GlobalAvgPoolVideo()
        self.fc = nn.Linear(co, nclass)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        if zero_init_residual:
            self.zero_init_blocks()
        self.ON_APU = globals.get_value('ON_APU')

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.module[1].weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.module[1].weight, 0)

    @staticmethod
    def _make_layer(block, ci, co, blocks, stride, noise,mode,time,hidden_channels=None,soma_params="all_share",it_batch=1):
        downsample = None
        if stride != 1 or ci != co:
            downsample = TimeDistributed(nn.Sequential(nn.AvgPool2d(stride, ceil_mode=True),
                nn.Conv2d(ci, co, 1, 1, 0), nn.BatchNorm2d(co)))
        layers = [
            block(ci, co, stride, downsample=downsample, noise=noise,mode=mode, time=time,hidden_channels=hidden_channels,soma_params=soma_params,it_batch=it_batch)
        ]
        for i in range(1, blocks):
            layers.append(block(co, co, noise=noise, time=time,hidden_channels=hidden_channels,soma_params=soma_params,it_batch=it_batch))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (b,t,c,h,w)
        """        
        if not self.ON_APU:
            x = x.transpose(0,1)
        s0 = x.shape  # t,b,c,h,w
        assert len(s0) in [3, 5]
        s1 = s0[0] * s0[1], *s0[2:]
        if x.is_contiguous():
            x = x.view(s1)
        else:
            x = x.reshape(s1)
        if not self.ON_APU:
            x = self.lif(self.conv(x))
        else:
            x1 = self.conv(x)
            x = self.lif(x1)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)       
        s2 = x.shape         
        x = x.view(s0[0], s0[1], *s2[1:])
        x = self.gap(x)        
        x = self.fc(x)       
        return x
