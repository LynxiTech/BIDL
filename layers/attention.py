# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch
import torch.nn as nn
from .base import MultiStepModule


class TCJA(nn.Module):
    def __init__(self, kernel_size_t: int = 2, kernel_size_c: int = 1, T: int = 8, channel: int = 128):
        super().__init__()

        self.conv_t = nn.Conv1d(in_channels=T, out_channels=T,
                              kernel_size=kernel_size_t, padding=1, bias=False)
        self.conv_c = nn.Conv1d(in_channels=channel, out_channels=channel,
                                kernel_size=kernel_size_c, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.T = T
        self.C = channel

    def forward(self, x_seq):
        
        # [batch_size * T, C, H, W]
        total_elements, C, H, W = x_seq.size()
        batch_size = total_elements // self.T 

        # [batch_size, T, C]
        x_avg = x_seq.mean(dim=[-2, -1]).view(batch_size, self.T, -1) 

        x_t = x_avg
        conv_t_out = self.conv_t(x_t).permute(1, 0, 2)  # [batch_size, T, C]

        x_c = x_avg.permute(0, 2, 1)
        conv_c_out = self.conv_c(x_c).permute(2, 0, 1)  # [batch_size, T, C]
        
        out = self.sigmoid(conv_c_out * conv_t_out)  # [batch_size, T, C]
        out = out.reshape(total_elements, -1)
               
        out_expanded = out[:, :, None, None]    # [batch_size*T, C, 1, 1]

        y_seq = x_seq * out_expanded  # [batch_size*T, C, H, W]

        return y_seq                    
        

class TemporalWiseAttention(nn.Module, MultiStepModule):
    def __init__(self, T: int, reduction: int = 2, dimension: int = 3):
        """
        * :ref:`API in English <MultiStepTemporalWiseAttention.__init__-en>`

        .. _MultiStepTemporalWiseAttention.__init__-cn:

        :param T: 输入数据的时间步长

        :param reduction: 压缩比

        :param dimension: 输入数据的维度。当输入数据为[T, N, C, H, W]时， dimension = 4；输入数据维度为[T, N, L]时，dimension = 2。

        `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_ 中提出
        的MultiStepTemporalWiseAttention层。MultiStepTemporalWiseAttention层必须放在二维卷积层之后脉冲神经元之前，例如：

        ``Conv2d -> MultiStepTemporalWiseAttention -> LIF``

        输入的尺寸是 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` ，经过MultiStepTemporalWiseAttention层，输出为 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` 。

        ``reduction`` 是压缩比，相当于论文中的 :math:`r`。

        * :ref:`中文API <MultiStepTemporalWiseAttention.__init__-cn>`

        .. _MultiStepTemporalWiseAttention.__init__-en:

        :param T: timewindows of input

        :param reduction: reduction ratio

        :param dimension: Dimensions of input. If the input dimension is [T, N, C, H, W], dimension = 4; when the input dimension is [T, N, L], dimension = 2.

        The MultiStepTemporalWiseAttention layer is proposed in `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_.

        It should be placed after the convolution layer and before the spiking neurons, e.g.,

        ``Conv2d -> MultiStepTemporalWiseAttention -> LIF``

        The dimension of the input is ``[T, N, C, H, W]`` or  ``[T, N, L]`` , after the MultiStepTemporalWiseAttention layer, the output dimension is ``[T, N, C, H, W]`` or  ``[T, N, L]`` .

        ``reduction`` is the reduction ratio，which is :math:`r` in the paper.

        """
        super().__init__()
        self.step_mode = 'm'
        # assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'

        self.dimension = dimension
        self.t = T
        # Sequence
        if self.dimension == 2:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        elif self.dimension == 3:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif self.dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

        assert T >= reduction, 'reduction cannot be greater than T'

        # Excitation
        self.sharedMLP = nn.Sequential(
            nn.Linear(T, T // reduction, bias=False),
            nn.ReLU(),
            # spiking_neuron(),
            nn.Linear(T // reduction, T, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        s = x_seq.shape
        
        aa = self.avg_pool(x_seq).mean(dim=1, keepdim=True).view(s[0] // self.t, self.t)   # [N, T]
        avgout = self.sharedMLP(aa)  
        # maxout = self.sharedMLP(self.max_pool(x_seq).view([x_seq.shape[0], x_seq.shape[1]]))
        # scores = self.sigmoid(avgout + maxout)
        scores = self.sigmoid(avgout) 

        if self.dimension == 2:
            y_seq = x_seq * scores[:, :, None]
        elif self.dimension == 3:
            scores = scores.view(s[0], -1)
            y_seq = x_seq * scores[:, :, None, None]
        elif self.dimension == 4:
            y_seq = x_seq * scores[:, :, None, None, None]
        
        return y_seq
        