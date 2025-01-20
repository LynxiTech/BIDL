# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt

class TimeDistributed(pt.nn.Module):

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, xi):
        x2 = self.module(xi)
        return x2
