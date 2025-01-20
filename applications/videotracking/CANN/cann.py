# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import hdf5storage
import torch
import numpy as np
from torch import nn, ops
import os
import sys
sys.path.append("../../../")
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel

root = os.getcwd()
for _ in range(3):
    root = os.path.dirname(root)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
ON_APU = False

class CANN_apu(nn.Module):
    def __init__(self, J=None, alpha=None, beta=None, k=0.005, prcn=15, NumOfNeRow=30, NumOfNeCol=56):
        super(CANN_apu, self).__init__()
        self.net_r = None
        self.J = J
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.NumOfNeRow = NumOfNeRow
        self.NumOfNeCol = NumOfNeCol
        self.prcn = prcn
        

        if self.J is None:
            self.J = torch.from_numpy(load_weight(root + "/weight_files/videotracking/weightMatrixAll_Convert.mat")).to(device)

        if self.alpha is None and self.beta is None:
            alpha_beta = np.zeros([3])
            alpha_beta[0], alpha_beta[1], alpha_beta[2] = 1, 1, k
            alpha_beta = torch.from_numpy(alpha_beta)
            self.alpha = alpha_beta[0]
            self.beta = alpha_beta[1]

    def reset(self, xi):
        self.net_r = torch.zeros_like(xi)

    def forward(self, net_Iext):   # net_Iext is external stimulus; net_r is firing rate
        if ON_APU:
            self.reset(net_Iext)
            self.net_r = load(self.net_r, f'net_r')
            self.J = torch.tensor(self.J, dtype=torch.float32)

        for cnt in range(1, self.prcn):
            # step_1
            net_r = self.net_r.clone()
            buf_1 = torch.reshape(net_r, [self.NumOfNeRow * self.NumOfNeCol, -1])

            J1 = torch.chunk(self.J, 4, dim=0)
            temp1 = torch.mm(J1[0], buf_1)
            temp2 = torch.mm(J1[1], buf_1)
            temp3 = torch.mm(J1[2], buf_1)
            temp4 = torch.mm(J1[3], buf_1)
            buf_1 = torch.cat([temp1, temp2, temp3, temp4], 0)

            # step_2
            temp = torch.reshape(buf_1, [self.NumOfNeRow, self.NumOfNeCol])
            net_U = self.alpha * (self.beta * temp) + self.alpha * net_Iext

            # step_3
            buf_2 = torch.reshape(net_U, [self.NumOfNeRow * self.NumOfNeCol, -1])
            buf_2 = torch.pow(0.2*buf_2, 2)

            # step_4
            net_recSum = torch.sum(self.k * buf_2)
            buf_3 = 0.04 / net_recSum

            # step_5
            net_r = torch.pow(0.2*net_U, 2) * buf_3 / 0.04
        self.net_r = net_r.clone()
        save(self.net_r, f'net_r')

        return net_U, net_recSum, net_r  # net_U is the membrane potential


class CANN(nn.Module):
    def __init__(self, J=None, alpha=None, beta=None, k=0.005, prcn=15, NumOfNeRow=30, NumOfNeCol=56):
        super(CANN, self).__init__()
        self.J = J
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.NumOfNeRow = NumOfNeRow
        self.NumOfNeCol = NumOfNeCol
        self.prcn = prcn

        if self.J is None:
            self.J = torch.from_numpy(load_weight(root + "/weight_files/videotracking/weightMatrixAll_Convert.mat")).to(device)

        if self.alpha is None and self.beta is None:
            alpha_beta = np.zeros([3])
            alpha_beta[0], alpha_beta[1], alpha_beta[2] = 1, 1, k
            alpha_beta = torch.from_numpy(alpha_beta)
            self.alpha = alpha_beta[0]
            self.beta = alpha_beta[1]

    def forward(self, net_in):   # net_Iext is external stimulus; net_r is firing rate
        net_Iext = net_in[0:1680].reshape([self.NumOfNeRow, self.NumOfNeCol])
        net_r = net_in[1680:3360].reshape([self.NumOfNeRow, self.NumOfNeCol])

        for cnt in range(1, self.prcn):
            # step_1
            if ON_APU:
                self.J = torch.tensor(self.J, dtype=torch.float32)
            buf_1 = torch.reshape(net_r, [self.NumOfNeRow * self.NumOfNeCol, -1])

            J1 = torch.chunk(self.J, 4, dim=0)
            temp1 = torch.mm(J1[0], buf_1)
            temp2 = torch.mm(J1[1], buf_1)
            temp3 = torch.mm(J1[2], buf_1)
            temp4 = torch.mm(J1[3], buf_1)
            buf_1 = torch.cat([temp1, temp2, temp3, temp4], 0)

            # step_2
            temp = torch.reshape(buf_1, [self.NumOfNeRow, self.NumOfNeCol])
            net_U = self.alpha * (self.beta * temp) + self.alpha * net_Iext

            # step_3
            buf_2 = torch.reshape(net_U, [self.NumOfNeRow * self.NumOfNeCol, -1])
            buf_2 = torch.pow(0.2*buf_2, 2)

            # step_4
            net_recSum = torch.sum(self.k * buf_2) 
            buf_3 = 0.04 / net_recSum

            # step_5
            net_r = torch.pow(0.2*net_U, 2) * buf_3 / 0.04

        return net_U, net_recSum, net_r  # net_U is the membrane potential


def load_weight(weight_path):
    weight = hdf5storage.loadmat(''.join(weight_path))["J"]
    weight.astype(np.float32)
    # print(weight)
    return weight

