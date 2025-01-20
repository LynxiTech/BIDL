'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

'''
#Warining: This model is currently in the experimental stage. Because the chip uses FP16 precision, the simulation results have limited precision, and can not be used for high-precision simulation.
'''

import numpy
import torch
from torch import ops, nn
import sys
sys.path.append("../../../")
from lynadapter.warp_load_save import load,save

class HodHux(nn.Module):
    def __init__(self, hh=0.1, on_apu=False):
        super(HodHux, self).__init__()
        self.hh = hh

        self.t_ref = 2.0  # ms
        self.g_Na = 12000.0  # nS
        self.g_K = 3600.0  # nS
        self.g_L = 30.0  # nS
        self.C_m = 100.0  # pF
        self.E_Na = 50.0  # mV
        self.E_K = -77.0  # mV
        self.E_L = -54.402  # mV
        self.tau_ex = 0.2  # ms
        self.tau_in = 2.0  # ms
        self.I_e = 10.0  # pA

        self.EPSC = 1.0 / self.tau_ex
        self.IPSC = 1.0 / self.tau_in
        self.refractory_counts = self.t_ref / self.hh
        self.ref = 0
        self.I_dc = 0

        self.gain = 1000
        self.on_apu = on_apu
        self.rng = numpy.random.default_rng(1234)

    def reset(self, xi):
        self.ref = 0 * torch.ones_like(xi)
        self.V_M = -65.0 * torch.ones_like(xi)
        # self.V_M = -65.0 * torch.tensor(self.rng.random(xi.shape), dtype=xi.dtype, device=xi.device)
        alpha_n = 0.01 * (self.V_M + 55.0) / (1.0 - torch.exp(-(self.V_M + 55.0) / 10.0))
        beta_n = 0.125 * torch.exp(-(self.V_M + 65.0) / 80.0)
        alpha_m = 0.1 * (self.V_M + 40.0) / (1.0 - torch.exp(-(self.V_M + 40.0) / 10.0))
        beta_m = 4.0 * torch.exp(-(self.V_M + 65.0) / 18.0)
        alpha_h = 0.07 * torch.exp(-(self.V_M + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + torch.exp(-(self.V_M + 35.0) / 10.0))
        self.n = alpha_n / (alpha_n + beta_n)
        self.m = alpha_m / (alpha_m + beta_m)
        self.h = alpha_h / (alpha_h + beta_h)

        self.dI_ex = torch.zeros_like(xi)
        self.I_ex = torch.zeros_like(xi)
        self.dI_in = torch.zeros_like(xi)
        self.I_in = torch.zeros_like(xi)

        self.b2 = torch.tensor(1.0 / 4.0, dtype=xi.dtype, device=xi.device)
        self.b3 = torch.tensor([3.0 / 32.0, 9.0 / 32.0], dtype=xi.dtype, device=xi.device)
        self.b4 = torch.tensor([1932.0 / 2179.0, -7200 / 2179.0, 7296.0 / 2179.0], dtype=xi.dtype, device=xi.device)
        self.b5 = torch.tensor([439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0], dtype=xi.dtype, device=xi.device)
        self.b6 = torch.tensor([-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0], dtype=xi.dtype, device=xi.device)

        self.c1 = torch.tensor([25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0], dtype=xi.dtype, device=xi.device)
        self.c2 = torch.tensor([16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0], dtype=xi.dtype, device=xi.device)

    def func(self, y):
        scale = 100
        # y=[V_M,n,m,h,dIex,Iex,dIin,Iin]
        alpha_n_tmp = -55.1 * (y[0] == -55.0).type(y[0].dtype) + y[0] * (1 - (y[0] == -55.0).type(y[0].dtype))
        # alpha_n = 0.01 * (y[0] + 55.0) / (1.0 - torch.exp(-(y[0] + 55.0) / 10.0))
        alpha_n = scale * 0.01 * (y[0] + 55.0) / (1.0 - torch.exp(-(alpha_n_tmp + 55.0) / 10.0))

        beta_n_tmp = -65.1 * (y[0] == -65.0).type(y[0].dtype) + y[0] * (1 - (y[0] == -65.0).type(y[0].dtype))
        # beta_n = 0.125 * torch.exp(-(y[0] + 65.0) / 80.0)
        beta_n = scale * 0.125 * torch.exp(-(beta_n_tmp + 65.0) / 80.0)

        alpha_m_tmp = -40.1 * (y[0] == -40.0).type(y[0].dtype) + y[0] * (1 - (y[0] == -40.0).type(y[0].dtype))
        # alpha_m = 0.1 * (y[0] + 40.0) / (1.0 - torch.exp(-(y[0] + 40.0) / 10.0))
        alpha_m = scale * 0.1 * (y[0] + 40.0) / (1.0 - torch.exp(-(alpha_m_tmp + 40.0) / 10.0))

        beta_m = scale * 4.0 * torch.exp(-(y[0] + 65.0) / 18.0)
        alpha_h = scale * 0.07 * torch.exp(-(y[0] + 65.0) / 20.0)

        beta_h_tmp = -35.1 * (y[0] == -35.0).type(y[0].dtype) + y[0] * (1 - (y[0] == -35.0).type(y[0].dtype))
        # beta_h = 1.0 / (1.0 + torch.exp(-(y[0] + 35.0) / 10.0))
        beta_h = scale * 1.0 / (1.0 + torch.exp(-(beta_h_tmp + 35.0) / 10.0))

        # I_Na = self.g_Na * y[2] * y[2] * y[2] * y[3] * (y[0] - self.E_Na)
        I_Na = self.g_Na / 1000. * y[2] * y[2] * y[2] * y[3] * (y[0] - self.E_Na)
        # I_K = self.g_K * y[1] * y[1] * y[1] * y[1] * (y[0] - self.E_K)
        I_K = self.g_K / 1000. * y[1] * y[1] * y[1] * y[1] * (y[0] - self.E_K)
        I_L = self.g_L * (y[0] - self.E_L)

        # res0 = 1.0 / self.C_m * (self.I_dc + self.I_e + y[5] + y[7] - I_K - I_Na - I_L)
        # res0 = (self.I_dc + self.I_e + y[5] + y[7] - I_L) / self.C_m - I_K / self.C_m - I_Na / self.C_m
        res0 = (self.I_dc + self.I_e + y[5] + y[7] - I_L) / self.C_m - I_K / self.C_m * 1000. - I_Na / self.C_m * 1000.
        res1 = (alpha_n * (1 - y[1]) - beta_n * y[1]) / scale
        res2 = (alpha_m * (1 - y[2]) - beta_m * y[2]) / scale
        res3 = (alpha_h * (1 - y[3]) - beta_h * y[3]) / scale
        res4 = -y[4] / self.tau_ex
        res5 = y[4] - y[5] / self.tau_ex
        res6 = -y[6] / self.tau_in
        res7 = y[6] - y[7] / self.tau_in

        return res0, res1, res2, res3, res4, res5, res6, res7

    def tuple_mul_add(self, tup1, factor, tup2):
        res0 = tup1[0] + factor * tup2[0]
        res1 = tup1[1] + factor * tup2[1]
        res2 = tup1[2] + factor * tup2[2]
        res3 = tup1[3] + factor * tup2[3]
        res4 = tup1[4] + factor * tup2[4]
        res5 = tup1[5] + factor * tup2[5]
        res6 = tup1[6] + factor * tup2[6]
        res7 = tup1[7] + factor * tup2[7]

        return res0, res1, res2, res3, res4, res5, res6, res7

    def tuple_mul(self, factor, tup):
        res0 = factor * tup[0]
        res1 = factor * tup[1]
        res2 = factor * tup[2]
        res3 = factor * tup[3]
        res4 = factor * tup[4]
        res5 = factor * tup[5]
        res6 = factor * tup[6]
        res7 = factor * tup[7]

        return res0, res1, res2, res3, res4, res5, res6, res7

    def forward(self, xi):
        if self.on_apu:
            self.ref = load(self.ref, "ref")
            self.V_M = load(self.V_M, "V_M")
            self.n = load(self.n, "n")
            self.m = load(self.m, "m")
            self.h = load(self.h, "h")
            self.dI_ex = load(self.dI_ex, "dI_ex")
            self.I_ex = load(self.I_ex, "I_ex")
            self.dI_in = load(self.dI_in, "dI_in")
            self.I_in = load(self.I_in, "I_in")

        y = (self.V_M, self.n, self.m, self.h, self.dI_ex, self.I_ex, self.dI_in, self.I_in)

        k1 = self.tuple_mul(self.hh, self.func(y))

        k2_1 = self.tuple_mul_add(y, self.b2, k1)
        k2 = self.tuple_mul(self.hh, self.func(k2_1))

        k3_1 = self.tuple_mul_add(y, self.b3[0], k1)
        k3_2 = self.tuple_mul_add(k3_1, self.b3[1], k2)
        k3 = self.tuple_mul(self.hh, self.func(k3_2))

        k4_1 = self.tuple_mul_add(y, self.b4[0], k1)
        k4_2 = self.tuple_mul_add(k4_1, self.b4[1], k2)
        k4_3 = self.tuple_mul_add(k4_2, self.b4[2], k3)
        k4 = self.tuple_mul(self.hh, self.func(k4_3))

        k5_1 = self.tuple_mul_add(y, self.b5[0], k1)
        k5_2 = self.tuple_mul_add(k5_1, self.b5[1], k2)
        k5_3 = self.tuple_mul_add(k5_2, self.b5[2], k3)
        k5_4 = self.tuple_mul_add(k5_3, self.b5[3], k4)
        k5 = self.tuple_mul(self.hh, self.func(k5_4))

        k6_1 = self.tuple_mul_add(y, self.b6[0], k1)
        k6_2 = self.tuple_mul_add(k6_1, self.b6[1], k2)
        k6_3 = self.tuple_mul_add(k6_2, self.b6[2], k3)
        k6_4 = self.tuple_mul_add(k6_3, self.b6[3], k4)
        k6_5 = self.tuple_mul_add(k6_4, self.b6[4], k5)
        k6 = self.tuple_mul(self.hh, self.func(k6_5))

        y1 = self.tuple_mul_add(y, self.c2[0], k1)
        y2 = self.tuple_mul_add(y1, self.c2[1], k2)
        y3 = self.tuple_mul_add(y2, self.c2[2], k3)
        y4 = self.tuple_mul_add(y3, self.c2[3], k4)
        y5 = self.tuple_mul_add(y4, self.c2[4], k5)
        y = self.tuple_mul_add(y5, self.c2[5], k6)

        # dI_ex[i]+=spike_ex * EPSC
        # dI_in[i]+=spike_in * IPSC

        cond = (self.ref == 0).type(torch.float) * (self.V_M >= 0).type(torch.float) * (self.V_M > y[0]).type(torch.float)
        self.ref = self.ref - (self.ref > 0).type(torch.float)
        self.ref = self.ref + cond * self.refractory_counts
        spike = cond

        self.V_M = y[0]
        self.n = y[1]
        self.m = y[2]
        self.h = y[3]
        self.dI_ex = y[4] + self.gain * xi
        self.I_ex = y[5]
        self.dI_in = y[6]
        self.I_in = y[7]
        # I_dc=currents

        if self.on_apu:
            self.ref = save(self.ref, "ref")
            self.V_M = save(self.V_M, "V_M")
            self.n = save(self.n, "n")
            self.m = save(self.m, "m")
            self.h = save(self.h, "h")
            self.dI_ex = save(self.dI_ex, "dI_ex")
            self.I_ex = save(self.I_ex, "I_ex")
            self.dI_in = save(self.dI_in, "dI_in")
            self.I_in = save(self.I_in, "I_in")

        return spike
