import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, ops
from lynadapter.warp_load_save import load_kernel, save_kernel

thresh = 0.15
lens = 0.5
probs = 0.5
decay = 0.6

cfg_cnn = [(2, 64, 3, 1, 3),
           (64, 128, 2, 1, 3),
           (128, 256, 2, 1, 3),
            (256, 256, 1, 1, 3),
           ]
cfg_fc = [512, 512, 200]

def mem_update(opts, x, mem, spike):
    mem = mem * decay * (1 - spike) + opts(x)
    spike = act_fun(mem - thresh)
    return mem, spike

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input)<lens
        return grad_input * temp.float()
act_fun = ActFun.apply


def mem_update(opts, x, mem, spike):
    mem = mem * decay * (1 - spike) + opts(x)
    spike = act_fun(mem - thresh)
    return mem, spike


def mem_update_compile(opts, x, mem, spike):
    mem = (mem * decay * (1 - spike) + opts(x))
    spike = (mem - thresh).gt(0).float()
    return mem, spike


class SNN(nn.Module):
    def __init__(self, **kwargs):
        super(SNN, self).__init__()
        self.device = kwargs.get('device')

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_planes)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(in_planes)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn4 = nn.BatchNorm2d(in_planes)

        inp_dim = 5 * 7 * out_planes
        self.fc1 = nn.Linear(inp_dim, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

        self.cmp = False

    def reset(self):
        batch_size = 1
        self.device = 'cpu'
        self.c1_mem = self.c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], 44, 58, device=self.device, dtype=torch.float32)
        self.c2_mem = self.c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], 22, 29, device=self.device, dtype=torch.float32)
        self.c3_mem = self.c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], 11, 15, device=self.device, dtype=torch.float32)
        self.c4_mem = self.c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], 11, 15, device=self.device, dtype=torch.float32)

        # self.h1_sum = self.h1_mem = self.h1_spike = torch.zeros(batch_size, self.cfg_fc[0], device=self.device)
        # self.h2_sum = self.h2_mem = self.h2_spike = torch.zeros(batch_size, self.cfg_fc[1], device=self.device)
        self.h1_mem = self.h1_spike = torch.zeros(batch_size, cfg_fc[0], device=self.device, dtype=torch.float32)
        self.h2_mem = self.h2_spike = torch.zeros(batch_size, cfg_fc[1], device=self.device, dtype=torch.float32)

    def forward(self, dvs_inp, out_mode='rate'):
        if not self.cmp:
            batch_size, seq_len, channel, w, h = dvs_inp.size()
            dvs_inp = dvs_inp.permute([1,0,2,3,4])
            c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], 44, 58, device=self.device)
            c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], 22, 29, device=self.device)
            c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], 11, 15, device=self.device)
            c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], 11, 15, device=self.device)

            h1_sum = h1_mem = h1_spike = torch.zeros(batch_size, cfg_fc[0], device=self.device)
            h2_sum = h2_mem = h2_spike = torch.zeros(batch_size, cfg_fc[1], device=self.device)
            out_spike = torch.zeros(seq_len, batch_size, cfg_fc[1], device=self.device)
            for step in range(seq_len):
                spike_inp = dvs_inp[step]
                c1_mem, c1_spike = mem_update(self.conv1, spike_inp.float(), c1_mem, c1_spike)
                c2_mem, c2_spike = mem_update(self.conv2, self.bn2(c1_spike), c2_mem, c2_spike)
                c3_mem, c3_spike = mem_update(self.conv3, self.bn3(c2_spike), c3_mem, c3_spike)
                # c4_mem, c4_spike = mem_update(self.conv4, c3_spike, c4_mem, c4_spike)
                c4_mem, c4_spike = mem_update(self.conv4, self.bn4(c3_spike), c4_mem, c4_spike)
                x = F.avg_pool2d(c4_spike, 2).view(batch_size, -1)

                h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
                h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
                h2_sum += h2_spike
                out_spike[step] = h2_spike
            out = h2_sum / seq_len
            if out_mode == 'time':
                out = out_spike
            else:
                out = h2_sum / seq_len
            return out
        else:
            self.reset()

            c1_mem = load_kernel(self.c1_mem, f'c1_mem')
            c2_mem = load_kernel(self.c2_mem, f'c2_mem')
            c3_mem = load_kernel(self.c3_mem, f'c3_mem')
            c4_mem = load_kernel(self.c4_mem, f'c4_mem')
            c1_spike = load_kernel(self.c1_spike, f'c1_spike')
            c2_spike = load_kernel(self.c2_spike, f'c2_spike')
            c3_spike = load_kernel(self.c3_spike, f'c3_spike')
            c4_spike = load_kernel(self.c4_spike, f'c4_spike')
            h1_mem = load_kernel(self.h1_mem, f'h1_mem')
            h2_mem = load_kernel(self.h2_mem, f'h2_mem')
            h1_spike = load_kernel(self.h1_spike, f'h1_spike')
            h2_spike = load_kernel(self.h2_spike, f'h2_spike')


            c1_mem, c1_spike = mem_update_compile(self.conv1, dvs_inp.float(), c1_mem, c1_spike)
            c2_mem, c2_spike = mem_update_compile(self.conv2, self.bn2(c1_spike), c2_mem, c2_spike)
            c3_mem, c3_spike = mem_update_compile(self.conv3, self.bn3(c2_spike), c3_mem, c3_spike)
            # # c4_mem, c4_spike = mem_update(self.conv4, c3_spike, c4_mem, c4_spike)
            c4_mem, c4_spike = mem_update_compile(self.conv4, self.bn4(c3_spike), c4_mem, c4_spike)
            x = F.avg_pool2d(c4_spike, 2).view(1, -1)
            h1_mem, h1_spike = mem_update_compile(self.fc1, x, h1_mem, h1_spike)
            h2_mem, h2_spike = mem_update_compile(self.fc2, h1_spike, h2_mem, h2_spike)

            self.c1_mem = c1_mem.clone()
            self.c2_mem = c2_mem.clone()
            self.c3_mem = c3_mem.clone()
            self.c4_mem = c4_mem.clone()
            self.c1_spike = c1_spike.clone()
            self.c2_spike = c2_spike.clone()
            self.c3_spike = c3_spike.clone()
            self.c4_spike = c4_spike.clone()
            self.h1_mem = h1_mem.clone()
            self.h2_mem = h2_mem.clone()
            self.h1_spike = h1_spike.clone()
            self.h2_spike = h2_spike.clone()


            save_kernel(self.c1_mem, f'c1_mem')
            save_kernel(self.c2_mem, f'c2_mem')
            save_kernel(self.c3_mem, f'c3_mem')
            save_kernel(self.c4_mem, f'c4_mem')
            save_kernel(self.c1_spike, f'c1_spike')
            save_kernel(self.c2_spike, f'c2_spike')
            save_kernel(self.c3_spike, f'c3_spike')
            save_kernel(self.c4_spike, f'c4_spike')
            save_kernel(self.h1_mem, f'h1_mem')
            save_kernel(self.h2_mem, f'h2_mem')
            save_kernel(self.h1_spike, f'h1_spike')
            save_kernel(self.h2_spike, f'h2_spike')

            return h2_spike

