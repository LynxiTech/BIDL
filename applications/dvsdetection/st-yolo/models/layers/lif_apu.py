# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

from uuid import uuid1
import torch
import torch as pt
import torch.nn as nn
from torch import ops
from typing import Optional, Tuple
import sys
sys.path.append("../../../../../")
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel

spike_func = lambda _: torch.gt(_, 0.).to(_.dtype)

MEMB_MODE = (0, pt.relu)
SOMA_PARAMS = {'alpha': .3, 'beta': 0., 'theta': .5, 'v_0': 0., 'shape': [], 'learn': False}


class Lif(nn.Module):
    def __init__(self,
                 norm: callable,  # state normalizer
                 mode: str,  # spike | analog
                 memb_mode: tuple,  # (0~3, pt.relu|sigmoid|tanh), valid when ``mode==analog``
                 soma_params: dict,  # {alpha, beta, theta, v_0, shape, learn}
                 noise: float,  # noise scale in training
                 on_apu: bool  # for lyngor apu
                 ):
        super(Lif, self).__init__()
        assert mode in ['spike', 'analog']
        assert len(memb_mode) == 2 and memb_mode[0] in range(4)

        self.norm = norm
        self.mode = mode
        self.memb_mode = memb_mode

        shape = soma_params['shape']
        assert isinstance(shape, (list, tuple))
        learn = soma_params['learn']
        func = lambda _: nn.Parameter(pt.randn(shape) if _ is None else (pt.ones(shape) * _), requires_grad=learn)

        self.alpha = func(soma_params['alpha'])
        self.beta = func(soma_params['beta'])
        self.theta = func(soma_params['theta'])
        self.v_0 = func(soma_params['v_0'])

        self.noise = noise
        if on_apu:
            self.id = uuid1()

        self.v = None

    def forward(self, x: pt.Tensor, h_and_c_previous: Optional[Tuple[pt.Tensor, pt.Tensor]] = None) -> pt.Tensor:
        v = None
        if hasattr(self, 'id'):  # XXX for lyngor apu
            v = load(self.v.clone(), f'v{self.id}')
        else:
            v = self.v  # .clone()

        # integration
        v_upd = x + v
        # homeostasis
        if self.norm:
            v_upd = self.norm(v_upd)

        v = v_upd

        if self.training and self.noise > 0:
            self.add_noise()

        v_ = v - self.theta

        # spike firing
        fire = spike_func(v_)
        fire_inv = 1. - fire
        # reset
        v = fire * self.v_0 + fire_inv * v
        # leakage
        v = self.alpha * v + self.beta

        oupt = pt.tanh(v)

        if hasattr(self, 'id'):
            # self.v = v.clone()
            self.v = v
            save(self.v, f'v{self.id}')
        else:
            self.v = v  # .clone()

        return oupt

    def add_noise(self):
        with pt.no_grad():
            v_shape = self.v.shape
            noise = pt.randn(v_shape, dtype=self.v.dtype, device=self.v.device)
            scale = pt.std(self.v, dim=[_ for _ in range(2, len(v_shape))], keepdim=True) * self.noise
            self.v += noise * scale

    def reset(self, xi):
        self.v = pt.zeros_like(xi)


########################################################################################################################


class Lif2d(Lif):

    def __init__(self,
                 norm_state=False,
                 mode='spike', memb_mode=MEMB_MODE, soma_params=SOMA_PARAMS, noise=0, hidden_channel=None, on_apu=True
                 ):
        norm = nn.BatchNorm2d(hidden_channel) if norm_state else None
        global SOMA_PARAMS
        if soma_params == 'all_share':
            SOMA_PARAMS = SOMA_PARAMS
        elif soma_params == 'channel_share':
            SOMA_PARAMS['shape'] = [1, hidden_channel, 1, 1]
        else:
            raise NotImplemented
        super(Lif2d, self).__init__(norm, mode, memb_mode, SOMA_PARAMS, noise, on_apu)
