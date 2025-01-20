# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


import math
from uuid import uuid1

import torch as pt
import torch.nn as nn
from torch import ops

from .gradient_approx import *
from .lif import Lif, Conv2dLif
import sys
sys.path.append("../../")
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel
from utils import globals
globals._init()
device = pt.device(f'cuda' if pt.cuda.is_available() else 'cpu')

MEMB_MODE = (0, pt.tanh)
SOMA_PARAMS = {
    'epsilon': None,  # Conductance decay constant, used to describe the decay rate of input accumulation or other conductance-related processes
    'v_g': None,  # Reverse voltage constant
    'tau_recip': None,  # ∆𝑡/𝜏, where ∆𝑡 is the sampling time interval and 𝜏 is the neuronal time constant, used to calculate time-related decay or growth
    'v_0': 0.,  # Reset membrane potential, i.e., the membrane potential value of a neuron in its resting state or after reset
    'epsilon_r': None,  # Relative refractory period decay constant, describing the decay rate of conductance or response during the relative refractory period
    'v_rr': None,  # Relative refractory period reversal voltage
    'v_ar': None,  # Adaptive reversal voltage, related to the reversal voltage or threshold of neuronal adaptive mechanisms
    'q_r': None,  # Relative refractory period jump size
    'b': None,  # Spike-triggered jump size
    'epsilon_w': None,  # Adaptive decay constant, describing the decay rate of neuronal adaptive mechanisms (e.g., weight adjustment)
    'theta': .5,  # Threshold, used to determine whether a neuron fires a spike or activates
    'v_theta': None,  # Trigger voltage, a voltage value related to the threshold
    'delta_t': None,  # Sharpness factor
    'v_c': None,  # Critical voltage
    'a': None,  # Subthreshold coupling constant, describing the coupling strength between a neuron and other neurons or inputs in the subthreshold state
    'v_w': None,  # Coupling membrane potential bias constant
    'alpha': .3,  # Multiplicative leak, describing the multiplicative factor of membrane potential decay over time, usually related to leak currents
    'beta': 0.,  # Additive leak, describing the additive factor of membrane potential decay over time
    'v_leak': None,  # Linear delay constant 
    'shape': [],  # Shape of parameter tensors, used to specify their dimensions and size when initializing parameters
    'learn': False  # Flag indicating whether the parameters are learnable, used to determine whether to update these parameters during training
}

class LifPlus(nn.Module):
    cntmax = 10

    def __init__(self,
            norm: callable,  # state normalizer
            mode: str,  # spike, analog
            memb_mode: tuple,  # (0~3, pt.relu|sigmoid|tanh), valid when ``mode==analog``
            soma_params: dict,  # {alpha, beta, theta, v_0, ..., shape, learn}
            input_accum: int, rev_volt: bool,  # 0: CUB; 1: COBE; 2: COBA; with or without REV
            fire_refrac: int,  # 0: no refractory; 1: AR, 2: RR
            spike_init: int,  # 0: naive; 1: EXI; 2: QDI
            trig_current: int,  # 0: naive; 1: ADT; 2: SBT
            memb_decay: int,  # 0: naive; 1: EXD; 2: LID
            noise: float,  # # noise scale in training
            spike_func=None,
            use_inner_loop=False
    ):
        super(LifPlus, self).__init__()
        assert mode in ['spike', 'analog']
        assert len(memb_mode) == 2 and memb_mode[0] in range(4)

        assert input_accum in [0, 1, 2]
        assert fire_refrac in [0, 1, 2]
        assert spike_init in [0, 1, 2]
        assert trig_current in [0, 1, 2]
        assert memb_decay in [0, 1, 2] 

        self.norm = norm
        self.mode = mode
        self.memb_mode = memb_mode
        self.use_inner_loop=use_inner_loop
        self.input_accum = input_accum
        self.rev_volt = rev_volt
        self.fire_refrac = fire_refrac
        self.spike_init = spike_init
        self.trig_current = trig_current
        self.memb_decay = memb_decay

        if spike_func is None:  
            self.spike_func = ThreshActRectangleGrad.apply  
        else:  
            assert callable(spike_func), "spike_func must be a callable function or object."  
            self.spike_func = spike_func  
    
        shape = soma_params['shape']
        assert isinstance(shape, (list, tuple))
        learn = soma_params['learn']
        func = lambda _: nn.Parameter(pt.rand(shape) if _ is None else (pt.ones(shape) * _), requires_grad=learn)

        if self.input_accum in [1, 2]:
            self.epsilon = func(soma_params['epsilon'])
        if self.rev_volt is True:
            self.v_g = func(soma_params['v_g'])

        self.tau_recip = func(soma_params['tau_recip'])
        self.v_0 = func(soma_params['v_0'])

        if self.fire_refrac == 2:
            self.epsilon_r = func(soma_params['epsilon_r'])
            self.v_rr = func(soma_params['v_rr'])
            self.v_ar = func(soma_params['v_ar'])
            self.q_r = func(soma_params['q_r'])
        if self.fire_refrac == 2 or self.trig_current in [1, 2]:
            self.epsilon_w = func(soma_params['epsilon_w'])
            self.b = func(soma_params['b'])

        if self.spike_init != 2: 
            self.theta = func(soma_params['theta'])
        if self.spike_init in [1, 2]:
            self.v_theta = func(soma_params['v_theta'])
            if self.spike_init == 1:
                self.delta_t = func(soma_params['delta_t'])
            elif self.spike_init == 2:
                self.v_c = func(soma_params['v_c'])

        if self.trig_current == 2:
            self.a = func(soma_params['a'])
            self.v_w = func(soma_params['v_w'])

        if self.memb_decay == 0:
            self.alpha = func(soma_params['alpha'])
            self.beta = func(soma_params['beta'])
        elif self.memb_decay == 2:
            self.v_leak = func(soma_params['v_leak'])

        self.noise = noise
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        if self.ON_APU:
            self.id = uuid1()

        if self.input_accum in [1, 2]:
            self.g = pt.tensor([0]).to(device)
        if self.input_accum == 2:
            self.y = pt.tensor([0]).to(device)
        if self.fire_refrac == 2:
            self.r = pt.tensor([0]).to(device)
        if self.fire_refrac == 1:
            self.cnt = pt.tensor([0]).to(device)
        self.v = pt.tensor([0]).to(device)
        if self.trig_current in [1, 2] or self.fire_refrac == 2:
            self.w = pt.tensor([0]).to(device)

    def forward(self, xi: pt.Tensor) -> pt.Tensor:
        if self.use_inner_loop == False:
            if hasattr(self, 'id'):  
                if self.input_accum in [1, 2]:
                    g = load(self.g.clone(), f'g{self.id}',init_one_use_data=xi)
                if self.input_accum == 2:
                    y = load(self.y.clone(), f'y{self.id}',init_one_use_data=xi)
                if self.fire_refrac == 2:
                    r = load(self.r.clone(), f'r{self.id}',init_one_use_data=xi)
                if self.fire_refrac == 1:
                    cnt = load(self.cnt.clone(), f'cnt{self.id}',init_zero_use_data=xi)
                v = load(self.v.clone(), f'v{self.id}',init_zero_use_data=xi)
                if self.trig_current in [1, 2] or self.fire_refrac == 2:
                    w = load(self.w.clone(), f'w{self.id}',init_zero_use_data=xi)
            else:
                if self.input_accum in [1, 2]:
                    g = self.g
                if self.input_accum == 2:
                    y = self.y
                if self.fire_refrac == 2:
                    r = self.r
                if self.fire_refrac == 1:
                    cnt = self.cnt
                v = self.v
                if self.trig_current in [1, 2] or self.fire_refrac == 2:
                    w = self.w
        else:
            if self.input_accum in [1, 2]:
                g = self.g
            if self.input_accum == 2:
                y = self.y
            if self.fire_refrac == 2:
                r = self.r
            if self.fire_refrac == 1:
                cnt = self.cnt
            v = self.v
            if self.trig_current in [1, 2] or self.fire_refrac == 2:
                w = self.w

        inpt = xi
        last_v = v
        fact4 = 0.  

        if self.input_accum == 0:
            ...
        elif self.input_accum in [1, 2]:
            if self.input_accum == 1:
                g = (1. - self.epsilon) * g + inpt
            else:
                y = (1. - self.epsilon) * y + inpt
                g = (1. - self.epsilon) * g + math.e * self.epsilon * y
        if self.rev_volt is False:
            v_rev = 1.
        else:
            assert self.input_accum != 0
            v_rev = self.v_g - v
        if self.input_accum == 0:
            fact2 = v_rev * inpt
        elif self.input_accum in [1, 2]:
            fact2 = v_rev * g

        if self.fire_refrac == 0:
            ...
        elif self.fire_refrac in [1, 2]:
            if self.fire_refrac == 1:
                fact2 = pt.where(cnt > 0, pt.zeros_like(fact2), fact2) 
                cnt = pt.max(cnt - 1., pt.zeros_like(cnt)) 
            else:
                r = (1. - self.epsilon_r) * r
                w = (1. - self.epsilon_w) * w
                fact4 = fact4 + r * (self.v_rr - v) + w * (self.v_ar - v) 

        if self.spike_init == 0:
            thresh = self.theta
            fact3 = self.v_0 - v
        elif self.spike_init in [1, 2]:
            thresh = self.v_theta
            if self.spike_init == 1:
                f_t_ = self.v_0 - v + self.delta_t * pt.exp((v - self.theta) - self.delta_t)
            else:
                f_t_ = (self.v_0 - v) * (self.v_c - v) 
            fact3 = f_t_


        if self.trig_current == 0:
            fact4 = fact4 + 0. 
        elif self.trig_current in [1, 2]:
            if self.trig_current == 1:
                w = (1. - self.epsilon_w) * w
            else:
                w = (1. - self.epsilon_w) * w + self.tau_recip * self.a * (v - self.v_w)
            fact4 = fact4 + w 
        inpt = inpt + self.tau_recip * (fact2 + fact3) + fact4 
        if self.norm: inpt = self.norm(inpt)
        o1 = v = inpt

        if self.training and self.noise > 0:
            self.add_noise()

        # spike firing
        o2 = v_ = v - thresh
        if hasattr(self, 'id') and self.FIT:
            if self.mode == "spike":
                fire = ops.custom.cmpandfire(v.clone(), thresh)
                o3 = v = ops.custom.resetwithdecay(v.clone(), thresh, self.v_0, 1.0, 0.0)
            else:
                fire = ops.custom.cmpandfire(v.clone(), thresh)
                o3 = v = ops.custom.resetwithdecay(v.clone(), thresh, self.v_0, 1.0, 0.0)
            fire_inv = 1. - fire
        else:
            fire = self.spike_func(v_)
            fire_inv = 1. - fire
            o3 = v = fire * self.v_0 + fire_inv * v  

        # refactory
        if self.fire_refrac == 1:
            cnt = fire_inv * cnt + fire * self.cntmax 
        elif self.fire_refrac == 2:
            r = fire_inv * r + fire * (r - self.q_r)  
            w = fire_inv * w + fire * (w - self.b)  
        if self.trig_current in [1, 2]:
            if self.fire_refrac != 2:
                w = fire_inv * w + fire * (w - self.b)  


        if self.memb_decay == 0: 
            v = self.alpha * v + self.beta
        elif self.memb_decay in [1, 2]: 
            if self.memb_decay == 1:
                fact1 = self.tau_recip
                v = fact1 * (fact2 + fact3) + fact4
            else:
                fact1 = lambda _: last_v + _ - self.v_leak
                v = fact1(fact2) + fact4
        o4 = v

        # output
        if self.mode == 'spike':
            oupt = fire
        else:
            oupt = [o1, o2, o3, o4][self.memb_mode[0]]
            if self.memb_mode[1] is not None:
                oupt = self.memb_mode[1](oupt)  
        if self.input_accum in [1, 2]:
            self.g = g.clone()
        if self.input_accum == 2:
            self.y = y.clone()
        if self.fire_refrac == 2:  
            self.r = r.clone()
        if self.fire_refrac == 1:  
            self.cnt = cnt.clone()
        self.v = v.clone()
        if self.trig_current in [1, 2] or self.fire_refrac == 2:
            self.w = w.clone()

        if self.use_inner_loop == False:
            if hasattr(self, 'id'):
                if self.input_accum in [1, 2]:
                    save(self.g, f'g{self.id}')
                if self.input_accum == 2:
                    save(self.y, f'y{self.id}')
                if self.fire_refrac == 2:
                    save(self.r, f'r{self.id}')
                if self.fire_refrac == 1:
                    save(self.cnt, f'cnt{self.id}')
                save(self.v, f'v{self.id}')
                if self.trig_current in [1, 2] or self.fire_refrac == 2:
                    save(self.w, f'w{self.id}')

        return oupt

    add_noise = Lif.add_noise

    def reset(self, xi):
        if self.input_accum in [1, 2]:
            self.g = pt.ones_like(xi, requires_grad=True)
        if self.input_accum == 2:
            self.y = pt.ones_like(xi, requires_grad=True)
        if self.fire_refrac == 2:
            self.r = pt.ones_like(xi, requires_grad=True)
        if self.fire_refrac == 1:
            self.cnt = pt.zeros_like(xi, dtype=pt.float32, requires_grad=False)
        self.v = pt.zeros_like(xi, requires_grad=True)
        if self.trig_current in [1, 2] or self.fire_refrac == 2:
            self.w = pt.zeros_like(xi, requires_grad=True)


########################################################################################################################


class LifPlus1d(LifPlus):

    def __init__(self,
            norm_state: int = 0,  # ``==0``: not normalize state; ``>0``: normalize state
            mode='spike', memb_mode=MEMB_MODE, soma_params=SOMA_PARAMS,
            input_accum=0, rev_volt=False, fire_refrac=0, spike_init=0, trig_current=0, memb_decay=0,
            noise=0,spike_func=None
    ):
        norm = nn.BatchNorm1d(norm_state) if norm_state else None
        super(LifPlus1d, self).__init__(
            norm, mode, memb_mode, soma_params,
            input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay,
            noise,spike_func
        )


class LifPlus2d(LifPlus):

    def __init__(self,
            norm_state: int = 0,  # ``==0``: not normalize state; ``>0``: normalize state
            mode='spike', memb_mode=MEMB_MODE, soma_params=SOMA_PARAMS,
            input_accum=0, rev_volt=False, fire_refrac=0, spike_init=0, trig_current=0, memb_decay=0,
            noise=0,spike_func=None
    ):
        norm = nn.BatchNorm2d(norm_state) if norm_state else None
        super(LifPlus2d, self).__init__(
            norm, mode, memb_mode, soma_params,
            input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay,
            noise,spike_func
        )


########################################################################################################################


class FcLifPlus(nn.Module):

    def __init__(self,
            input_channel: int, hidden_channel: int,
            feed_back=False, norm_state=True,
            mode='spike', memb_mode=MEMB_MODE, soma_params='all_share',
            input_accum=0, rev_volt=False, fire_refrac=0, spike_init=0, trig_current=0, memb_decay=0,
            noise=1e-3,spike_func=None,use_inner_loop=False,it_batch=1
    ):
        """
            FcLifPlus_step is a fully connected Leaky Integrate-and-Fire Plus (LIFPlus) neuron implementation for a single time step.

            :param input_channel: Number of input channels
            :param hidden_channel: Number of hidden or output channels
            :param feed_back: Whether to use feedback loop (not supported at the moment)
            :param norm_state: Whether to use BatchNorm to prevent gradient explosion, cell state
            :param mode: Discrete spike/continuous membrane potential analog
            :param memb_mode: (0~3, in analog mode, can activate the membrane potential)
            :param soma_params: Specify key parameters of the cell body: alpha, beta, thresh, reset (only support 'all_share' and 'channel_share' now).
                                If necessary, please modify Specify key parameters of the cell body in SOMA_PARAMS.
            :param input_accum: Select current or conductance accumulation mode (0: CUB, 1: COBE, 2: COBA)
            :param rev_volt: Set reverse voltage (True or False)
            :param fire_refrac: Set refractory period (0: no refractory, 1: AR)
            :param spike_init: Set spike initiation mode (0: naive, 1: EXI, 2: QDI)
            :param trig_current: Set current inhibition mode (0: naive, 1: ADT, 2: SBT)
            :param memb_decay: Set membrane potential decay mode (0: naive, 1: EXD)
            :param noise: The level of Gaussian noise within the LIF neuron
            :param spike_func:Specifies the spiking and gradient surrogate function for the LIF neuron; if 'spike_func=None', the default function is used.
        """
        super(FcLifPlus, self).__init__()
        self.use_inner_loop = use_inner_loop
        self.b=it_batch
        self.process_func = self.process_with_inner_loop if self.use_inner_loop==True else self.process_without_inner_loop
        self.p0 = nn.Linear(input_channel, hidden_channel)
        assert feed_back is False, 'Not frequently used, not supported at the moment'

        norm = None
        if norm_state:
            norm = nn.BatchNorm1d(hidden_channel)

        global SOMA_PARAMS
        if soma_params == 'all_share':
            SOMA_PARAMS = SOMA_PARAMS
        elif soma_params == 'channel_share':
            SOMA_PARAMS['shape'] = [1, hidden_channel]
        else:
            raise NotImplementedError

        self.lifplus = LifPlus(
            norm, mode, memb_mode, SOMA_PARAMS, 
            input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay,
            noise,spike_func,use_inner_loop
        )
    def process_with_inner_loop(self, xi):
        x1 = self.p0(xi)       
        x2 = x1.reshape(-1,self.b,x1.shape[-3],x1.shape[-2],x1.shape[-1])
        self.lifplus.reset(x2[0,...])
            
        xo_list = [self.lifplus(x2[_,...]) for _ in range(x2.size(0))]
        xos = pt.stack(xo_list, dim=0).reshape(x2.shape[0]*x2.shape[1],x2.shape[2],x2.shape[3],x2.shape[4])  
        return xos
    
    def process_without_inner_loop(self, xi):
        x1 = self.p0(xi)  # projection
        xos = self.lifplus(x1)  # population
        return xos
    
    def forward(self, xi: pt.Tensor,b=1) -> pt.Tensor:
        xos = self.process_func(xi)
        return xos

    def reset(self, xi: pt.Tensor):
        size = [xi.size(0), self.p0.out_features]
        if self.lifplus.input_accum in [1, 2]:
            self.lifplus.g = pt.ones(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.input_accum == 2:
            self.lifplus.y = pt.ones(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.fire_refrac == 2:
            self.lifplus.r = pt.ones(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.fire_refrac == 1:
            self.lifplus.cnt = pt.zeros(size, dtype=pt.float32, device=xi.device, requires_grad=False)
        self.lifplus.v = pt.zeros(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.trig_current in [1, 2] or self.lifplus.fire_refrac == 2:
            self.lifplus.w = pt.zeros(size, dtype=xi.dtype, device=xi.device, requires_grad=True)


class Conv2dLifPlus(nn.Module):

    def __init__(self,
            input_channel: int, hidden_channel: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1,
            feed_back=False, norm_state=True,
            mode='spike', memb_mode=MEMB_MODE, soma_params='all_share',
            input_accum=0, rev_volt=False, fire_refrac=0, spike_init=0, trig_current=0, memb_decay=0,
            noise=1e-3,spike_func=None,use_inner_loop=False,it_batch=1
    ):
        """
            Conv2dLifPlus_step is a convolutional Leaky Integrate-and-Fire Plus (LIFPlus) neuron implementation for a single time step.

            :param input_channel: Number of input channels
            :param hidden_channel: Number of hidden or output channels
            :param feed_back: Whether to use feedback loop (not supported at the moment)
            :param norm_state: Whether to use BatchNorm to prevent gradient explosion, cell state
            :param mode: Discrete spike/continuous membrane potential analog
            :param memb_mode: (0~3, in analog mode, can activate the membrane potential)
            :param soma_params: Specify key parameters of the cell body: alpha, beta, thresh, reset (only support 'all_share' and 'channel_share' now).
                                If necessary, please modify Specify key parameters of the cell body in SOMA_PARAMS.
            :param input_accum: Select current or conductance accumulation mode (0: CUB, 1: COBE, 2: COBA)
            :param rev_volt: Set reverse voltage (True or False)
            :param fire_refrac: Set refractory period (0: no refractory, 1: AR)
            :param spike_init: Set spike initiation mode (0: naive, 1: EXI, 2: QDI)
            :param trig_current: Set current inhibition mode (0: naive, 1: ADT, 2: SBT)
            :param memb_decay: Set membrane potential decay mode (0: naive, 1: EXD)
            :param noise: The level of Gaussian noise within the LIF neuron
            :param spike_func:Specifies the spiking and gradient surrogate function for the LIF neuron; if 'spike_func=None', the default function is used.
        """
        super(Conv2dLifPlus, self).__init__()
        self.use_inner_loop = use_inner_loop
        self.b=it_batch
        self.process_func = self.process_with_inner_loop if self.use_inner_loop==True else self.process_without_inner_loop
        self.p0 = nn.Conv2d(input_channel, hidden_channel, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups)
        assert feed_back is False, 'Not frequently used, not supported at the moment'

        norm = None
        if norm_state:
            norm = nn.BatchNorm2d(hidden_channel)

        global SOMA_PARAMS
        if soma_params == 'all_share':
            SOMA_PARAMS = SOMA_PARAMS
        elif soma_params == 'channel_share':
            SOMA_PARAMS['shape'] = [1, hidden_channel, 1, 1]
        else:
            raise NotImplementedError
        self.lifplus = LifPlus(
            norm, mode, memb_mode, SOMA_PARAMS, 
            input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay,
            noise,spike_func,use_inner_loop
        )
    def process_with_inner_loop(self, xi):
        x1 = self.p0(xi)       
        x2 = x1.reshape(-1,self.b,x1.shape[-3],x1.shape[-2],x1.shape[-1])
        self.lifplus.reset(x2[0,...])
            
        xo_list = [self.lifplus(x2[_,...]) for _ in range(x2.size(0))]
        xos = pt.stack(xo_list, dim=0).reshape(x2.shape[0]*x2.shape[1],x2.shape[2],x2.shape[3],x2.shape[4]) 
        return xos
    
    def process_without_inner_loop(self, xi):
        x1 = self.p0(xi)  # projection
        xos = self.lifplus(x1)  # population
        return xos
    
    def forward(self, xi: pt.Tensor,b=1) -> pt.Tensor:
        xos=self.process_func(xi)
        return xos


        return xos
    def reset(self, xi: pt.Tensor):
        h2, w2 = [Conv2dLif.calc_size(_, self.p0.kernel_size[0], self.p0.padding[0], self.p0.stride[0])
            for _ in [xi.size(2), xi.size(3)]]
        size = [xi.size(0), self.p0.out_channels, h2, w2]
        if self.lifplus.input_accum in [1, 2]:
            self.lifplus.g = pt.ones(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.input_accum == 2:
            self.lifplus.y = pt.ones(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.fire_refrac == 2:
            self.lifplus.r = pt.ones(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.fire_refrac == 1:
            self.lifplus.cnt = pt.zeros(size, dtype=pt.float32, device=xi.device, requires_grad=False)
        self.lifplus.v = pt.zeros(size, dtype=xi.dtype, device=xi.device, requires_grad=True)
        if self.lifplus.trig_current in [1, 2] or self.lifplus.fire_refrac == 2:
            self.lifplus.w = pt.zeros(size, dtype=xi.dtype, device=xi.device, requires_grad=True)

########################################################################################################################

