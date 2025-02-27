# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

from uuid import uuid1
import torch as pt
import torch.nn as nn
from torch import ops
from .gradient_approx import ThreshActRectangleGrad
from .stdp import RSTDPLeanrner,Hebbian
import math
import sys
sys.path.append("../../")
from lynadapter.warp_load_save import load,save,load_kernel,save_kernel
from utils import globals
globals._init()
spike_func = ThreshActRectangleGrad.apply

MEMB_MODE = (0, pt.relu)
SOMA_PARAMS = {'alpha': .3, 'beta': 0., 'theta': .5, 'v_0': 0., 'shape': [], 'learn': False}


device = pt.device(f'cuda' if pt.cuda.is_available() else 'cpu')

class Lif(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    """
    def __init__(self,
                 norm: callable,  # state normalizer
                 mode: str,  # spike | analog
                 memb_mode: tuple,  # (0~3, pt.relu|sigmoid|tanh), valid when ``mode==analog``
                 soma_params: dict,  # {alpha, beta, theta, v_0, shape, learn}
                 noise: float,  # noise scale in training
                 spike_func=None,
                 use_inner_loop=False
                 ):
        super(Lif, self).__init__()
        assert mode in ['spike', 'analog']
        assert len(memb_mode) == 2 and memb_mode[0] in range(4)

        self.norm = norm
        self.mode = mode
        self.memb_mode = memb_mode
        self.use_inner_loop=use_inner_loop
        if spike_func is None:  
            self.spike_func = ThreshActRectangleGrad.apply  
        else:  
            assert callable(spike_func), "spike_func must be a callable function or object."  
            self.spike_func = spike_func   
        shape = soma_params['shape']
        assert isinstance(shape, (list, tuple))
        learn = soma_params['learn']
        func = lambda _: nn.Parameter(pt.randn(shape) if _ is None else (pt.ones(shape) * _), requires_grad=learn)

        self.alpha = func(soma_params['alpha'])
        self.beta = func(soma_params['beta'])
        self.theta = func(soma_params['theta'])
        self.v_0 = func(soma_params['v_0'])
        self.noise = noise
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')
        self.MULTINET = globals.get_value('MULTINET')
        self.MODE = globals.get_value('MODE')
        if self.MODE is None:
            self.MODE = 0
        if self.ON_APU:
            self.id = uuid1()

        self.v = pt.tensor([0]).to(device)
    
    def forward(self, xi: pt.Tensor) -> pt.Tensor:
        if self.use_inner_loop == False:
            if hasattr(self, 'id'):  
                if self.MULTINET:
                    v = load(self.v.clone(), f'v{self.id}', uselookup=True, mode=self.MODE,init_zero_use_data=xi)
                else:
                    v = load(self.v.clone(), f'v{self.id}', mode=self.MODE,init_zero_use_data=xi)
            else:
                v = self.v
        else:
            v = self.v
        # integration
        v = xi + v
        # homeostasis
        if self.norm: v = self.norm(v)
        o1 = v
        if self.training and self.noise > 0:
            self.v = v.clone()
            self.add_noise()
            v=self.v
        o2 = v - self.theta

        if hasattr(self, 'id') and self.FIT:  
            # reset & leakage
            self.v = v          
            if self.mode == "spike":
                fire = ops.custom.cmpandfire(self.v.clone(), self.theta)                
                o3 = o4 = v = ops.custom.resetwithdecay(self.v.clone(), self.theta, self.v_0, self.alpha, self.beta)
            else:   
                      
                o3 = v = ops.custom.resetwithdecay(self.v.clone(), self.theta, self.v_0, 1.0, 0.0)
                o4 = v = self.alpha * v + self.beta
        else:     
            # spike firing
            fire = spike_func(o2)
            fire_inv = 1. - fire
            # reset
            o3 = v = fire * self.v_0 + fire_inv * v
            # leakage
            o4 = v = self.alpha * v + self.beta

        if self.mode == 'spike':
            oupt = fire
        else:
            oupt = [o1, o2, o3, o4][self.memb_mode[0]]
            if self.memb_mode[1] is not None:
                oupt = self.memb_mode[1](oupt)        

        self.v = v

        if self.use_inner_loop == False:
            if hasattr(self, 'id'):
                if self.MULTINET:
                    save(self.v.clone(), f'v{self.id}', uselookup=True, mode=self.MODE)
                else:
                    save(self.v.clone(), f'v{self.id}', mode=self.MODE)
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


class Lif1d(nn.Module):

    def __init__(self,
                 norm_state: int = 0,  # ``==0``: not normalize state; ``>0``: normalize state
                 mode='spike', memb_mode=MEMB_MODE, soma_params='all_share', noise=0
                 ,spike_func=None,use_inner_loop=False,it_batch=1):
        super(Lif1d, self).__init__()
        self.use_inner_loop = use_inner_loop
        self.b=it_batch
        self.process_func = self.process_with_inner_loop if self.use_inner_loop==True else self.process_without_inner_loop
        norm = nn.BatchNorm1d(norm_state) if norm_state else None
        self.lif = Lif(norm, mode, memb_mode, SOMA_PARAMS, noise,spike_func,use_inner_loop)

    def process_with_inner_loop(self, xi):
        x2 = xi.reshape(-1,self.b,xi.shape[-3],xi.shape[-2],xi.shape[-1])
        self.lif.reset(x2[0,...])
        xo_list = [self.lif(x2[_,...]) for _ in range(x2.size(0))]
        xos = pt.stack(xo_list, dim=0).reshape(x2.shape[0]*x2.shape[1],x2.shape[2],x2.shape[3],x2.shape[4])  
        return xos
    
    def process_without_inner_loop(self, xi):
        xos = self.lif(xi)
        return xos
    
    def forward(self, xi: pt.Tensor) -> pt.Tensor:

        xos=self.process_func(xi)

        return xos

class Lif2d(nn.Module):

    def __init__(self,
                 norm_state=False,
                 mode='spike', memb_mode=MEMB_MODE, soma_params='all_share', noise=0, hidden_channel=None,
                 spike_func=None,use_inner_loop=False,it_batch=1):
        super(Lif2d, self).__init__()
        self.use_inner_loop = use_inner_loop
        self.b=it_batch
        self.process_func = self.process_with_inner_loop if self.use_inner_loop==True else self.process_without_inner_loop

        norm = nn.BatchNorm2d(hidden_channel) if norm_state else None

        global SOMA_PARAMS
        if soma_params == 'all_share':
            SOMA_PARAMS = SOMA_PARAMS
        elif soma_params == 'channel_share':
            SOMA_PARAMS['shape'] = [1, hidden_channel, 1, 1]
        else:
            raise NotImplementedError
        
        self.lif = Lif(norm, mode, memb_mode, SOMA_PARAMS, noise,spike_func,use_inner_loop)
    def process_with_inner_loop(self, xi):    
        x2 = xi.reshape(-1,self.b,xi.shape[-3],xi.shape[-2],xi.shape[-1])
        self.lif.reset(x2[0,...])
        xo_list = [self.lif(x2[_,...]) for _ in range(x2.size(0))]
        xos = pt.stack(xo_list, dim=0).reshape(x2.shape[0]*x2.shape[1],x2.shape[2],x2.shape[3],x2.shape[4])  
        return xos
    
    def process_without_inner_loop(self, xi):
        xos = self.lif(xi)
        return xos
    
    def forward(self, xi: pt.Tensor) -> pt.Tensor:

        xos=self.process_func(xi)

        return xos


########################################################################################################################


class FcLif(nn.Module):

    def __init__(self,
                 input_channel: int, hidden_channel: int,
                 feed_back=False, norm_state=True,
                 mode='spike', memb_mode=MEMB_MODE, soma_params='all_share',
                 noise=0,
                 spike_func=None,
                 use_inner_loop=False,
                 it_batch=1
                 ):
        """
            FcLif_step is a fully connected Leaky Integrate-and-Fire (LIF) neuron implementation for a single time step.

            :param input_channel: Number of input channels
            :param hidden_channel: Number of hidden or output channels
            :param feed_back: Whether to use feedback loop (not supported at the moment)
            :param norm_state: Whether to use BatchNorm to prevent gradient explosion, cell state
            :param mode: Discrete spike/continuous membrane potential analog
            :param memb_mode: (0~3, in analog mode, can activate the membrane potential)
            :param soma_params: Specify key parameters of the cell body: alpha, beta, thresh, reset (only support 'all_share' and 'channel_share' now).
                                If necessary, please modify Specify key parameters of the cell body in SOMA_PARAMS.
            :param noise: The level of Gaussian noise within the LIF neuron
            :param spike_func:Specifies the spiking and gradient surrogate function for the LIF neuron; if 'spike_func=None', the default function is used.
        """
        super(FcLif, self).__init__()
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

        self.lif = Lif(norm, mode, memb_mode, SOMA_PARAMS, noise,spike_func,use_inner_loop)
    
    def process_with_inner_loop(self, xi):
        x1 = self.p0(xi)       
        x2 = x1.reshape(-1,self.b,x1.shape[-3],x1.shape[-2],x1.shape[-1])
        self.lif.reset(x2[0,...])
            
        xo_list = [self.lif(x2[_,...]) for _ in range(x2.size(0))]
        xos = pt.stack(xo_list, dim=0).reshape(x2.shape[0]*x2.shape[1],x2.shape[2],x2.shape[3],x2.shape[4])  
        return xos
    
    def process_without_inner_loop(self, xi):
        x1 = self.p0(xi)  # projection
        xos = self.lif(x1)  # population
        return xos
    
    def forward(self, xi: pt.Tensor) -> pt.Tensor:
        xos = self.process_func(xi)
        return xos

    def reset(self, xi: pt.Tensor):
        self.lif.v = pt.zeros([xi.size(0), self.p0.out_features], dtype=xi.dtype, device=xi.device,
                              requires_grad=True)

class AdaptiveFcLif(nn.Module):

    def __init__(self,
                 input_channel: int, hidden_channel: int,
                 feed_back=False, norm_state=True,
                 mode='spike', memb_mode=MEMB_MODE, soma_params='all_share',
                 noise=0, class_num=0,neurons_num=0,init_cfg=None
                 ):
        """
        :param input_channel: Number of input channels
        :param hidden_channel: Number of hidden or output channels
        :param feed_back: Whether to use feedback loop
        :param norm_state: Whether to use BatchNorm to prevent gradient explosion, cell state
        :param mode: Discrete spike/continuous membrane potential analog
        :param memb_mode: (0~3, in analog mode, can activate the membrane potential)
        :param soma_params: Specify key parameters of the cell body: alpha, beta, thresh, reset
        :param noise:
        """
        super(AdaptiveFcLif, self).__init__()
        self.p0 = nn.Linear(input_channel, hidden_channel,bias=False)
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
            raise NotImplemented
        self.ON_APU = globals.get_value('ON_APU')
        self.lif = Lif(norm, mode, memb_mode, SOMA_PARAMS, noise)    
        
        self.w = self.p0.weight.data.t()
        self.plasticity_rule(rule = "stdp",nu=(0.0001,0.0010),class_num=class_num,neurons_num=neurons_num)   
        
        
    
    def plasticity_rule(self,rule,nu,class_num,neurons_num):
        if rule == "stdp":
            self.rule = RSTDPLeanrner(nu=nu,class_num=class_num,neurons_num=neurons_num,ON_APU=self.ON_APU)#,norm=30.0)
        elif rule == "Hebb":
            self.rule = Hebbian(weight=self.w,nu=nu,ON_APU=self.ON_APU)
        

    def forward(self,xi:pt.Tensor,  gt_label:pt.Tensor,finetune:pt.Tensor) -> pt.Tensor:
        if self.ON_APU:
            weight = load(self.w.clone(), f'#w')    
        else:
            weight = self.w.clone()  
        x1 = pt.mm(xi, weight)  
        x2 = self.lif(x1)
        delta_w  = self.rule.update(xi,x2,gt_label)
        weight = weight + delta_w*finetune
        self.w = weight.clone()
        if self.ON_APU:
            save(self.w, f'#w')
        return x2

    def reset(self, xi: pt.Tensor):
        self.lif.v = pt.zeros([xi.size(0), self.p0.out_features], dtype=xi.dtype, device=xi.device,
                              requires_grad=False)  

        self.w = self.w.to(xi.device)


class AdaptiveFcLif_Scale(nn.Module):

    def __init__(self,
                 input_channel: int, hidden_channel: int,
                 feed_back=False, norm_state=True,
                 mode='spike', memb_mode=MEMB_MODE, soma_params='all_share',
                 noise=0, class_num=0,neurons_num=0,init_cfg=None
                 ):
        """
        :param input_channel: Number of input channels
        :param hidden_channel: Number of hidden or output channels
        :param feed_back: Whether to use feedback loop
        :param norm_state: Whether to use BatchNorm to prevent gradient explosion, cell state
        :param mode: Discrete spike/continuous membrane potential analog
        :param memb_mode: (0~3, in analog mode, can activate the membrane potential)
        :param soma_params: Specify key parameters of the cell body: alpha, beta, thresh, reset
        :param noise:
        """
        super(AdaptiveFcLif_Scale, self).__init__()
        self.p0 = nn.Linear(input_channel, hidden_channel,bias=False)
        assert feed_back is False, 'Not frequently used, not supported at the moment.'
        norm = None
        if norm_state:
            norm = nn.BatchNorm1d(hidden_channel)

        global SOMA_PARAMS
        if soma_params == 'all_share':
            SOMA_PARAMS = SOMA_PARAMS
        elif soma_params == 'channel_share':
            SOMA_PARAMS['shape'] = [1, hidden_channel]
        else:
            raise NotImplemented      

 
        nn.init.uniform_(self.p0.weight,a=-1.0,b =1.0) 
        self.ON_APU = globals.get_value('ON_APU')

        self.lif = Lif(norm, mode, memb_mode, SOMA_PARAMS, noise)        
        
        self.w = self.p0.weight.data.t()*5000.
        self.plasticity_rule(rule = "stdp",nu=(0.0001,0.0013),class_num=class_num,neurons_num=neurons_num)   
        
     
    
    def plasticity_rule(self,rule,nu,class_num,neurons_num):

        if rule == "stdp":
            self.rule = RSTDPLeanrner(nu=nu,class_num=class_num,neurons_num=neurons_num,ON_APU=self.ON_APU)#,norm=30.0)
        elif rule == "Hebb":
            self.rule = Hebbian(weight=self.w,nu=nu,ON_APU=self.ON_APU)
        

    def forward(self,xi:pt.Tensor,  gt_label:pt.Tensor,finetune:pt.Tensor) -> pt.Tensor:
        if self.ON_APU:
            weight = load(self.w.clone(), f'#w')          

        else:
            weight = self.w.clone()
        
        weight_scale = weight/5000.
        x1 = pt.mm(xi, weight_scale)  
        x2 = self.lif(x1)


        delta_w  = self.rule.update_scale(xi,x2,gt_label)
        weight = weight + delta_w*finetune
        weight = pt.clamp(weight, max=65500,min=-65500)
        self.w = weight.clone()    
        
        if self.ON_APU:
            save(self.w, f'#w')
        return x2


    def reset(self, xi: pt.Tensor):
        self.lif.v = pt.zeros([xi.size(0), self.p0.out_features], dtype=xi.dtype, device=xi.device,
                              requires_grad=False)  

        self.w = self.w.to(xi.device)

class Conv2dLif(nn.Module):

    def __init__(self,
                 input_channel: int, hidden_channel: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1,
                 feed_back=False, norm_state=True,
                 mode='spike', memb_mode=MEMB_MODE, soma_params='all_share',
                 noise=0,
                 mp=False,
                 spike_func=None,
                 use_inner_loop=False,
                 it_batch=1
                 ):
        """
        Conv2dLif_step is a convolutional Leaky Integrate-and-Fire (LIF) neuron implementation for a single time step.

        :param input_channel: Number of input channels
        :param hidden_channel: Number of hidden or output channels
        :param feed_back: Whether to use feedback loop (not supported at the moment)
        :param norm_state: Whether to use BatchNorm to prevent gradient explosion, cell state
        :param mode: Discrete spike/continuous membrane potential analog
        :param memb_mode: (0~3, in analog mode, can activate the membrane potential)
        :param soma_params: Specify key parameters of the cell body: alpha, beta, thresh, reset (only support 'all_share' and 'channel_share' now)
                                If necessary, please modify Specify key parameters of the cell body in SOMA_PARAMS
        :param noise: The level of Gaussian noise within the LIF neuron
        :param mp:Determines whether to incorporate a max pooling layer into this layer
        :param spike_func:Specifies the spiking and gradient surrogate function for the LIF neuron; if 'spike_func=None', the default function is used.
        """
        super(Conv2dLif, self).__init__()
        self.use_inner_loop=use_inner_loop
        self.b=it_batch
        self.process_func = self.process_with_inner_loop if self.use_inner_loop==True else self.process_without_inner_loop
        self.p0 = nn.Conv2d(input_channel, hidden_channel, kernel_size, stride=stride, padding=padding,
                            dilation=dilation, groups=groups)
        self.mp = mp
        if self.mp== True:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()
        assert feed_back is False, 'Not frequently used, not supported at the moment.'

        norm = None
        if norm_state:
            norm = nn.BatchNorm2d(hidden_channel)
            self.norm=nn.Identity()
        else:
            self.norm=nn.BatchNorm2d(hidden_channel)
        global SOMA_PARAMS
        if soma_params == 'all_share':
            SOMA_PARAMS = SOMA_PARAMS
        elif soma_params == 'channel_share':
            SOMA_PARAMS['shape'] = [1, hidden_channel, 1, 1]
        else:
            raise NotImplementedError

        self.lif = Lif(norm, mode, memb_mode, SOMA_PARAMS, noise,spike_func,use_inner_loop)
    
    def process_with_inner_loop(self, xi):
        x1 = self.p0(xi)
        x2 = self.pool(x1)
        x2 = self.norm(x2)
        x3 = x2.reshape(-1, self.b, x2.shape[-3], x2.shape[-2], x2.shape[-1])
        self.lif.reset(x3[0,...])
        
        xo_list = [self.lif(x3[_,...]) for _ in range(x3.size(0))]
        xos = pt.stack(xo_list, dim=0).reshape(x3.shape[0]*x3.shape[1], x3.shape[2], x3.shape[3], x3.shape[4])
        return xos
    
    def process_without_inner_loop(self, xi):
        x1 = self.p0(xi)
        x2 = self.pool(x1)
        x2 = self.norm(x2)
        xos = self.lif(x2)
        return xos
    
    def forward(self, xi: pt.Tensor) -> pt.Tensor:

        xos=self.process_func(xi)

        return xos

    @staticmethod
    def calc_size(h, k, p, s):
        return (h - k + 2 * p) // s + 1

    def reset(self, xi):
        h2, w2 = [self.calc_size(_, self.p0.kernel_size[0], self.p0.padding[0], self.p0.stride[0])
                  for _ in [xi.size(2), xi.size(3)]]
        if self.mp == True:
            self.lif.v = pt.zeros([xi.size(0), self.p0.out_channels,  math.floor((h2 - 2) / 2) + 1, math.floor((w2 - 2) / 2) + 1], dtype=xi.dtype, device=xi.device,
                                    requires_grad=True)  
        else:
            self.lif.v = pt.zeros([xi.size(0), self.p0.out_channels,  h2, w2] , dtype=xi.dtype, device=xi.device,
                                    requires_grad=True)


########################################################################################################################


