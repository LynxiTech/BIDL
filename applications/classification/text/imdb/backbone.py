# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch as pt
import torch.nn as nn
import sys
sys.path.append("../")
from layers.lif import FcLif
from layers.lifplus import FcLifPlus
from layers.gradient_approx import *
sys.path.append("../../") 
from utils import globals
globals._init()

class FastTextItout(nn.Module):


    def __init__(self,timestep,
                 vocab_size, embedding_dim, hidden_dim, cmode='analog', amode='mean', soma_params='all_share',
                 noise=1e-3, neuron='lif', neuron_config=None,spike_func=None,use_inner_loop=False):
        super(FastTextItout, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lif']
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lif = FcLif(embedding_dim, hidden_dim, mode=cmode, noise=noise,spike_func=None,use_inner_loop=False)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.embedding_fc = nn.Linear(in_features=1001, out_features=16, bias=False)
        self.embedding_fc.weight = pt.nn.parameter.Parameter(self.embedding.weight.transpose(0, 1))
        assert amode == 'mean'
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')

    def forward(self, xis: pt.Tensor) -> pt.Tensor:
        if self.ON_APU:
            assert len(xis.shape) == 2
            x1 = self.embedding_fc(xis) 
            x2 = self.lif(x1)
            x3 = self.dropout(x2)  
            xo = self.fc(x3).squeeze(-1)  
            return xo
        else:
           
            t = xis.size(1)
            for i in range(t):
                x0 = xis[:, i].long()
                x1 = self.embedding(x0)            
                if i == 0: self.lif.reset(x1)
                x2 = self.lif(x1)                
                x3 = self.dropout(x2)             
                x4 = self.fc(x3).squeeze(-1)           
          
            return x4
        
class FastTextlifplusItout(nn.Module):

    def __init__(self,
                 timestep,
                 vocab_size, embedding_dim, hidden_dim, cmode='analog', amode='mean', soma_params='all_share',
                 noise=1e-3, neuron='lif', neuron_config=None):
        super(FastTextlifplusItout, self).__init__()
        neuron = neuron.lower()
        assert neuron in ['lifplus']
        if neuron == 'lifplus' and neuron_config != None:
            input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay = neuron_config
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim

        self.lif = FcLifPlus(embedding_dim, hidden_dim, mode=cmode, input_accum=input_accum, rev_volt=rev_volt, 
                                fire_refrac=fire_refrac, spike_init=spike_init, trig_current=trig_current, memb_decay=memb_decay, 
                                noise=noise,norm_state=True,spike_func=None,use_inner_loop=False)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.embedding_fc = nn.Linear(in_features=1001, out_features=16, bias=False)
        self.embedding_fc.weight = pt.nn.parameter.Parameter(self.embedding.weight.transpose(0, 1))
        assert amode == 'mean'
        self.ON_APU = globals.get_value('ON_APU')
        self.FIT = globals.get_value('FIT')

    def forward(self, xis: pt.Tensor) -> pt.Tensor:
        if self.ON_APU:
            assert len(xis.shape) == 2
            x1 = self.embedding_fc(xis) 
            x2 = self.lif(x1)
            x3 = self.dropout(x2) 
            xo = self.fc(x3).squeeze(-1)  
            return xo
        else:           
            t = xis.size(1)
            
            for i in range(t):              
                x0 = xis[:, i].long()                    
                x1 = self.embedding(x0)            
                if i == 0: self.lif.reset(x1)
                x2 = self.lif(x1)                
                x3 = self.dropout(x2)               
                x4 = self.fc(x3).squeeze(-1) 

            return x4