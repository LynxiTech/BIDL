'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import torch
import random
import numpy as np
from torch import ops

from .learning import STDPLearner
from uuid import uuid1
import sys
sys.path.append("../../../")
from lynadapter.warp_load_save import load,save



class Projection(torch.nn.Module):
    """
    连接前后神经元簇,设置稀疏连接比例
    sparse_ratio(默认为1不稀疏)，生成四元组
    """

    def __init__(self,
            pre_population,
            post_population,
            sparse_ratio=None,
            synapse_type=None,
            receptor_type='excitatory',
            on_apu=False,
            learning=False):
        super(Projection, self).__init__()

        self.on_apu = on_apu
        self.pre_popu = pre_population
        self.pos_popu = post_population
        self.sparse_ratio = sparse_ratio
        self.synapse_type = synapse_type
        self.receptor_type = receptor_type


        self.connect_mat = self.connection_matrix()
        self.learning = learning
        if self.learning:
            self.learner = STDPLearner(on_apu=on_apu)
            self.learner.reset(self.pre_popu.num,self.pos_popu.num)
        if self.on_apu:
            self.id = uuid1()

    def forward(self): 
        delta_w = 0.0      
        if self.on_apu:             
            weighted_spikes = load(self.pre_popu.weighted_spikes.clone(), f'spike_{self.pre_popu.pop_index}')
            if self.learning:
                weight = load(self.connect_mat.clone(), f'w{self.id}') 
                weighted_spikes_pos = load(self.pos_popu.weighted_spikes.clone(), f'spike_{self.pos_popu.pop_index}')
                delta_w = self.learner.update(weighted_spikes, weighted_spikes_pos) 
                weight = weight + delta_w 
            else:
                weight = self.connect_mat.clone()
                       
        else:
            weight = self.connect_mat.clone()
            if self.learning:            
                delta_w = self.learner.update(self.pre_popu.weighted_spikes.clone(), self.pos_popu.weighted_spikes.clone())       
                weight = weight + delta_w
       
        if self.on_apu:
            synapse_current = torch.mm(weighted_spikes, weight)            
        else:
            synapse_current = torch.mm(self.pre_popu.weighted_spikes, weight)

        if self.pre_popu.ex_inh_type == "excitatory":
            self.pos_popu.spikes_ex += synapse_current
        else:
            self.pos_popu.spikes_in += synapse_current
        
        self.connect_mat = weight.clone()
        if self.on_apu:
            if self.learning:
                save(self.connect_mat, f'w{self.id}')
                
        return delta_w



    def connection_matrix(self):
        if self.sparse_ratio >= 0 and self.sparse_ratio <= 1:
            num_pre = int(np.ceil(self.pre_popu.num * self.sparse_ratio))
            num_pos = int(np.ceil(self.pos_popu.num * self.sparse_ratio))

            pre_sparse = random.sample(range(self.pre_popu.num), num_pre)
            pos_sparse = random.sample(range(self.pos_popu.num), num_pos)

            pre_exclude = set(range(self.pre_popu.num)) - set(pre_sparse)
            pos_exclude = set(range(self.pos_popu.num)) - set(pos_sparse)

            if self.pre_popu.ex_inh_type == "excitatory":
                connect_list = torch.rand((self.pre_popu.num, self.pos_popu.num))
            else:
                connect_list = -1. * torch.rand((self.pre_popu.num, self.pos_popu.num))
                # connect_list = -3. * torch.rand((self.pre_popu.num, self.pos_popu.num))

            connect_list[list(pre_exclude)] = 0.
            connect_list[:, list(pos_exclude)] = 0.
            return connect_list
        else:
            raise ValueError("sparse_ratio must in [0, 1] !")
