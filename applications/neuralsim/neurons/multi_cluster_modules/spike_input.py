'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import torch
import random
import numpy as np

torch.set_printoptions(threshold=np.Inf)


class SpikeInput(torch.nn.Module):
    def __init__(self,
            input_num,
            post_population,
            sparse_ratio=None):
        super(SpikeInput, self).__init__()
        self.input_num = input_num
        self.pos_popu = post_population
        self.sparse_ratio = sparse_ratio
        self.connect_mat = self.connection_matrix()

    def forward(self, inpt):
        if self.connect_mat.device != inpt.device:
            connect_mat_device = self.connect_mat.device
            synapse_current = torch.mm(inpt, self.connect_mat.to(inpt.device))
            synapse_current = synapse_current.to(connect_mat_device)
        else:
            synapse_current = torch.mm(inpt, self.connect_mat)
        self.pos_popu.spikes_ex += synapse_current


    def connection_matrix(self):
        if self.sparse_ratio:
            num_pre = int(np.ceil(self.input_num * self.sparse_ratio))
            num_pos = int(np.ceil(self.pos_popu.num * self.sparse_ratio))

            pre_sparse = random.sample(range(self.input_num), num_pre)
            pos_sparse = random.sample(range(self.pos_popu.num), num_pos)

            pre_exclude = set(range(self.input_num)) - set(pre_sparse)
            pos_exclude = set(range(self.pos_popu.num)) - set(pos_sparse)

            # connect_list = 2. * torch.randint(1, 10, (self.input_num, self.pos_popu.num)).float()
            connect_list = torch.rand((self.input_num, self.pos_popu.num))

            connect_list[list(pre_exclude)] = 0.
            connect_list[:, list(pos_exclude)] = 0.
            return connect_list
        else:
            return None
