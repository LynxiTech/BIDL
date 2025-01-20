'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import torch


class PoissonInput(torch.nn.Module):
    def __init__(self, input_num, post_population):
        super(PoissonInput, self).__init__()
        self.input_num = input_num
        self.pos_popu = post_population
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
        connect_list = torch.rand((self.input_num, self.pos_popu.num))
        return connect_list
