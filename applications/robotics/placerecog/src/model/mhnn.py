import os
import torch
import torch.utils.model_zoo
import torchvision.models as models
from model.snn import *
#from src.model.cann_torch_odeint import CANN
from model.cann import CANN
from tools.utils import *

from torch import nn
import sys
import numpy as np
from lynadapter.warp_load_save import load_kernel, save_kernel


## TODO 状态变量以注释的形式标记

# trace
# spike
# membrane

# self.project.weight.data
# self.lateral_conn.weight.data
# self.thr_decay1.data
# self.ref_decay1.data
# self.cur_decay1.data
# r_spike
# r_mem
# thr_trace
# ref_trace
# cur_trace
# r_sumspike

# gpu environment
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def compare_result(apu_x, torch_y):
    ret = np.sqrt(np.sum((np.float32(apu_x) - np.float32(torch_y)) ** 2)) / np.sqrt(
        np.sum(np.float32(apu_x) ** 2))
    print(f'the error value of apu_x and torch_y is: {ret}')

def build_onnx(model_file, model_name="tmp_net/", model_shape=(1,3,48,320)):  # 3000 NCHW 385.81, 3000 NHWC 393.004
    model = lyn.DLModel()
    model.load(model_file, model_type="ONNX", inputs_dict={"input.1":model_shape},
                    in_type="float32",
                    out_type="float32",
                    transpose_axis=[(0,3,1,2)]
                    )
    offline_builder = lyn.Builder(target="apu")
    # offline_builder.build(model.graph, model.params, out_path=model_name)
    offline_builder.build(model.graph, model.params, out_path=model_name, run_batch=9)
    print(f'[lyn_build] model build end! save out_path is {model_name}\n')


def compare_spike(apu_x, torch_y):
    apu_x = torch.from_numpy(apu_x)
    torch_y = torch.from_numpy(torch_y)
    nonzeros = torch.nonzero(apu_x - torch_y)
    len1 = nonzeros[:, 1].flatten(0).shape[0]
    len_x = apu_x.flatten(0).shape[0]
    len_y = torch_y.flatten(0).shape[0]
    assert len_x == len_y
    ret = len1 / len_x
    # print('nonzeros = {}, len_x = {}'.format(len1, len_x))
    print(f'error rate: {ret * 100:.2f}%')


base_path = "../../../../../model_files/robotics/placerecog/"
class MHNN(nn.Module):
    def __init__(self, **kwargs):
        super(MHNN, self).__init__()
        self.USE_LYNGOR = kwargs.get('USE_LYNGOR')
        self.USE_LEGACY = kwargs.get('USE_LEGACY')
        self.COMPILE_ONLY = kwargs.get('COMPILE_ONLY')
        self.cmp_cnt = 0
        self.chip_id = kwargs.get('chip_id')
        if self.USE_LEGACY:
            from lynadapter.lyn_sdk_model import ApuRun_Single
            self.arun_0 = ApuRun_Single(self.chip_id, base_path + "model_config/resnet50/Net_0/")
            self.arun_1 = ApuRun_Single(self.chip_id, base_path + "model_config/snn/Net_0/")
            self.arun_2 = ApuRun_Single(self.chip_id, base_path + "model_config/cann/Net_0/")
            self.arun_3 = ApuRun_Single(self.chip_id, base_path + "model_config/mlsm/Net_0/")
        if self.USE_LYNGOR:
            import lyngor as lyn
            self.lyn_model = lyn.DLModel()

        self.cnn_arch = kwargs.get('cnn_arch')
        self.num_class = kwargs.get('num_class')
        self.cann_num = kwargs.get('cann_num')
        self.rnn_num = kwargs.get('rnn_num')
        self.lr = kwargs.get('lr')
        self.sparse_lambdas = kwargs.get('sparse_lambdas')
        self.r = kwargs.get('r')

        self.reservoir_num = kwargs.get('reservoir_num')
        self.threshold = kwargs.get('spiking_threshold')

        self.num_epoch = kwargs.get('num_epoch')
        self.num_iter = kwargs.get('num_iter')
        self.w_fps = kwargs.get('w_fps')
        self.w_gps = kwargs.get('w_gps')
        self.w_dvs = kwargs.get('w_dvs')
        self.w_head = kwargs.get('w_head')
        self.w_time = kwargs.get('w_time')

        self.seq_len_aps = kwargs.get('seq_len_aps')
        self.seq_len_gps = kwargs.get('seq_len_gps')
        self.seq_len_dvs = kwargs.get('seq_len_dvs')
        self.seq_len_head = kwargs.get('seq_len_head')
        self.seq_len_time = kwargs.get('seq_len_time')
        self.dvs_expand = kwargs.get('dvs_expand')

        # self.expand_len = int(
        #     self.seq_len_gps * self.seq_len_aps / np.gcd(self.seq_len_gps, self.seq_len_dvs) * self.dvs_expand)
        self.expand_len = 9
        self.ann_pre_load = kwargs.get('ann_pre_load')
        self.snn_pre_load = kwargs.get('snn_pre_load')
        self.re_trained = kwargs.get('re_trained')

        self.train_exp_idx = kwargs.get('train_exp_idx')
        self.test_exp_idx = kwargs.get('test_exp_idx')

        self.data_path = kwargs.get('data_path')
        self.snn_path = kwargs.get('snn_path')

        #aps input size
        self.batch_size =  kwargs.get('batch_size')
        self.channel = 3
        self.w = 320
        self.h = 240
        #self.device = kwargs.get('device')
        self.device = device


        if self.ann_pre_load:
            print("=> Loading pre-trained model '{}'".format(self.cnn_arch))
            self.cnn = models.__dict__[self.cnn_arch](pretrained=self.ann_pre_load)
        else:
            print("=> Using randomly inizialized model '{}'".format(self.cnn_arch))
            self.cnn = models.__dict__[self.cnn_arch](pretrained=self.ann_pre_load)

        if self.cnn_arch == "mobilenet_v2":
            """ MobileNet """
            self.feature_dim = self.cnn.classifier[1].in_features
            self.cnn.classifier[1] = nn.Identity()

        elif self.cnn_arch == "resnet50":
            """ Resnet50 """
            self.feature_dim = self.cnn.fc.in_features
            self.cnn.fc = nn.Identity()

            # self.feature_dim = 512
            #
            # # self.cnn.layer1 = nn.Identity()
            # self.cnn.layer2 = nn.Identity()
            # self.cnn.layer3 = nn.Identity()
            # self.cnn.layer4 = nn.Identity()
            # self.cnn.fc = nn.Identity()
            #
            # self.cnn.layer1[1] = nn.Identity()
            # self.cnn.layer1[2] = nn.Identity()
            #
            # # self.cnn.layer2[0] = nn.Identity()
            # # self.cnn.layer2[0].conv2 = nn.Identity()
            # # self.cnn.layer2[0].bn2 = nn.Identity()
            #
            # fc_inputs = 256
            # self.cnn.fc = nn.Linear(fc_inputs,self.feature_dim)

        else:
            print("=> Please check model name or configure architecture for feature extraction only, exiting...")
            exit()

        for param in self.cnn.parameters():
            param.requires_grad = self.re_trained

        #############
        # SNN module
        #############
        self.snn = SNN(device = self.device).to(self.device)
        self.snn_out_dim = self.snn.fc2.weight.size()[1]
        self.ann_out_dim = self.feature_dim
        self.cann_out_dim = 4 * self.cann_num
        self.reservior_inp_num = self.ann_out_dim + self.snn_out_dim + self.cann_out_dim
        self.LN = nn.LayerNorm(self.reservior_inp_num)
        if self.snn_pre_load:
            self.snn.load_state_dict(torch.load(self.snn_path)['snn'])

        #############
        # CANN module
        #############
        self.cann_num = self.cann_num
        self.cann = None
        self.num_class = self.num_class

        #############
        # MLSM module
        #############
        self.input_size = self.feature_dim
        self.reservoir_num = self.reservoir_num

        self.threshold = 0.5
        self.decay = nn.Parameter(torch.rand(self.reservoir_num))

        self.K = 128
        self.num_block = 5
        self.num_blockneuron = int(self.reservoir_num / self.num_block)

        self.decay_scale = 0.5
        self.beta_scale = 0.1

        self.thr_base1 = self.threshold

        self.thr_beta1 = nn.Parameter(self.beta_scale * torch.rand(self.reservoir_num))

        self.thr_decay1 = nn.Parameter(self.decay_scale * torch.rand(self.reservoir_num))

        self.ref_base1 = self.threshold
        self.ref_beta1 = nn.Parameter(self.beta_scale * torch.rand(self.reservoir_num))
        self.ref_decay1 = nn.Parameter(self.decay_scale * torch.rand(self.reservoir_num))

        self.cur_base1 = 0
        self.cur_beta1 = nn.Parameter(self.beta_scale * torch.rand(self.reservoir_num))
        self.cur_decay1 = nn.Parameter(self.decay_scale * torch.rand(self.reservoir_num))

        self.project = nn.Linear(self.reservior_inp_num, self.reservoir_num)

        self.project_mask_matrix = torch.zeros((self.reservior_inp_num, self.reservoir_num))


        input_node_list = [0, self.ann_out_dim, self.snn_out_dim, self.cann_num * 2, self.cann_num]

        input_cum_list = np.cumsum(input_node_list)

        for i in range(len(input_cum_list) - 1):
            self.project_mask_matrix[input_cum_list[i]:input_cum_list[i + 1],
            self.num_blockneuron * i:self.num_blockneuron * (i + 1)] = 1

        self.project.weight.data = self.project.weight.data * self.project_mask_matrix.t()

        self.lateral_conn = nn.Linear(self.reservoir_num, self.reservoir_num)

        # control the ratio of # of lateral conn.
        self.lateral_conn_mask = torch.rand(self.reservoir_num, self.reservoir_num) > 0.8
        # remove self-recurrent conn.
        self.lateral_conn_mask = self.lateral_conn_mask * (1 - torch.eye(self.reservoir_num, self.reservoir_num))
        # adjust the intial weight of conn.
        self.lateral_conn.weight.data = 1e-3 * self.lateral_conn.weight.data * self.lateral_conn_mask.T

        #############
        # readout module
        #############

        # self.mlp1 = nn.Linear(self.reservoir_num, 256)
        # self.mlp2 = nn.Linear(256, self.num_class)
        self.mlp =  nn.Linear(self.reservoir_num, self.num_class)

    def cann_init(self, data):
        self.cann = CANN(data)

    def lr_initial_schedule(self, lrs=1e-3):
        hyper_param_list = ['decay',
                            'thr_beta1', 'thr_decay1',
                            'ref_beta1', 'ref_decay1',
                            'cur_beta1', 'cur_decay1']
        hyper_params = list(filter(lambda x: x[0] in hyper_param_list, self.named_parameters()))
        base_params = list(filter(lambda x: x[0] not in hyper_param_list, self.named_parameters()))
        hyper_params = [x[1] for x in hyper_params]
        base_params = [x[1] for x in base_params]
        optimizer = torch.optim.SGD(
            [
                {'params': base_params, 'lr': lrs},
                {'params': hyper_params, 'lr': lrs / 2},
            ], lr=lrs, momentum=0.9, weight_decay=1e-7
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.num_epoch))
        return optimizer, scheduler

    def wta_mem_update(self, fc, fv, k, inputx, spike, mem, thr, ref, last_cur):
        # scaling_lateral_inputs = 0.1
        #state = fc(inputx) + fv(spike) * scaling_lateral_inputs
        state = fc(inputx) + fv(spike)

        mem = (mem - spike * ref) * self.decay + state + last_cur
        # ratio of # of winners
        q0 = (100 -20)/100
        mem = mem.reshape(self.batch_size, self.num_blockneuron, -1)

        ## TODO replace torch.quantile with torch.topk
        nps = torch.quantile(mem, q=q0, keepdim=True, axis=1)
        # nps =mem.max(axis=1,keepdim=True)[0] - thr

        # by Shengpeng
        # mem_t = mem.transpose(1, 2)
        # mem_s, _ = torch.sort(mem_t)
        # mem_q1, mem_q2 = mem_s[:, :, 639], mem_s[:, :, 640]
        # nps_c = mem_q1 + (mem_q2 - mem_q1) * 0.2
        # nps = nps_c.reshape(self.batch_size, 1, 5)

        # allows that only winner can fire spikes
        mem = F.relu(mem - nps).reshape(self.batch_size, -1)

        # spiking firing function
        spike = act_fun(mem - thr)
        return spike.float(), mem

    def trace_update(self, trace, spike, decay):
        return trace * decay + spike

    def resnet50_apu(self, aps_inp):
        apu_0_list = []
        for i in range(aps_inp.shape[0]):
            # in_dat = aps_inp[i][None].permute(0, 2, 3, 1).numpy()
            in_dat = aps_inp[i][None].numpy()
            out_0 = self.arun_0.run(np.ascontiguousarray(in_dat), i)
            apu_0_list.append(out_0[0])
        apu_0 = np.concatenate(apu_0_list)
        aps_out = torch.from_numpy(apu_0)
        return aps_out

    def snn_apu(self, dvs_inp):
        apu_1_list = []
        for i in range(dvs_inp.shape[1]):
            in_dat = dvs_inp[:, i, ...].numpy()
            out_1 = self.arun_1.run(in_dat, i)
            apu_1_list.append(out_1[0][None])
        apu_1 = np.concatenate(apu_1_list)
        out2 = torch.from_numpy(apu_1)
        return out2

    def cann_apu(self, gps_inp, head_inp):
        apu_2_record = []
        for idx in range(self.batch_size):
            apu_2_list = []
            inpt = torch.cat((gps_inp[idx], head_inp[idx].cpu()), axis=1)
            for i in range(inpt.shape[0]):
                out_2 = self.arun_2.run(inpt[i].numpy().astype(np.float32), i)
                apu_2_list.append(
                    np.expand_dims(out_2[0].reshape(-1, 4), axis=0)
                )
            apu_2_record.append(np.concatenate(apu_2_list)[None])
        apu_2 = np.concatenate(apu_2_record)
        gps_out = torch.from_numpy(apu_2)
        return gps_out

    def mlsm_apu(self, out1, out2, out3, out4, out5):
        for ti in range(self.expand_len):
            in1_t = out1[ti].numpy().reshape(-1)
            in2_t = out2[ti].numpy().reshape(-1)
            in3_t = out3[ti].numpy().reshape(-1)
            in4_t = out4[ti].numpy().reshape(-1)
            in5_t = out5[ti].numpy().reshape(-1)
            data = np.concatenate([in1_t, in2_t, in3_t, in4_t, in5_t])
            out_3 = self.arun_3.run(data, ti)
        apu_out = out_3[0]
        apu_r_sumspike = out_3[1]

        out = torch.from_numpy(apu_out)
        r_sumspike = torch.from_numpy(apu_r_sumspike)
        return out, r_sumspike

    def forward(self, inp, epoch=100):

        ## TODO 将初始值赋值类操作移到__init__函数

        aps_inp = inp[0].to(self.device)
        gps_inp = inp[1]
        dvs_inp = inp[2].to(self.device)
        head_inp = inp[3].to(self.device)

        ## TODO 不支持batchsize > 1的操作，用for 循环实现，循环次数为常数

        #print('aps_input shape:', aps_inp.shape)

        #batch_size, seq_len, channel, w, h = aps_inp.size()
        #if self.w_fps > 0:
        #aps_inp = aps_inp.view(batch_size * seq_len, channel, w, h)
        aps_inp = aps_inp.view(self.batch_size * self.seq_len_aps * 3, self.channel, self.w, self.h)  # 9=seq_len_aps

        #print('aps_input view shape:', aps_inp.shape)
        if not self.USE_LEGACY:
            if self.USE_LYNGOR:
                if self.cmp_cnt < 4:
                    import lyngor as lyn
                    # img = torch.rand((1, 3, 320, 240))
                    # torch.onnx.export(self.cnn, img, 'resnet50.onnx')
                    # build_onnx("./resnet50.onnx", "../../model_config/resnet50/", model_shape=(1, 3, 320, 240))

                    save_path = base_path + f"model_config/resnet50"
                    self.lyn_model.load(self.cnn,
                                   model_type='Pytorch',
                                   inputs_dict={'input': (1, 3, 320, 240)},
                                   in_type="float32",
                                   out_type="float32")

                    offline_builder = lyn.Builder(target="apu")
                    offline_builder.build(self.lyn_model.graph, self.lyn_model.param, out_path=save_path)
                    print(f'[lyn_build] model build end! save path is {save_path}\n', )
                    if not self.COMPILE_ONLY:
                        from lynadapter.lyn_sdk_model import ApuRun_Single
                        self.arun_0 = ApuRun_Single(self.chip_id, base_path + "model_config/resnet50/Net_0/")
                    self.cmp_cnt += 1

                if not self.COMPILE_ONLY:
                    aps_out = self.resnet50_apu(aps_inp)
                else:
                    aps_out = self.cnn(aps_inp)
            else:
                aps_out = self.cnn(aps_inp)
        else:
            aps_out = self.resnet50_apu(aps_inp)

        # arun_0.release()
        # res_reshape = aps_out.detach().numpy().reshape(-1)
        # apu_reshape = apu_0.reshape(-1)
        # compare_result(apu_reshape, res_reshape)

        out1 = aps_out.reshape(self.batch_size, self.seq_len_aps * 3, -1).permute([1, 0, 2])
        #out1 = aps_out.reshape(1, 1, -1).permute([1, 0, 2])
        # else:
        #     out1 = torch.zeros(self.seq_len_aps, batch_size, -1, device=self.device).to(torch.float32)

        if not self.USE_LEGACY:
        #if self.w_dvs > 0:
            if self.USE_LYNGOR:
                if self.cmp_cnt < 4:
                    import lyngor as lyn
                    self.snn.cmp = True

                    save_path = base_path + f"model_config/snn"
                    self.lyn_model.load(self.snn,
                                   model_type='Pytorch',
                                   inputs_dict={'input': tuple(dvs_inp[:, 0, ...].shape)},
                                   in_type="float32",
                                   out_type="float32")
                    
                    offline_builder = lyn.Builder(target="apu")
                    offline_builder.build(self.lyn_model.graph, self.lyn_model.param, out_path=save_path)
                    print(f'[lyn_build] model build end! save path is {save_path}\n', )

                    self.snn.cmp = False
                    if not self.COMPILE_ONLY:
                        from lynadapter.lyn_sdk_model import ApuRun_Single
                        self.arun_1 = ApuRun_Single(self.chip_id, base_path + "model_config/snn/Net_0/")
                    self.cmp_cnt += 1
                if not self.COMPILE_ONLY:
                    out2 = self.snn_apu(dvs_inp)
                else:
                    out2 = self.snn(dvs_inp, out_mode='time')
            else:
                out2 = self.snn(dvs_inp, out_mode='time')
        else:
            out2 = self.snn_apu(dvs_inp)

        # # arun_1.release()
        # res_reshape = out2.detach().numpy().reshape(-1)
        # apu_reshape = apu_1.reshape(-1)
        # compare_spike(apu_reshape[None], res_reshape[None])

        #     out2 = torch.zeros(self.seq_len_dvs * 3, batch_size, self.snn_out_dim, device=self.device).to(torch.float32)
        #else:
        res_list = []
        ### CANN module


        #if self.w_gps + self.w_head + self.w_time > 0:
        if not self.USE_LEGACY:
            if self.USE_LYNGOR:
                if self.cmp_cnt < 4:
                    import lyngor as lyn
                    save_path = base_path + f"model_config/cann"
                    self.lyn_model.load(self.cann,
                                   model_type='Pytorch',
                                   inputs_dict={'input': (1, 4)},
                                   in_type="float32",
                                   out_type="float32")
                    offline_builder = lyn.Builder(target="apu")
                    offline_builder.build(self.lyn_model.graph, self.lyn_model.param, out_path=save_path)
                    print(f'[lyn_build] model build end! save path is {save_path}\n')
                    if not self.COMPILE_ONLY:
                        from lynadapter.lyn_sdk_model import ApuRun_Single
                        self.arun_2 = ApuRun_Single(self.chip_id, base_path + "model_config/cann/Net_0/")
                    self.cmp_cnt += 1
                if not self.COMPILE_ONLY:
                    gps_out = self.cann_apu(gps_inp, head_inp)
                else:
                    gps_record = []
                    for idx in range(self.batch_size):
                        buf = self.cann.update(torch.cat((gps_inp[idx], head_inp[idx].cpu()), axis=1),
                                               trajactory_mode=True)
                        gps_record.append(buf[None, :, :, :])
                    # gps_out = torch.from_numpy(np.concatenate(gps_record)).cuda()
                    gps_out = torch.cat(gps_record)  # .cuda()
            else:
                gps_record = []
                for idx in range(self.batch_size):
                    buf = self.cann.update(torch.cat((gps_inp[idx],head_inp[idx].cpu()),axis=1), trajactory_mode=True)
                    gps_record.append(buf[None, :, :, :])
                # gps_out = torch.from_numpy(np.concatenate(gps_record)).cuda()
                gps_out = torch.cat(gps_record)#.cuda()
        else:
            gps_out = self.cann_apu(gps_inp, head_inp)

        # arun_2.release()
        # res_reshape = gps_out.detach().numpy().reshape(-1)
        # apu_reshape = apu_1.reshape(-1)
        # compare_result(apu_reshape, res_reshape)

        #gps_out = gps_out.permute([1, 0, 2, 3]).reshape(self.seq_len_gps, batch_size, -1)
        gps_out = gps_out.permute([1, 0, 2, 3]).reshape(self.seq_len_gps * 3, self.batch_size, -1)
        # else:
        #     gps_out = torch.zeros((self.seq_len_gps, batch_size, self.cann_out_dim), device=self.device)

        # A generic CANN module was used for rapid testing; CANN1D/2D are provided in cann.py
        out3 = gps_out[:, :, self.cann_num:self.cann_num * 3].to(self.device).to(torch.float32)  # position
        out4 = gps_out[:, :, : self.cann_num].to(self.device).to(torch.float32)  # time
        out5 = gps_out[:, :, - self.cann_num:].to(self.device).to(torch.float32)  # direction

        out3 *= self.w_gps
        out4 *= self.w_time
        out5 *= self.w_head


        input_num = self.feature_dim + self.snn.fc2.weight.size()[1] + self.cann_num * self.dvs_expand

        r_spike = r_mem = r_sumspike = torch.zeros(self.batch_size, self.reservoir_num, device=self.device)

        thr_trace = torch.zeros(self.batch_size, self.reservoir_num, device=self.device)
        ref_trace = torch.zeros(self.batch_size, self.reservoir_num, device=self.device)
        cur_trace = torch.zeros(self.batch_size, self.reservoir_num, device=self.device)

        K_winner = self.K * (1 + np.clip(self.num_epoch - epoch, a_min=0, a_max=self.num_epoch) / self.num_epoch)

        # out1_zeros = torch.zeros_like(out1[0], device=self.device)
        # out3_zeros = torch.zeros_like(out3[0], device=self.device)
        # out4_zeros = torch.zeros_like(out4[0], device=self.device)
        # out5_zeros = torch.zeros_like(out5[0], device=self.device)
        if torch.cuda.is_available() and not self.USE_LYNGOR:
            self.project.weight.data = self.project.weight.data * self.project_mask_matrix.t().cuda()
            self.lateral_conn.weight.data = self.lateral_conn.weight.data * self.lateral_conn_mask.T.cuda()
        else:
            self.project.weight.data = self.project.weight.data * self.project_mask_matrix.t()#.cuda()
            self.lateral_conn.weight.data = self.lateral_conn.weight.data * self.lateral_conn_mask.T#.cuda()

        self.decay.data = torch.clamp(self.decay.data, min=0, max=1.)
        self.thr_decay1.data = torch.clamp(self.thr_decay1.data, min=0, max=1.)
        self.ref_decay1.data = torch.clamp(self.ref_decay1.data, min=0, max=1)
        self.cur_decay1.data = torch.clamp(self.cur_decay1.data, min=0, max=1)

        if not self.USE_LEGACY:
            if self.USE_LYNGOR:
                if self.cmp_cnt < 4:
                    import lyngor as lyn
                    model = MLSM(self)
                    save_path = base_path + f"model_config/mlsm"
                    inputs_dict = {'input_0': (1, 2048),
                                   'input_1': (1, 512),
                                   'input_2': (1, 256),
                                   'input_3': (1, 128),
                                   'input_4': (1, 128)}
                    self.lyn_model.load(model,
                                   model_type='Pytorch',
                                   inputs_dict=inputs_dict,
                                   in_type="float32",
                                   out_type="float32")
                    offline_builder = lyn.Builder(target="apu")
                    offline_builder.build(self.lyn_model.graph, self.lyn_model.param, out_path=save_path)
                    print(f'[lyn_build] model build end! save path is {save_path}\n')
                    if not self.COMPILE_ONLY:
                        from lynadapter.lyn_sdk_model import ApuRun_Single
                        self.arun_3 = ApuRun_Single(self.chip_id, base_path + "model_config/mlsm/Net_0/")
                    self.cmp_cnt += 1
                if not self.COMPILE_ONLY:
                    out, r_sumspike = self.mlsm_apu(out1, out2, out3, out4, out5)
            else:
                out = 0.

                for step in range(self.expand_len):

                    # ## TODO 将APS补帧处理移到输入侧
                    # idx = step % 3
                    # # simulate multimodal information with different time scales
                    # if idx == 2:
                    #     combined_input = torch.cat((out1[step // 3], out2[step], out3[step // 3], out4[step // 3], out5[step // 3]), axis=1)
                    # else:
                    #     combined_input = torch.cat((out1_zeros, out2[step],out3_zeros, out4_zeros, out5_zeros), axis=1)
                    combined_input = torch.cat((out1[step], out2[step], out3[step], out4[step], out5[step]), axis=1)

                    thr = self.thr_base1 + thr_trace * self.thr_beta1
                    # ref = self.ref_base1 + ref_trace * self.ref_beta1 # option: ref = self.ref_base1
                    ref = self.ref_base1
                    cur = self.cur_base1 + cur_trace * self.cur_beta1

                    inputx = combined_input.float()
                    #print(inputx.shape)
                    r_spike, r_mem = self.wta_mem_update(self.project, self.lateral_conn, K_winner, inputx, r_spike, r_mem,
                                                         thr, ref, cur)
                    thr_trace = self.trace_update(thr_trace, r_spike, self.thr_decay1)
                    ref_trace = self.trace_update(ref_trace, r_spike, self.ref_decay1)
                    cur_trace = self.trace_update(cur_trace, r_spike, self.cur_decay1)
                    r_sumspike = r_sumspike + r_spike


                    # cat_out = F.dropout(r_sumspike, p=0.5, training=self.training)
                    # out1 = self.mlp1(r_sumspike).relu()
                    # out2 = self.mlp2(out1)

                    ## TODO 将mlp 和 reshape 放到循环内，要求没拍操作一样
                    out += self.mlp(r_spike)
        else:
            out, r_sumspike = self.mlsm_apu(out1, out2, out3, out4, out5)

        # arun_3.release()
        # res_reshape = np.concatenate([out.detach().numpy().reshape(-1), r_sumspike.detach().numpy().reshape(-1)])
        # apu_reshape = np.concatenate([apu_out.reshape(-1), apu_r_sumspike.reshape(-1)])
        # compare_result(apu_reshape, res_reshape)
        if self.COMPILE_ONLY:
            sys.exit(0)

        neuron_pop = r_sumspike.reshape(self.batch_size, -1, self.num_blockneuron).permute([1, 0, 2])
        return out, (neuron_pop[0], neuron_pop[1])


class MLSM(torch.nn.Module):
    def __init__(self, cls):
        super(MLSM, self).__init__()
        self.batch_size = cls.batch_size
        self.num_blockneuron = cls.num_blockneuron
        self.ref = 0.5
        self.batch_size = cls.batch_size
        self.reservoir_num = cls.reservoir_num
        self.decay = cls.decay
        self.fc = cls.project
        self.fv = cls.lateral_conn
        self.thr_base1 = cls.thr_base1
        self.thr_beta1 = cls.thr_beta1
        self.ref_base1 = cls.ref_base1
        self.cur_base1 = cls.cur_base1
        self.cur_beta1 = cls.cur_beta1
        self.thr_decay1 = cls.thr_decay1
        self.ref_decay1 = cls.ref_decay1
        self.cur_decay1 = cls.cur_decay1
        self.mlp = cls.mlp

    def trace_update(self, trace, spike, decay):
        return trace * decay + spike

    def wta_mem_update(self, inputx, spike, mem, thr, last_cur):
        state = self.fc(inputx) + self.fv(spike)

        mem = (mem - spike * self.ref) * self.decay + state + last_cur
        mem = mem.reshape(self.batch_size, self.num_blockneuron, -1)

        mem_t = mem.transpose(1, 2)
        # # ori
        # mem_s, _ = torch.sort(mem_t)
        # mem_q1, mem_q2 = mem_s[:, :, 639], mem_s[:, :, 640]

        # # equal 1
        # mem_s, _ = torch.sort(mem_t, descending=True)
        # mem_s = torch.flip(mem_s, dims=[-1])
        # mem_q1, mem_q2 = mem_s[:, :, 639], mem_s[:, :, 640]

        # equal 2
        mem_s, _ = torch.sort(mem_t, descending=True)
        mem_q1, mem_q2 = mem_s[:, :, 160], mem_s[:, :, 159]

        nps_c = mem_q1 + (mem_q2 - mem_q1) * 0.2
        nps = nps_c.reshape(self.batch_size, 1, 5)

        # allows that only winner can fire spikes
        mem = F.relu(mem - nps).reshape(self.batch_size, -1)
        # spiking firing function
        spike = (mem - thr).gt(0).float()
        return spike, mem


    def forward(self, in1, in2, in3, in4, in5):
        self.thr_trace = torch.zeros(self.batch_size, self.reservoir_num, dtype=torch.float32)
        self.ref_trace = torch.zeros(self.batch_size, self.reservoir_num, dtype=torch.float32)
        self.cur_trace = torch.zeros(self.batch_size, self.reservoir_num, dtype=torch.float32)
        self.r_spike = torch.zeros(self.batch_size, self.reservoir_num, dtype=torch.float32)
        self.r_mem = torch.zeros(self.batch_size, self.reservoir_num, dtype=torch.float32)
        self.r_sumspike = torch.zeros(self.batch_size, self.reservoir_num, dtype=torch.float32)
        self.out = torch.zeros(1, 100, dtype=torch.float32)

        thr_trace = load_kernel(self.thr_trace, f'thr_trace')
        ref_trace = load_kernel(self.ref_trace, f'ref_trace')
        cur_trace = load_kernel(self.cur_trace, f'cur_trace')
        r_spike = load_kernel(self.r_spike, f'r_spike')
        r_mem = load_kernel(self.r_mem, f'r_mem')
        r_sumspike = load_kernel(self.r_sumspike, f'r_sumspike')
        out = load_kernel(self.out, f'out')

        combined_input = torch.cat((in1, in2, in3, in4, in5), axis=1)

        thr = self.thr_base1 + thr_trace * self.thr_beta1
        # ref = self.ref_base1 + ref_trace * self.ref_beta1 # option: ref = self.ref_base1
        # ref = self.ref_base1
        cur = self.cur_base1 + cur_trace * self.cur_beta1

        inputx = combined_input.float()
        # print(inputx.shape)
        r_spike, r_mem = self.wta_mem_update(inputx, r_spike, r_mem, thr, cur)
        thr_trace = self.trace_update(thr_trace, r_spike, self.thr_decay1)
        ref_trace = self.trace_update(ref_trace, r_spike, self.ref_decay1)
        cur_trace = self.trace_update(cur_trace, r_spike, self.cur_decay1)
        r_sumspike= r_sumspike + r_spike

        # cat_out = F.dropout(r_sumspike, p=0.5, training=self.training)
        # out1 = self.mlp1(r_sumspike).relu()
        # out2 = self.mlp2(out1)

        ## TODO 将mlp 和 reshape 放到循环内，要求没拍操作一样
        out += self.mlp(r_spike)


        self.thr_trace = thr_trace.clone()
        self.ref_trace = ref_trace.clone()
        self.cur_trace = cur_trace.clone()
        self.r_spike = r_spike.clone()
        self.r_mem = r_mem.clone()

        save_kernel(self.thr_trace, f'thr_trace')
        save_kernel(self.ref_trace, f'ref_trace')
        save_kernel(self.cur_trace, f'cur_trace')
        save_kernel(self.r_spike, f'r_spike')
        save_kernel(self.r_mem, f'r_mem')

        self.r_sumspike = r_sumspike.clone()
        save_kernel(self.r_sumspike, f'r_sumspike')
        self.out = out.clone()
        save_kernel(self.out, f'out')

        return out, r_sumspike









