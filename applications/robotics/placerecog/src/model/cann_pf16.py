import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch

## TODO 变量过小，fp16会被看作0，考虑精度要求
## TODO 测试fp16是否正确工作

class CANN():
    """
    Generic CANN modules for rapid testing
    """
    def __init__(self, data,  num = 128, tau = 0.02, Inh_inp = -.1, I_dur = 0.5, dt = 0.05, A = 20, k = 1):
        self.num = num
        self.tau = tau
        self.Inh_inp = Inh_inp
        self.I_dur = I_dur
        self.dt = dt
        self.A = A
        self.k = k
        self.t_wins = data.shape[0] + 1
        self.n_units = data.shape[1]
        self.z_min, self.z_max = np.min(data, axis=0), np.max(data, axis=0) + 1e-4
        self.z_range = self.z_max - self.z_min
        self.centers = np.linspace(self.z_min, self.z_max, num)  # sample centers
        self.w =  np.zeros((self.num, self.num))
        self.a = 0.5
        self.J0 = 8
        self.u = []
        self.u.append(np.zeros((2,num)))
        self.r = []
        self.r.append(np.zeros((2,num)))

        for i in range(self.num - 1):
            for j in range(self.num - 1):
                d = np.float16(np.square((i - j) / self.a))
                self.w[i, j] = np.float16(self.J0 * np.exp(-0.5 * d) / (np.sqrt(2 * np.pi) * self.a))

    def data_reverse_transform(self, x):
        x_min, x_max = np.min(x, axis=0),np.max(x, axis=0)
        x_range = x_max - x_min
        return np.float16((x - x_min)/(np.float16(1e-11)+ x_range) * self.z_range) + self.z_min

    def get_stimulus_by_pos(self, x):
        d = (x - self.centers) / (self.z_range + np.float16(1e-4))
        y = np.float16(np.exp(-1 * np.square(d / 0.5)))
        return y

    def update(self, data, trajactory_mode = False):
        seq_len, seq_dim = data.shape
        u = np.zeros((self.num,seq_dim))
        u_record = np.zeros((seq_len, self.num, seq_dim))

        for i in range(0, seq_len):
            if i == 0:
                cur_stimulus = np.float16(self.get_stimulus_by_pos(data[0])) #.to(torch.float16)
            else:
                cur_stimulus = np.float16(self.get_stimulus_by_pos(data[i - 1]))
            t = np.arange(self.I_dur * i, self.I_dur * i + self.I_dur, self.dt)

            ## TODO replace odeint
            cur_du = odeint(int_u, u.flatten(), t,
                            args=(self.w, cur_stimulus.t().numpy(), self.tau, self.Inh_inp, self.k, self.dt))

            u = cur_du[-1].reshape(-1, seq_dim)
            r1 = np.square(u)
            r2 = 1.0 + 0.5 * self.k * np.sum(r1,axis = 0)
            u_record[i] = np.float16(r1/r2.reshape(1,-1))
        
            out =  np.float16(r1/r2)
        if trajactory_mode:
            out = u_record
            
        return out

def int_u(u, t, w, Iext, tau, I_inh, k, dt):
    # membrane potential dynamics
    # parameter configuration mainly followed by ref. (wu et al.2008)
    cann_num = w.shape[0]
    u = u.reshape(-1, cann_num)
    r1 = np.square(u)
    r2 = 1.0 + k * np.sum(r1)
    r = np.float16(r1 / r2)
    Irec = np.dot(r, w)
    du = (-u + Irec + Iext + I_inh) * dt / tau
    return du.flatten()


# class CANN1D():
#     def __init__(self, data, num=128, tau=0.02, Inh_inp=-.1, I_dur=0.5, dt=0.05, A=20, k=1):
#         self.num = num
#         self.tau = tau
#         self.Inh_inp = Inh_inp
#         self.I_dur = I_dur
#         self.dt = dt
#         self.A = A
#         self.k = k
#         self.t_wins = data.shape[0] + 1
#         self.n_units = data.shape[1]
#         self.z_min, self.z_max = np.min(data, axis=0), np.max(data, axis=0) + 1e-4
#         self.z_range = self.z_max - self.z_min
#         self.centers = np.linspace(self.z_min, self.z_max, num)  # sample centers
#         self.w = np.zeros((self.num, self.num))
#         self.a = 0.5
#         self.J0 = 8
#         self.u = []
#         self.u.append(np.zeros((2, num)))
#         self.r = []
#         self.r.append(np.zeros((2, num)))
#
#         for i in range(self.num - 1):
#             for j in range(self.num - 1):
#                 d = np.square((i - j) / self.a)
#                 if d > np.pi: # set up connectivity matrix on a  ring
#                     d = 2*np.pi-d
#                 self.w[i, j] = self.J0 * np.exp(-0.5 * d) / (np.sqrt(2 * np.pi) * self.a)
#
#     def data_reverse_transform(self, x):
#         x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
#         x_range = x_max - x_min
#         return (x - x_min) / (1e-11 + x_range) * self.z_range + self.z_min
#
#     def get_stimulus_by_pos(self, x):
#         d = (x - self.centers) / (self.z_range + 1e-4)
#         y = np.exp(-1 * np.square(d / 0.5))
#         return y
#
#     def update(self, data, trajactory_mode=False):
#         seq_len, seq_dim = data.shape
#         u = np.zeros((self.num, seq_dim))
#         u_record = np.zeros((seq_len, self.num, seq_dim))
#
#         for i in range(0, seq_len):
#             if i == 0:
#                 cur_stimulus = self.get_stimulus_by_pos(data[0])
#             else:
#                 cur_stimulus = self.get_stimulus_by_pos(data[i - 1])
#             t = np.arange(self.I_dur * i, self.I_dur * i + self.I_dur, self.dt)
#             cur_du = odeint(int_u, u.flatten(), t,
#                             args=(self.w, cur_stimulus.t().numpy(), self.tau, self.Inh_inp, self.k, self.dt))
#
#             u = cur_du[-1].reshape(-1, seq_dim)
#             r1 = np.square(u)
#             r2 = 1.0 + 0.5 * self.k * np.sum(r1, axis=0)
#             u_record[i] = r1 / r2.reshape(1, -1)
#
#             out = r1 / r2
#         if trajactory_mode:
#             out = u_record
#
#         return out


# class CANN2D():
#     def __init__(self, data,  length = 128, tau = 0.02, Inh_inp = -.1, I_dur = 0.4, dt = 0.05, A = 20, k = 1):
#         self.length = length  # 2D-CANN :(length, length)
#         self.tau = tau  # key parameters: control the pred curve
#         self.Inh_inp = Inh_inp  # key parameters: inhibitive stimulus, key parameters
#         self.I_dur = I_dur
#         self.dt = dt
#         self.A = A
#         self.k = 1
#         self.t_wins = data.shape[0] + 1
#         self.n_units = data.shape[1]
#         self.z_min, self.z_max = np.min(data), np.max(data) + 1e-7
#         self.z_range = self.z_max - self.z_min
#         self.x = np.linspace(self.z_min, self.z_max, length)  # sample centers
#         self.dx = self.z_range / length
#
#         self.a = 2
#         self.J0 = 4
#         self.u = [] # membrane
#         self.u.append(np.zeros((length,length)))
#         self.r = [] # firing rate
#         self.r.append(np.zeros((length,length)))
#         self.conn_mat = self.make_conn()
#
#     def dist(self,d):
#         v_size = np.asarray([self.z_range,self.z_range])
#         return np.where(d > v_size/2, v_size -d ,d)
#
#     def make_conn(self):
#         x1, x2 = np.meshgrid(self.x/self.z_max,self.x/self.z_max)
#         value = np.stack([x1.flatten(), x2.flatten()]).T
#
#         dd = self.dist(np.abs(value[0] - value))
#         d = np.linalg.norm(dd, ord=1, axis=1)
#         d = d.reshape((self.length, self.length))
#         Jxx = self.J0 * np.exp(-0.5 * np.square(d/self.a))/(np.sqrt(2*np.pi)*self.a)
#         return Jxx
#
#
#     def data_reverse_transform(self, x):
#         x_min, x_max = np.min(x, axis=0),np.max(x, axis=0)
#         x_range = x_max - x_min
#         return (x - x_min)/(1e-11+x_range) * self.z_range + self.z_min
#
#     def get_stimulus_by_pos(self, x):
#         # code external stimilus by Guassian coding
#         x1, x2 = np.meshgrid(self.x, self.x)
#         value = np.stack([x1.flatten(), x2.flatten()]).T
#         # dd = self.dist(np.abs(np.asarray(x)-value))
#         dd = np.abs(np.asarray(x) - value)
#         d = np.linalg.norm(dd, axis=1)
#         d = d.reshape((self.length, self.length))
#         return self.A * np.exp(-0.25*np.square(d/2))
#
#
#     def update(self, data, trajactory_mode = False):
#         # convert the data input a 3D tensor: seq_len * dim * dim
#         seq_len, seq_dim = data.shape
#         u = np.zeros((self.length, self.length))
#         u_record = np.zeros((seq_len, self.length, self.length))
#
#         for i in range(1, seq_len):
#             # generate the 2D inputs
#             cur_stimulus = self.get_stimulus_by_pos(data[i - 1])
#             t = np.arange(self.I_dur * i, self.I_dur * i + self.I_dur, self.dt)
#             # perform updates
#             cur_du = odeint(int_u, u.flatten(), t,
#                             args=(self.conn_mat, cur_stimulus.T,
#                                   self.tau, self.Inh_inp, self.k, self.dt))
#             u = cur_du[-1].reshape(-1, self.length)
#             r1 = np.square(u)
#             r2 = 1.0 + self.k * np.sum(r1)
#             u_record[i] = r1/r2
#
#
#         # output the matrix and down-sampling
#         out = r1/r2
#         if trajactory_mode == True:
#             out = u_record