import os
from PIL import Image
import torch
import torch.utils.model_zoo
from torch.utils.data import Dataset
import torchvision
import math
import numpy as np
import random
# from sklearn.metrics import precision_recall_curve

def least_common_multiple(num):
    mini = 1
    for i in num:
        mini = int(i) * int(mini) /math.gcd(int(i), mini)
        mini = int(mini)
    return mini
    
class SequentialDataset(Dataset):
    def __init__(self, data_dir, exp_idx, transform, nclass=None, seq_len_aps=None, seq_len_dvs = None,
                 seq_len_gps=None, seq_len_head = None, seq_len_time = None):
        self.data_dir = data_dir
        self.transform = transform
        self.num_exp = len(exp_idx)
        self.exp_idx = exp_idx

        self.total_aps = [np.sort(os.listdir(data_dir + str(idx) + '/dvs_frames')) for idx in exp_idx]
        self.total_dvs = [np.sort(os.listdir(data_dir + str(idx) + '/dvs_7ms_3seq')) for idx in exp_idx]
        self.num_imgs = [len(x) for x in self.total_aps]
        self.raw_pos = [np.loadtxt(data_dir + str(idx) + '/position.txt', delimiter=' ') for idx in exp_idx]
        self.raw_head = [np.loadtxt(data_dir + str(idx) + '/direction.txt', delimiter=' ') for idx in exp_idx]

        self.t_pos = [x[:, 0] for x in self.raw_pos]
        self.t_aps = [[float(x[:-4]) for x in y] for y in self.total_aps]
        self.t_dvs = [[float(x[:-4]) for x in y] for y in self.total_dvs]
        self.data_pos = [idx[:, 0:3]-idx[:, 0:3].min(axis=0) for idx in self.raw_pos]
        self.data_head = [idx[:, 0:3] - idx[:, 0:3].min(axis=0) for idx in
                         self.raw_head]
        self.seq_len_aps = seq_len_aps
        self.seq_len_gps = seq_len_gps
        self.seq_len_dvs = seq_len_dvs
        self.seq_len_head = seq_len_head
        self.seq_len_time = seq_len_time
        self.seq_len = max(seq_len_gps, seq_len_aps)
        self.nclass = nclass

        self.lens = len(self.total_aps) - self.seq_len
        self.dvs_data = None
        self.duration = [x[-1] - x[0] for x in self.t_dvs]

        nums = 1e5
        for x in self.total_aps:
            if len(x) < nums: nums = len(x)
        for x in self.total_dvs:
            if len(x) < nums: nums = len(x)
        for x in self.raw_pos:
            if len(x) < nums: nums = len(x)

        self.lens = nums

    def __len__(self):
        return self.lens - self.seq_len * 2

    def __getitem__(self, idx):
        exp_index = np.random.randint(self.num_exp)
        idx = max(min(idx, self.num_imgs[exp_index] - self.seq_len * 2), self.seq_len_dvs * 3)
        img_seq = []
        for i in range(self.seq_len_aps):
            img_loc = self.data_dir + str(self.exp_idx[exp_index]) + '/dvs_frames/' + \
                      self.total_aps[exp_index][idx - self.seq_len_aps + i]
            img_seq += [Image.open(img_loc).convert('RGB')]
        img_seq_pt = []

        if self.transform:
            for images in img_seq:
                img_seq_pt += [torch.unsqueeze(self.transform(images), 0)]

        img_seq = torch.cat(img_seq_pt, dim=0)
        t_stamps = self.raw_pos[exp_index][:, 0]
        t_target = self.t_aps[exp_index][idx]

        idx_pos = max(np.searchsorted(t_stamps, t_target), self.seq_len_aps)
        pos_seq = self.data_pos[exp_index][idx_pos - self.seq_len_gps:idx_pos, :]
        pos_seq = torch.from_numpy(pos_seq.astype('float32'))

        t_stamps = self.raw_head[exp_index][:, 0]
        t_target = self.t_aps[exp_index][idx]
        idx_head = max(np.searchsorted(t_stamps, t_target), self.seq_len_aps)
        head_seq = self.data_head[exp_index][idx_head - self.seq_len_gps:idx_pos, 1]
        head_seq = torch.from_numpy(head_seq.astype('float32')).reshape(-1,1)

        idx_dvs = np.searchsorted(self.t_dvs[exp_index], t_target, sorter=None) - 1
        t_stamps = self.t_dvs[exp_index][idx_dvs]
        dvs_seq = torch.zeros(self.seq_len_dvs * 3, 2, 130, 173)
        for i in range(self.seq_len_dvs):
            dvs_path = self.data_dir + str(self.exp_idx[exp_index]) + '/dvs_7ms_3seq/' \
                       + self.total_dvs[exp_index][idx_dvs - self.seq_len_dvs + i + 1]
            dvs_buf = torch.load(dvs_path)
            dvs_buf = dvs_buf.permute([1, 0, 2, 3])
            dvs_seq[i * 3: (i + 1) * 3] = torch.nn.functional.avg_pool2d(dvs_buf, 2)
        ids = int((t_stamps - self.t_dvs[exp_index][0]) / self.duration[exp_index] * self.nclass)

        ids = np.clip(ids, a_min=0, a_max= self.nclass - 1)
        ids = np.array(ids)
        ids = torch.from_numpy(ids).type(torch.long)
        return (img_seq, pos_seq, dvs_seq, head_seq), ids

def Data(data_path=None, batch_size=None, exp_idx=None, is_shuffle=True, normalize = None, nclass=None, seq_len_aps=None,
                 seq_len_dvs=None, seq_len_gps=None, seq_len_head = None, seq_len_time = None):
    dataset = SequentialDataset(data_dir=data_path,
                                exp_idx=exp_idx,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((240, 320)),
                                    torchvision.transforms.ToTensor(),
                                    normalize,
                                ]),
                                nclass=nclass,
                                seq_len_aps=seq_len_aps,
                                seq_len_dvs=seq_len_dvs,
                                seq_len_gps=seq_len_gps,
                                seq_len_head = seq_len_head,
                                seq_len_time = seq_len_time)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              shuffle=is_shuffle,
                                              drop_last=True,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=0)
    return data_loader

def Data_brightness(data_path=None, batch_size=None, exp_idx=None, is_shuffle=True, normalize = None, nclass=None, seq_len_aps=None,
                 seq_len_dvs=None, seq_len_gps=None, seq_len_head = None, seq_len_time = None):
    dataset = SequentialDataset(data_dir=data_path,
                                exp_idx=exp_idx,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((240, 320)),
                                    torchvision.transforms.ToTensor(),
                                    normalize,
                                    torchvision.transforms.ColorJitter(brightness=0.5),
                                ]),
                                nclass=nclass,
                                seq_len_aps=seq_len_aps,
                                seq_len_dvs=seq_len_dvs,
                                seq_len_gps=seq_len_gps,
                                seq_len_head=seq_len_head,
                                seq_len_time=seq_len_time)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              shuffle=is_shuffle,
                                              drop_last=True,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=8)
    return data_loader

def Data_mask(data_path=None, batch_size=None, exp_idx=None, is_shuffle=True, normalize = None, nclass=None, seq_len_aps=None,
                 seq_len_dvs=None, seq_len_gps=None, seq_len_head = None, seq_len_time = None):
    dataset = SequentialDataset(data_dir=data_path,
                                exp_idx=exp_idx,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((240, 320)),
                                    torchvision.transforms.ToTensor(),
                                    normalize,
                                    torchvision.transforms.RandomCrop(size=128, padding=128),
                                ]),
                                nclass=nclass,
                                seq_len_aps=seq_len_aps,
                                seq_len_dvs=seq_len_dvs,
                                seq_len_gps=seq_len_gps,
                                seq_len_head=seq_len_head,
                                seq_len_time=seq_len_time)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              shuffle=is_shuffle,
                                              drop_last=True,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              num_workers=8)
    return data_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res

def compute_matches(retrieved_all, ground_truth_info):
    matches=[]
    itr=0
    for retr in retrieved_all:
        if (retr == ground_truth_info[itr]):
            matches.append(1)
        else:
            matches.append(0)
        itr=itr+1
    return matches

def compute_precision_recall(matches,scores_all):
    precision, recall, _ = precision_recall_curve(matches, scores_all)
    return precision, recall
