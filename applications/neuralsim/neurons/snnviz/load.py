'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

from math import inf

import numpy as np
import pandas as pd


def spike_pair_to_dict(spike_pair, fill_keys=False):
    spike_dict = {}
    for m, n in zip(*spike_pair):
        if n in spike_dict:
            spike_dict[n].append(m)
        else:
            spike_dict.update({n: [m]})
    if fill_keys is True:
        spike_dict.update({_: [] for _ in range(max(spike_dict.keys())) if _ not in spike_dict})
    spike_dict = {_: spike_dict[_] for _ in sorted(spike_dict)}
    return {_: np.array(__) for _, __ in spike_dict.items()}


def parse_egids(files: list, map_csv: str, resolut: float, trunc_n=(0, inf), trunc_t=(0, inf), with_dict=True):
    spike_dict = {}
    map_data = pd.read_csv(map_csv, header=None, low_memory=True).values
    mapping = dict(zip(map_data[:, 1], map_data[:, 0]))

    moments_all, numbers_all = [], []
    for file in files:
        records = np.fromfile(file, dtype='uint32')
        moments, numbers = records.reshape([-1, 2]).transpose([1, 0])  # in unit ``step``
        # moments = (moments.astype('float32') + np.array(1, dtype='float32')) * np.array(resolut, dtype='float32')
        moments = (moments + 1) * resolut  # change to unit ``sec``
        numbers = np.array([mapping[_] for _ in numbers], dtype='uint32')

        if trunc_n[0] > 1 or trunc_n[1] < inf:
            flag1 = np.logical_and(numbers >= trunc_n[0], numbers < trunc_n[1])
            idxs1 = np.where(flag1)
            moments, numbers = moments[idxs1], numbers[idxs1]
        if trunc_t[0] > 0 or trunc_t[1] < inf:
            flag2 = np.logical_and(moments >= trunc_t[0], moments < trunc_t[1])
            idxs2 = np.where(flag2)
            moments, numbers = moments[idxs2], numbers[idxs2]

        if with_dict is True:
            spike_dict.update(spike_pair_to_dict([moments, numbers]))

        moments_all.append(moments)
        numbers_all.append(numbers)

    spike_pair = np.concatenate(moments_all), np.concatenate(numbers_all)
    # if sort is True:
    #     idxs2 = np.argsort(moments)
    #     numbers, moments = numbers[idxs2], moments[idxs2]  # (z,), (z,)
    return spike_pair, spike_dict


def load_gids(files: list, resolut: float = None, trunc_n=(0, inf), trunc_t=(0, inf), with_dict=True):
    assert resolut is None
    spike_dict = {}

    moments_all, numbers_all = [], []
    for file in files:
        # XXX ``pd.read_csv`` is much faster than ``np.loadtxt``!!!
        records = pd.read_csv(file, sep='\t', header=None, usecols=[0, 1], low_memory=True).values
        numbers, moments = records.reshape([-1, 2]).transpose([1, 0])
        # moments = moments.astype('float32') * np.array(1 / 1000, dtype='float32')
        moments = moments * 1e-3
        numbers = numbers.astype('uint32')

        if trunc_n[0] > 1 or trunc_n[1] < inf:
            flag1 = np.logical_and(numbers >= trunc_n[0], numbers < trunc_n[1])
            idxs1 = np.where(flag1)
            moments, numbers = moments[idxs1], numbers[idxs1]
        if trunc_t[0] > 0 or trunc_t[1] < inf:
            flag2 = np.logical_and(moments >= trunc_t[0], moments < trunc_t[1])
            idxs2 = np.where(flag2)
            moments, numbers = moments[idxs2], numbers[idxs2]

        if with_dict is True:
            spike_dict.update(spike_pair_to_dict([moments, numbers]))

        moments_all.append(moments)
        numbers_all.append(numbers)

    spike_pair = np.concatenate(moments_all), np.concatenate(numbers_all)
    # if sort is True:
    #     idxs2 = np.argsort(moments)
    #     numbers, moments = numbers[idxs2], moments[idxs2]
    return spike_pair, spike_dict
