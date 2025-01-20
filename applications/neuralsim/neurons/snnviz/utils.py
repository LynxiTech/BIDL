'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import os
from multiprocessing import Process

import numpy as np


def concur(funcs: list, argvs: list):
    """Does not support ``lambda`` as ``funcs``!"""
    works = []
    for func, argv in zip(funcs, argvs):
        p = Process(target=func, args=argv)
        p.start()
        works.append(p)
    for p in works:
        p.join()


def match_files_in_path(path_pattern: str):
    assert isinstance(path_pattern, str)
    if os.path.isfile(path_pattern):
        files = [path_pattern]
    elif os.path.isdir(path_pattern):
        fns = os.listdir(path_pattern)
        files = [os.path.join(path_pattern, _) for _ in fns]
        files = [_ for _ in files if os.path.isfile(_)]
    else:
        prefix, fn = os.path.split(os.path.abspath(path_pattern))
        with os.popen(f'find {prefix} -maxdepth 1 -type f -name {fn}') as fd:
            fns = fd.readlines()
        files = [os.path.join(prefix, _.strip()) for _ in fns]
    return files


def calc_norm_diff(gt: np.ndarray, pr: np.ndarray, ord=2):
    """Calculate normalized (of some order) relative difference."""
    assert len(gt.shape) in (1, 2)
    assert gt.shape == pr.shape
    nmrt = np.linalg.norm(gt - pr, ord=ord)
    dnmrt = np.linalg.norm(gt, ord=ord)
    diff = nmrt / (dnmrt+1e-5)
    return diff

def spike_mask_to_pair(spike_mask: np.ndarray, resolut=0.1):
    """
    :param spike_mask: in shape (b=1,t,...)  # TODO ???
    :param resolut:
    """
    assert len(spike_mask.shape) >= 3
    assert spike_mask.shape[0] == 1
    spike_mask = np.reshape(spike_mask, [*spike_mask.shape[:2], -1])[0]  # (t,...)
    moments, numbers = np.where(spike_mask)  # ([moment,..], [number,..])
    moments = moments * resolut
    return moments, numbers
