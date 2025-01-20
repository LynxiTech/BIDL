# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

import numpy as np
import bisect
import os
from copy import deepcopy

import h5py
import numpy as np
from torchvision.datasets.folder import default_loader


class LoadSpikesInHdf5(object):

    def __init__(self,
            mode: str,
            timestep=60, down_t=1, down_s=4, dt=25000, size=(2, 32, 32), offset=0, frame_jitter=25000 // 100
    ):
        self.mode = mode
        self.timestep = timestep
        self.down_t = down_t
        self.down_s = down_s
        self.dt = dt
        self.size = size
        self.offset = offset
        self.frame_jitter = frame_jitter  # for data augmentation
        assert 0 <= frame_jitter < dt / 2
        self.fd = None
        self.group = None

    def __call__(self, results):
        if self.fd is None:  # for multi-processing
            self.fd = h5py.File(results['prefix'], 'r', swmr=True, libver="latest")
            self.group = self.fd[self.mode]

        t0, t1 = results['range']
        if self.mode == 'train':
            t1 = t1 - self.timestep * self.dt
            start_time = np.random.randint(t0, t1 + 1)
        elif self.mode == 'test':
            start_time = t0 + self.offset * self.dt
        else:
            raise NotImplemented
        end_time = start_time + self.timestep * self.dt

        shoot = self.group[results['shoot']]
        times = shoot['time']
        addrs = shoot['data']

        i0 = bisect.bisect_left(times, start_time)
        i1 = bisect.bisect_left(times[i0:], end_time) + i0
        # move ``down_s``(``down_t``) out
        data = self.framing(times[i0:i1], addrs[i0:i1], self.dt, self.timestep, self.size, self.down_t, self.down_s,
            self.frame_jitter)
        results['img'] = data
        results['img_shape'] = data.shape
        results['range'] = start_time, end_time
        return results

    @staticmethod
    def framing(times, addrs, dt, timestep, size, down_t, down_s, jitter=0):
        t_start = times[0]
        cutts = np.arange(t_start, t_start + timestep * dt, dt)  # cutting points

        if jitter > 0:
            jitts = np.random.randint(-jitter, jitter + 1, len(cutts) - 1)
            jitts[-1] = -jitts[:-1].sum()
            diff = max(abs(jitts[-1]) - jitter, 0)

            if diff > 0:
                sign = np.sign(jitts[-1])
                j1 = diff // (len(jitts) - 1)
                j2 = diff % (len(jitts) - 1)
                jitts[:-1] += j1 * sign
                jitts[-1] = (jitter + j2) * sign

            assert jitts.sum() == 0
            np.random.shuffle(jitts)

            cutts[1:] += jitts
            assert np.all(cutts[1:] - cutts[:-1] > 0)

        frames = np.zeros((len(cutts),) + size, dtype='uint8')  # (t,c,h,w)

        idx0 = idx1 = 0
        for i, cutt in enumerate(cutts):
            idx1 += bisect.bisect_left(times[idx1:], cutt + dt)
            if idx1 > idx0:
                spikes = addrs[idx0:idx1]  # (t,3), where cols=(h,w,c)
                pol, xy = spikes[:, 2], (spikes[:, 0:2] / down_s).astype(frames.dtype)
                np.add.at(frames, (i, pol, xy[:, 0], xy[:, 1]), 1)
            idx0 = idx1

        return frames[::down_t]

class LoadFramesInFolder:

    def __init__(self,
            n_frame=16, down_t=(1,), img_loader=default_loader, dropout=0., random=True
    ):
        self.n_frame = n_frame
        self.down_t = down_t
        self.img_loader = img_loader  # its ``PIL``
        assert 0 <= dropout < 1
        self.dropout = dropout
        self.random = random

    @staticmethod
    def bell_rand(lower, upper, sigma, size=1):
        mean = (lower + upper) / 2
        xs = np.random.normal(mean, sigma, size)
        xs = np.where(np.logical_or(xs < lower, xs > upper), mean, xs)
        return xs

    @staticmethod
    def select_frames(n_frame, down_t, frame_files, random, trim_dt):
        length = len(frame_files)
        down_t = list(down_t)
        dt_max = length / n_frame
        if random is True:
            if trim_dt is True and down_t[1] > dt_max:
                down_t[1] = max(1 + 1e-5, dt_max)
                # print(f'``dt`` upper is too large; trim to {down_t[1]}')
            nargs = len(down_t)

            assert nargs in [2, 3] and down_t[0] < down_t[1]
            if nargs == 2:
                dt = np.random.rand() * (down_t[1] - down_t[0]) + down_t[0]
            elif nargs == 3:
                dt = LoadFramesInFolder.bell_rand(down_t[0], down_t[1], down_t[2])
            else:
                raise NotImplemented
        else:
            assert len(down_t) == 1
            dt = down_t[0]
            if trim_dt is True:
                dt = min(dt_max, dt)
        idxs0 = np.round(np.arange(0, n_frame) * dt).astype('int32')
        idx_delta = idxs0[-1] + 1
        if random is True:

            idx_start = np.random.choice(range(length - idx_delta + 1))
        else:
            idx_start = (length - idx_delta) // 2
        idxs = idx_start + idxs0
        frame_files2 = [frame_files[_] for _ in idxs]
        return frame_files2

    @staticmethod
    def repeat_to(frames, n_frame):
        length = len(frames)
        if n_frame <= length:
            return frames
        frames2 = deepcopy(frames)
        for i in range(n_frame - length):
            frames2.append(frames[i % length])
        return frames2

    def __call__(self, results):
        video_path = results['video']
        frame_files = results['frames']
        frame_files = self.repeat_to(frame_files, self.n_frame)
        frame_files2 = self.select_frames(self.n_frame, self.down_t, frame_files, self.random, trim_dt=True)
        frames = [self.img_loader(os.path.join(video_path, _)) for _ in frame_files2]
        clip = np.stack(frames, axis=0).transpose([0, 3, 1, 2])  # ``ImageToTensor`` will do this

        if self.dropout > 0 and self.random is True:
            clip = np.where(np.random.rand(clip.shape[0], 1, 1, 1) < self.dropout, 0, clip)

        results['img'] = clip
        return results


class LoadNumpy(object):

    def __init__(self,):
        self.fd = None
    
    def __call__(self,results):
        file = os.path.join(results['img_prefix'],results['img_info'])
        data = np.load(file)['arr_0']
        results["img"] = data.transpose([1,0,2,3])
        results["pack"] = data.shape
        results["gt_label"] = results["gt_label"]

        return results