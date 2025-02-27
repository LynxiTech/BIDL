# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
import os
import pickle as pk
import cv2
import numpy as np
from datasets.base_dataset import BaseDataset

class RgbGesture(BaseDataset):
    CLASSES = ['Hand Clapping', 'Left Hand Wave', 'Right Hand Wave', 'Left Arm CW', 'Left Arm CCW', 'Right Arm CW',
        'Right Arm CCW', 'Arm Roll', 'Air Drums', 'Air Guitar', 'Other']

    def __init__(self,
            data_prefix, pipeline, classes=None, ann_file=None, test_mode=False,
            shape=None, jitter=1
    ):
        self.shape = shape
        self.jitter = jitter  # assert 0 <= jitter <= 1
        super(RgbGesture, self).__init__(data_prefix, pipeline, classes, ann_file, test_mode)

    def load_annotations(self):
        file = os.path.join(self.data_prefix, self.ann_file)
        with open(file, 'rb') as f:
            data, label = pk.load(f)

        if self.test_mode is False:
            jitt = np.random.randn(*data.shape[:2], 1, *data.shape[3:]) * self.jitter
            data_new = data + jitt.astype('int8')
            data = np.clip(np.concatenate([data, data_new], 0), 0, 255).astype('uint8')
            label = np.concatenate([label] * 2, 0)

        data_infos = []
        for dat, lbl in zip(data, label):
            if self.shape is not None:
                dat_ = []
                for _ in dat.transpose([0, 2, 3, 1]):
                    _ = cv2.resize(_, self.shape)
                    dat_.append(_)
                dat = np.array(dat_).transpose([0, 3, 1, 2])
            info = {
                'img': dat,  # .astype('float32'),
                'gt_label': np.array(lbl.item(), dtype='int64')
            }
            data_infos.append(info)

        return data_infos


########################################################################################################################


def main():
    """
    Generate more samples before training, mainly via rotation.
    Should be executed in the project home directory.
    """
    file_t = [
        './data/rgbgesture/train_num1.npy', './data/rgbgesture/train_label1.npy', './data/rgbgesture/train.pkl'
    ]
    file_v = [
        './data/rgbgesture/test_num1.npy', './data/rgbgesture/test_label1.npy', './data/rgbgesture/val.pkl'
    ]
    for dfi, lfi, fo in (file_t, file_v):
        dat = np.load(dfi)
        lbl = np.load(lfi)
        dat = np.clip(dat, 0, 255).astype('uint8')
        lbl = lbl.astype('uint8')

        dat2 = []
        print(f'augmenting {dfi}, {lfi}...')
        b, t, c, h, w = dat.shape
        for _ in dat.transpose([0, 1, 3, 4, 2]).reshape([b * t, h, w, c]):
            rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), np.random.choice([-10, +10]), 1.)
            _ = cv2.warpAffine(_, rot_mat, (w, h))   
            dat2.append(_)

        dat2 = np.array(dat2).transpose([0, 3, 1, 2]).reshape([b, t, c, h, w])
        print(f'saving {fo}...')
        dat3 = np.concatenate([dat, dat2], 0)
        lbl3 = np.concatenate([lbl] * 2, 0)

        with open(fo, 'wb') as f:
            pk.dump([dat3, lbl3], f)


if __name__ == '__main__':
    main()
