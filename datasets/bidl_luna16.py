# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


import os
import random
import pandas as pd
import numpy as np
from datasets.base_dataset import BaseDataset

class Luna16Cls(BaseDataset):
    """
    LUNA (LUng Nodule Analysis) 16 - ISBI 2016 Challenge
    https://academictorrents.com/collection/luna-lung-nodule-analysis-16---isbi-2016-challenge
    classification/detection/segmentation...
    """
    CLASSES = ['malignant', 'benign']

    def load_annotations(self):
        data_infos = []

        for fn in os.listdir(self.data_prefix):
            file = os.path.join(self.data_prefix, fn)
            feat = np.load(file).transpose([0, 3, 1, 2])
            lbl = int(fn[0])
            info = {
                'img': feat, 'file': file,
                'gt_label': np.array(lbl, dtype=np.int64)
            }
            data_infos.append(info)

        return data_infos



class Luna16Cls32(BaseDataset):

    CLASSES = ['malignant', 'benign']

    def load_annotations(self):
        data_infos = []
        for fn in os.listdir(self.data_prefix):
            file = os.path.join(self.data_prefix, fn)
            temp = np.load(file)
            feat = np.expand_dims(temp, axis=3).transpose([0, 3, 1, 2])
            lbl = int(fn[0])
            info = {
                'img': feat, 'file': file,
                'gt_label': np.array(lbl, dtype=np.int64)
            }
            data_infos.append(info)

        
        return data_infos



class Luna16Det(BaseDataset):

    def load_annotations(self):
        raise NotImplementedError



class Luna16Seg(BaseDataset):

    def load_annotations(self):
        raise NotImplementedError

