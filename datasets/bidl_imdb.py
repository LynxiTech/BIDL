# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import numpy as np

from datasets.base_dataset import BaseDataset

class imdb(BaseDataset):

    CLASSES = ['neg', 'pos']

    def load_annotations(self):
        assert isinstance(self.ann_file, str) 

        data_infos = []
        data = np.load(self.data_prefix)
 
        label = np.load(self.ann_file)

        for dat, lbl in zip(data, label):
            
            info = {
                'img': dat, 'pack': dat.shape, 
                'gt_label': np.array(lbl, dtype='float')
            }
            
            data_infos.append(info)



        return data_infos




