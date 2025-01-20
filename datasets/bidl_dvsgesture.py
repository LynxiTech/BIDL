# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
 
# Acknowledgments
# This code is built upon the implementation of the spikingjelly project, with substantial modifications.
# spikingjelly: [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)

import os
import os.path

import numpy as np

from typing import Callable, Dict, Optional, Tuple
import numpy as np
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import tqdm
from .base_dvsdataset import DvsDatasetFolder,np_savez,max_threads_number_for_datasets_preprocess

from .utils import load_aedat


class DVS128Gesture(DvsDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            train_ratio:int =0.8,
            test_only:bool = False
    ) -> None:
       
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform,train_ratio,test_only)
    @staticmethod
    def resource_url_md5() -> list:
        url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'
        return [
            ('DvsGesture.tar.gz', url, '8a5c71fb11e24e5ca5b11866ca6c00a1'),
            ('gesture_mapping.csv', url, '109b2ae64a0e1f3ef535b18ad7367fd1'),
            ('LICENSE.txt', url, '065e10099753156f18f51941e6e44b66'),
            ('README.txt', url, 'a0663d3b1d8307c329a43d949ee32d19')
        ]

    @staticmethod
    def downloadable() -> bool:
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract [{fpath}] to [{extract_root}].')
        extract_archive(fpath, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        return load_aedat(file_name)

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str ,progress_bar):
        events = DVS128Gesture.load_origin_data(aedat_file)
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

        label_file_num = [0] * 11

        for i in range(csv_data.shape[0]):
            label = csv_data[i][0] - 1
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np_savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            label_file_num[label] += 1
            
        progress_bar.update(1)

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        aedat_dir = os.path.join(extract_root, 'DvsGesture')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir}, {test_dir}].')
        for label in range(11):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')
    
        
        with open(os.path.join(aedat_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            

            train_files = [line.strip() for line in trials_to_train_txt if line.strip()]
            test_files = [line.strip() for line in trials_to_test_txt if line.strip()]
            total_files = len(train_files) + len(test_files)

            progress_bar = tqdm.tqdm(total=total_files, desc="Processing files")
    
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), max_threads_number_for_datasets_preprocess)) as tpe: 
    
                for fname in train_files:
                    aedat_file = os.path.join(aedat_dir, fname)
                    fname = os.path.splitext(fname)[0]
                    tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '_labels.csv'), train_dir,progress_bar)
    
                for fname in test_files:
                    aedat_file = os.path.join(aedat_dir, fname)
                    fname = os.path.splitext(fname)[0]
                    tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '_labels.csv'), test_dir,progress_bar)
    
            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        
        print(f'All aedat files have been split to samples and saved into [{train_dir}, {test_dir}].')
        
    @staticmethod
    def get_H_W() -> Tuple:
        return 128, 128
    
    @staticmethod
    def get_class_num() -> int:
        return 11
    
    @staticmethod
    def class_to_label(class_name: str):
        classes = ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8','9']
        class_to_label_map = {cls: idx for idx, cls in enumerate(classes)}
        
        if class_name in class_to_label_map:
            return class_to_label_map[class_name]
        else:
            raise ValueError(f"Class name '{class_name}' not found in CLASSES.")
        
    @staticmethod
    def get_data_type() -> int:
        return float