# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
 
# Acknowledgments
# This code is built upon the implementation of the spikingjelly project, with substantial modifications.
# spikingjelly: [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)

import os
from typing import Callable, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor,as_completed
from torchvision.datasets.utils import extract_archive
import multiprocessing
from .base_dvsdataset import DvsDatasetFolder,np_savez,max_threads_number_for_datasets_preprocess
import time
from .utils import load_events,load_aedat
import tqdm

class DvsMnist(DvsDatasetFolder):
    def __init__(
            self,
            root: str,
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

        super().__init__(root, None, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform,
                         target_transform,train_ratio,test_only)
    @staticmethod
    def resource_url_md5() -> list:
        return [
            ('dat2mat.m', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/dat2mat.m', None),
            ('grabbed_data0.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data0.zip',  None),
            ('grabbed_data1.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data1.zip',  None),
            ('grabbed_data2.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data2.zip',  None),
            ('grabbed_data3.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data3.zip',  None),
            ('grabbed_data4.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data4.zip',  None),
            ('grabbed_data5.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data5.zip',  None),
            ('grabbed_data6.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data6.zip',  None),
            ('grabbed_data7.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data7.zip',  None),
            ('grabbed_data8.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data8.zip',  None),
            ('grabbed_data9.zip', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/grabbed_data9.zip',  None),
            ('mat2dat.m', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/mat2dat.m',  None),
            ('process_mnist.m', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/process_mnist.m',  None),
            ('readme.txt', 'http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/readme.txt',  None) 
        ]

    @staticmethod
    def downloadable() -> bool:
        return True

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):

        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 10)) as tpe:
            for zip_file in os.listdir(download_root):
                zip_file = os.path.join(download_root, zip_file)
                print(f'Extract [{zip_file}] to [{extract_root}].')
                tpe.submit(extract_archive, zip_file, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        with open(file_name, 'rb') as fp:
            t, x, y, p = load_events(fp)
            return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}

    @staticmethod
    def get_H_W() -> Tuple:
        return 128, 128
    
    @staticmethod
    def get_class_num() -> int:
        return 10
    
    @staticmethod
    def read_aedat_save_to_np(bin_file: str, np_file: str,pbar):
        events = DvsMnist.load_origin_data(bin_file)
        np_savez(np_file,
                 t=events['t'],
                 x=events['x'],
                 y=events['y'],
                 p=events['p']
                 )
        pbar.update(1)
    '''
    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), max_threads_number_for_datasets_preprocess)) as tpe:
            for class_name in os.listdir(extract_root):
                aedat_dir = os.path.join(extract_root, class_name+'/scale16')
                np_dir = os.path.join(events_np_root, class_name)
                os.mkdir(np_dir)
                print(f'Mkdir [{np_dir}].')
                bin_files = os.listdir(aedat_dir)
                for bin_file in tqdm.tqdm(bin_files, desc=f'Converting files in {class_name}', unit='file'):
                    source_file = os.path.join(aedat_dir, bin_file)
                    target_file = os.path.join(np_dir, os.path.splitext(bin_file)[0] + '.npz')
                    #print(f'Start to convert [{source_file}] to [{target_file}].')
                    tpe.submit(DvsMnist.read_aedat_save_to_np, source_file,target_file)
        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
    '''
    
    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        t_ckp = time.time()
        
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), max_threads_number_for_datasets_preprocess)) as tpe:
            total_files = 0
    
            for class_name in os.listdir(extract_root):
                aedat_dir = os.path.join(extract_root, class_name+'/scale16')
                if os.path.isdir(aedat_dir):  
                    bin_files = os.listdir(aedat_dir)
                    total_files += len(bin_files)
    
            with tqdm.tqdm(total=total_files, desc='Creating events_np files', unit='file') as pbar:
                for class_name in os.listdir(extract_root):
                    aedat_dir = os.path.join(extract_root, class_name+'/scale16')
                    if not os.path.isdir(aedat_dir):
                        continue  
    
                    np_dir = os.path.join(events_np_root, class_name)
                    os.makedirs(np_dir, exist_ok=True)
                    #print(f'Mkdir [{np_dir}].')
    
                    bin_files = os.listdir(aedat_dir)
                    class_futures = []
    
    
                    for bin_file in bin_files:
                        source_file = os.path.join(aedat_dir, bin_file)
                        target_file = os.path.join(np_dir, os.path.splitext(bin_file)[0] + '.npz')
                        class_futures.append(tpe.submit(DvsMnist.read_aedat_save_to_np, source_file, target_file,pbar))
    
                    for future in as_completed(class_futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error in task: {e}")
    
    
        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

    @staticmethod    
    def class_to_label(class_name: str):
        classes = ['grabbed_data0', 'grabbed_data1', 'grabbed_data2', 'grabbed_data3', 'grabbed_data4', 'grabbed_data5', 'grabbed_data6', 'grabbed_data7', 'grabbed_data8', 'grabbed_data9']
        class_to_label_map = {cls: idx for idx, cls in enumerate(classes)}
        
        if class_name in class_to_label_map:
            return class_to_label_map[class_name]
        else:
            raise ValueError(f"Class name '{class_name}' not found in CLASSES.")