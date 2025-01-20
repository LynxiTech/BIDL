# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
 
# Acknowledgments
# This code is built upon the implementation of the spikingjelly project, with substantial modifications.
# spikingjelly: [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)

from typing import Callable, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor,as_completed
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from .base_dvsdataset import DvsDatasetFolder,np_savez,max_threads_number_for_datasets_preprocess
import time
from .utils import load_events
import tqdm


class CIFAR10DVS(DvsDatasetFolder):
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
            ('airplane.zip', 'https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'),
            ('automobile.zip', 'https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'),
            ('bird.zip', 'https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'),
            ('cat.zip', 'https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'),
            ('deer.zip', 'https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'),
            ('dog.zip', 'https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'),
            ('frog.zip', 'https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'),
            ('horse.zip', 'https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'),
            ('ship.zip', 'https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'),
            ('truck.zip', 'https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')
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
            t, x, y, p = load_events(fp,
                        x_mask=0xfE,
                        x_shift=1,
                        y_mask=0x7f00,
                        y_shift=8,
                        polarity_mask=1,
                        polarity_shift=None)
            return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}

    @staticmethod
    def get_H_W() -> Tuple:
        return 128, 128
    
    @staticmethod
    def get_class_num() -> int:
        return 10
    
    @staticmethod
    def read_aedat_save_to_np(bin_file: str, np_file: str,pbar):
        events = CIFAR10DVS.load_origin_data(bin_file)
        np_savez(np_file,
                 t=events['t'],
                 x=events['x'],
                 y=events['y'],
                 p=events['p']
                 )
        pbar.update(1)

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        t_ckp = time.time()
        
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), max_threads_number_for_datasets_preprocess)) as tpe:
            total_files = 0
    
            
            for class_name in os.listdir(extract_root):
                aedat_dir = os.path.join(extract_root, class_name)
                if os.path.isdir(aedat_dir):  
                    bin_files = os.listdir(aedat_dir)
                    total_files += len(bin_files)
    
            with tqdm.tqdm(total=total_files, desc='Creating events_np files', unit='file') as pbar:
                for class_name in os.listdir(extract_root):
                    aedat_dir = os.path.join(extract_root, class_name)
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
                        class_futures.append(tpe.submit(CIFAR10DVS.read_aedat_save_to_np, source_file, target_file,pbar))
    
                    for future in as_completed(class_futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error in task: {e}")
    
    
        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        
    @staticmethod
    def class_to_label(class_name: str):
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_to_label_map = {cls: idx for idx, cls in enumerate(classes)}
        
        if class_name in class_to_label_map:
            return class_to_label_map[class_name]
        else:
            raise ValueError(f"Class name '{class_name}' not found in CLASSES.")