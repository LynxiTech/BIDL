# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
 
# Acknowledgments
# This code is built upon the implementation of the spikingjelly project, with substantial modifications.
# spikingjelly: [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)

from torch.utils.data import Dataset
from torchvision.datasets import utils
from typing import Callable, Dict, Optional, Tuple
from abc import abstractmethod
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time
import math
import tqdm
from pathlib import Path  
from losses.accuracy import *
from typing import Any
from .utils import replicate_directory_structure,load_npz_frames,integrate_events_file_to_frames_file_by_fixed_frames_number
save_datasets_compressed = True
max_threads_number_for_datasets_preprocess = 16
import re


np_savez = np.savez_compressed if save_datasets_compressed else np.savez

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset,
                            num_classes: int, random_split: bool = False,
                            save_path: str = None, load_path: str = None):
    # If a load path is provided and the file exists, try to load the indices
    if load_path and os.path.exists(load_path + '_train_idx.npy') and os.path.exists(load_path + '_test_idx.npy'):
        train_idx = np.load(load_path + '_train_idx.npy')
        test_idx = np.load(load_path + '_test_idx.npy')
        return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)
 
    # Initialize an empty list of numpy arrays to store indices for each class
    label_idx = [np.array([], dtype=int) for _ in range(num_classes)]
    
    # Iterate over the dataset to collect indices for each class
    # If you don't need progress bar, remove tqdm.tqdm
    for i, item in enumerate(tqdm.tqdm(origin_dataset)):
        y = item[1]
        if isinstance(y, (np.ndarray, torch.Tensor)):
            y = y.item()  # Convert the label to a single integer
        # Use numpy's append (still not the most efficient, but better than list append)
        label_idx[y] = np.append(label_idx[y], i)
    
    # If required, randomly shuffle the indices for each class
    if random_split:
        for idx_array in label_idx:
            np.random.shuffle(idx_array)
    
    # Split the indices based on the training ratio
    # Using numpy arrays directly
    train_idx = np.concatenate([idx_array[:math.ceil(len(idx_array) * train_ratio)] for idx_array in label_idx])
    test_idx = np.concatenate([idx_array[math.ceil(len(idx_array) * train_ratio):] for idx_array in label_idx])
 
    # If a save path is provided, save the indices to a file
    if save_path:
        np.save(save_path + '_train_idx.npy', train_idx)
        np.save(save_path + '_test_idx.npy', test_idx)
 
    # Return the training and test subsets
    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

def is_non_empty_dir(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            files = os.listdir(path)
            
            if files:
                return True
    return False

class DvsDatasetFolder(Dataset):
    """
    A custom dataset folder class for loading and preparing DVS (Dynamic Vision Sensor) data.
    Inherits from torchvision's DatasetFolder to facilitate loading of data in a folder structure.
    """
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
        """
        Initializes the DvsDatasetFolder class with specified parameters.
        
        Args:
            root (str): Root directory where the dataset is stored.
            train (bool): Specifies if the dataset is for training or testing.
            data_type (str): Type of data to load ('event' or 'frame').
            frames_number (int): Number of frames to integrate from events.
            split_by (str): Specifies how to split events into frames ('time' or 'event_count').
            duration (int): Duration for which to integrate events into frames.
            custom_integrate_function (Callable): Custom function to integrate events into frames.
            custom_integrated_frames_dir_name (str): Name of the directory for custom integrated frames.
            transform (Optional[Callable]): Function/transform that takes in a PIL image and returns a transformed version.
            target_transform (Optional[Callable]): Function/transform that takes in the target and returns a transformed version.
        """
        self.root = root  
        self.train = train  
        self.data_type = data_type  
        self.frames_number = frames_number  
        self.split_by = split_by  
        self.duration = duration  
        self.custom_integrate_function = custom_integrate_function  
        self.custom_integrated_frames_dir_name = custom_integrated_frames_dir_name  
        self.transform = transform  
        self.target_transform = target_transform  
  
        self.prepare_data()  
        if self.train is not None:
            if self.train:
                _, _, train_data = self.get_loader_and_root(train_ratio,test_only)
                self.traindata=train_data
                self.testdata=None
            else:
                _, _, test_data = self.get_loader_and_root(train_ratio,test_only)
                self.traindata=None
                self.testdata=test_data
        else:
            if test_only:
                _, _, test_data = self.get_loader_and_root(train_ratio,test_only)
                self.traindata=None
                self.testdata=test_data
            else:
                _, _, train_data,test_data = self.get_loader_and_root(train_ratio,test_only)
                self.traindata=train_data
                self.testdata=test_data
        


    def prepare_data(self):  
        """
        Prepare the data by downloading, extracting, and processing if necessary.
        """
        events_np_root = os.path.join(self.root, 'events_np')  
        if not os.path.exists(events_np_root):  
            self.download_and_prepare_data(events_np_root)  
  
    def download_and_prepare_data(self, events_np_root: str): 
        """
        Download and prepare the dataset by downloading, extracting, and converting files.
        
        Args:
            events_np_root (str): Directory where the processed .npz event files will be stored.
        """ 
        download_root = os.path.join(self.root, 'download')  
        extract_root = os.path.join(self.root, 'extract')  
   
        if not os.path.exists(download_root):  
            os.makedirs(download_root)  
            resource_list = self.resource_url_md5()
            if self.downloadable():
                for i in range(resource_list.__len__()):
                    file_name, url, md5 = resource_list[i]
                    print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                    utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
            else:
                raise NotImplementedError(f'This dataset can not be downloaded '
                                        f'please download files manually and put files at [{download_root}]. '
                                        f'The resources file_name, url, and md5 are: \n{resource_list}')
        else:  
            self.check_and_download_files(download_root)  
   
        if not os.path.exists(extract_root):  
            os.makedirs(extract_root)  
            self.extract_downloaded_files(download_root, extract_root)  
   
        if not os.path.exists(events_np_root):  
            os.makedirs(events_np_root)  
            self.create_events_np_files(extract_root, events_np_root)  
  
    def check_and_download_files(self, download_root: str):  
        """
        Check the integrity of downloaded files and download any missing or corrupted files.
        
        Args:
            download_root (str): Directory where downloaded files are stored.
        """ 
        resource_list = self.resource_url_md5()  
        for file_name, url, md5 in resource_list:  
            fpath = os.path.join(download_root, file_name)  
            if not utils.check_integrity(fpath=fpath, md5=md5):
                print(f'The file [{fpath}] does not exist or is corrupted.')
                if os.path.exists(fpath):  
                    os.remove(fpath)  
                if self.downloadable():
                    print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                    utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(
                        f'This dataset can not be downloaded, please download [{file_name}] from [{url}] manually and put files at {download_root}.')

#9.23
    def get_loader_and_root(self,train_ratio,test_only) -> Tuple[Callable, str]: 
        """
        Get the appropriate loader and data root based on the data type.
        
        Returns:
            Tuple[Callable, str]: A tuple containing the loader function and the data root directory.
        """
        H, W = self.get_H_W()  
        events_np_root = os.path.join(self.root, 'events_np')
        if self.train is not None:
            if self.train:
                events_np_root = os.path.join(events_np_root, 'train')
            else:
                events_np_root = os.path.join(events_np_root, 'test') 
        if self.data_type == 'event':  
            return np.load, events_np_root

        elif self.data_type == 'frame':  
            if self.train is not None:
                if self.train:
                    frames_np_root,train_data = self.get_frames_np_root(events_np_root,H, W,train_ratio,test_only)  
                    return load_npz_frames, frames_np_root,train_data   
                else:   
                    frames_np_root,test_data = self.get_frames_np_root(events_np_root,H, W,train_ratio,test_only)  
                    return load_npz_frames, frames_np_root,test_data  
            else:
                if test_only:
                    frames_np_root,test_data = self.get_frames_np_root(events_np_root,H, W,train_ratio,test_only)  
                    return load_npz_frames, frames_np_root,test_data   
                else:
                    frames_np_root,train_data,test_data = self.get_frames_np_root(events_np_root,H, W,train_ratio,test_only)  
                    return load_npz_frames, frames_np_root,train_data,test_data   
  
        else:  
            raise ValueError("Invalid data type.")  
  
    def get_frames_np_root(self, events_np_root,H: int, W: int,train_ratio,test_only) -> str:
        """
        Get the root directory for frames based on the specified frame integration parameters.
        
        Args:
            events_np_root (str): Directory where event files are stored.
            H (int): Height of the frames.
            W (int): Width of the frames.
            
        Returns:
            str: Root directory for the integrated frames.
        """ 
        if self.frames_number is not None:  
            frames_np_root = os.path.join(self.root, f'frames_number_{self.frames_number}_split_by_{self.split_by}')
            if os.path.exists(frames_np_root):
                pass
            else:
                os.mkdir(frames_np_root)
            if self.train is not None:
                if self.train:
                    frames_np_root = os.path.join(frames_np_root, 'train')
                else:
                    frames_np_root = os.path.join(frames_np_root, 'test')
            if self.train is not None:
                if self.train:
                    train_data=self.prepare_frames_by_number(frames_np_root, events_np_root,H, W,train_ratio,test_only) 
                    return frames_np_root,train_data
                else:
                    test_data=self.prepare_frames_by_number(frames_np_root, events_np_root,H, W,train_ratio,test_only) 
                    return frames_np_root,test_data
            else:
                if test_only:
                    test_data=self.prepare_frames_by_number(frames_np_root, events_np_root,H, W,train_ratio,test_only)  
                    return frames_np_root,test_data
                else:
                    train_data,test_data=self.prepare_frames_by_number(frames_np_root, events_np_root,H, W,train_ratio,test_only) 
                    return frames_np_root,train_data,test_data 
        else:  
            raise ValueError("'frames_number' must be provided.")  

    def prepare_frames_by_number(self,frames_np_root, events_np_root,H, W,train_ratio=0.8,test_only=False):

        if is_non_empty_dir(frames_np_root):
            print(f'The frames files [{frames_np_root}] already exists and is not empty.')
            if self.train is not None:
                if self.train:
                    train_dataset=np.load(frames_np_root+'/train_datas.npy',allow_pickle=True)
                    return train_dataset.tolist()
                else:
                    test_dataset=np.load(frames_np_root+'/test_datas.npy',allow_pickle=True)
                    return test_dataset.tolist()
            else:  
                if test_only:
                    test_dataset=np.load(frames_np_root+'/test_datas.npy',allow_pickle=True)
                    return test_dataset
                else:
                    train_dataset=np.load(frames_np_root+'/train_datas.npy',allow_pickle=True)
                    test_dataset=np.load(frames_np_root+'/test_datas.npy',allow_pickle=True)
                    return train_dataset,test_dataset
        else:
            if os.path.exists(frames_np_root):
                pass
            else:
                os.mkdir(frames_np_root)
                print(f'Mkdir [{frames_np_root}].')
            t_ckp = time.time()
            frames_list=[]
            targets_list=[]
            for e_root, e_dirs, e_files in os.walk(events_np_root):
                if e_files.__len__() > 0:
                    e_files.sort()
                    for e_file in tqdm.tqdm(e_files, desc=f'Processing files in {e_root}', unit='file'):
                        events_np_file = os.path.join(e_root, e_file)
                        frames=integrate_events_file_to_frames_file_by_fixed_frames_number(self.load_events_np, events_np_file, None, self.split_by, self.frames_number, H, W, True)
                        frames=frames.astype(np.uint8).copy()
                        target=self.class_to_label(e_root.split("/")[-1])
                        frames_list.append(frames)
                        targets_list.append(target)
                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
            zipped_list = list(zip(frames_list, targets_list))
            sorted_zipped_list = sorted(zipped_list, key=lambda x: x[1])
            if self.train is not None:
                classes_num=self.get_class_num()
                if self.train:
                    train_dataset=np.asanyarray(sorted_zipped_list, dtype=object)
                    np.save(frames_np_root+'/train_datas.npy', train_dataset)
                    return train_dataset
                else:
                    test_dataset =np.asanyarray(sorted_zipped_list, dtype=object)
                    np.save(frames_np_root+'/test_datas.npy', test_dataset)   
                    return test_dataset
            else:
                classes_num=self.get_class_num()
                train_dataset, test_dataset = split_to_train_test_set(train_ratio, sorted_zipped_list, classes_num, random_split=True,save_path=self.root + '/data_idex', load_path=self.root + '/data_idex')
                if test_only:
                    print("saving test frames files...")
                    test_dataset=np.asanyarray(test_dataset, dtype=object)
                    np.save(frames_np_root+'/test_datas.npy', test_dataset)   
                    return test_dataset
                else:
                    print("saving frames files...")
                    train_dataset=np.asanyarray(train_dataset, dtype=object)  
                    test_dataset=np.asanyarray(test_dataset, dtype=object)
                    np.save(frames_np_root+'/train_datas.npy', train_dataset)
                    np.save(frames_np_root+'/test_datas.npy', test_dataset) 

                    return train_dataset,test_dataset


    def save_frames_to_npz_and_print(fname: str, frames):
        np_savez(fname, frames=frames)
        print(f'Frames [{fname}] saved.')

    @staticmethod
    @abstractmethod
    def resource_url_md5() -> list:
        pass

    @staticmethod
    @abstractmethod
    def downloadable() -> bool:
        pass

    @staticmethod
    @abstractmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        pass

    @staticmethod
    @abstractmethod
    def create_events_np_files(extract_root: str, events_np_root: str):

        pass

    @staticmethod
    @abstractmethod
    def class_to_label(class_name: str):
        pass
    
    @staticmethod
    @abstractmethod
    def get_H_W() -> Tuple:

        pass
    
    @staticmethod
    @abstractmethod
    def get_class_num() -> int:
       pass

    @staticmethod
    def load_events_np(fname: str):

        return np.load(fname)
    
    def get_gt_labels(self):
        labels=[]
        for index in range(len(self.samples)):
            _, target = self.samples[index]
            labels.append(target)
        return np.array(labels)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = self.data_set[index]
        target = self.target[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 ):
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy'
        ]
        eval_results = {}
        results = np.vstack(results)
   
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        gt_labels = gt_labels[:num_imgs]
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
                
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})


        return eval_results