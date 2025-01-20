# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
 
# Acknowledgments
# This code is built upon the implementation of the spikingjelly project, with substantial modifications.
# spikingjelly: [https://github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)

import numpy as np
from typing import Callable
import numpy as np
import os
import struct
import torch
import math
import tqdm
import gzip
import hashlib
import os
import os.path
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
import time
from pathlib import Path  
from torch.utils.data import Dataset
from losses.accuracy import *
EVT_DVS = 0 
 
y_mask = 0x7F00
y_shift = 8

x_mask =0xFE
x_shift = 1

polarity_mask =1
polarity_shift = 0

valid_mask = 0x80000000
valid_shift = 31

def skip_header(fp):  
    skipped_bytes = 0  
    while True:  
        line = fp.readline()   
        if not line:  
            break   
        try:  
            decoded_line = line.decode().strip()  
        except UnicodeDecodeError:   
            break  
        if not decoded_line.startswith("#"):  
            break  
        skipped_bytes += len(line)  
    return skipped_bytes

def read_bits(arr, mask=None, shift=None):  
    arr = arr & mask if mask is not None else arr  
    arr = arr >> shift if shift is not None else arr  
    return arr

def load_raw_events(file_handle, skip_amount=0, trim_amount=0, filter_dvs=False, timestamps_priority=False):  
    
    header_bytes_skipped = skip_header(file_handle)  
    file_handle.seek(header_bytes_skipped + skip_amount)  
      
    raw_data_chunk = file_handle.read()  
    if trim_amount > 0:  
        raw_data_chunk = raw_data_chunk[:-trim_amount]  
      
    data_stream = np.frombuffer(raw_data_chunk,  dtype='>u4').reshape(-1, 2)  
    addresses_list = data_stream[:, 0]  
    timestamps_list = data_stream[:, 1]  
   
    if timestamps_priority:  
        sorted_data = (timestamps_list, addresses_list)  
    else:  
        sorted_data = (addresses_list, timestamps_list)  
   
    if filter_dvs:  
        valid_flags = read_bits(addresses_list, valid_mask, valid_shift) == EVT_DVS  
        sorted_data = tuple(arr[valid_flags] for arr in sorted_data)  
      
    return sorted_data 

def load_events(fp,filter_dvs=False,                      
                      x_mask=x_mask,
                      x_shift=x_shift,
                      y_mask=y_mask,
                      y_shift=y_shift,
                      polarity_mask=polarity_mask,
                      polarity_shift=polarity_shift):
    addr,timestamp = load_raw_events(fp,filter_dvs=filter_dvs)
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return timestamp, x, y, polarity


def load_npz_frames(file_name: str) -> np.ndarray:
    return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)
    
def load_aedat(file_name: str) -> dict:  
    txyp_lists = {'t': [], 'x': [], 'y': [], 'p': []}  
  
    with open(file_name, 'rb') as file:  
         
        line = file.readline()  
        while line.startswith(b'#'):  
            if line == b'#!END-HEADER\r\n':  
                break  
            line = file.readline()  
   
        while True:  
            header = file.read(28)  
            if not header:  
                break  

            #event_type, event_size, timestamp_overflow, capacity = struct.unpack('<HIII', header)
            
            event_type = struct.unpack('H', header[0:2])[0]
            event_size = struct.unpack('I', header[4:8])[0]
            timestamp_overflow = struct.unpack('I', header[12:16])[0]
            capacity = struct.unpack('I', header[16:20])[0] 

            data_length = capacity * event_size  
            data = file.read(data_length)  
  
            if event_type == 1:  
                for i in range(0, len(data), event_size):  
                    aer_data = struct.unpack('<I', data[i:i+4])[0]  
                    timestamp = struct.unpack('<I', data[i+4:i+8])[0] | (timestamp_overflow << 31)  
                    x = (aer_data >> 17) & 0x7FFF  
                    y = (aer_data >> 2) & 0x7FFF  
                    polarity = (aer_data >> 1) & 1  
  
                    txyp_lists['t'].append(timestamp)  
                    txyp_lists['x'].append(x)  
                    txyp_lists['y'].append(y)  
                    txyp_lists['p'].append(polarity)  
            else:  
                continue  
   
        txyp = {  
        't': np.array(txyp_lists['t'], dtype=np.int32),  
        'x': np.array(txyp_lists['x'], dtype=np.int16),  
        'y': np.array(txyp_lists['y'], dtype=np.int16),  
        'p': np.array(txyp_lists['p'], dtype=np.uint8)  
    }  
    return txyp  

def replicate_directory_structure(source_dir: Path, target_dir: Path) -> None:   
    target_dir.mkdir(parents=True, exist_ok=True)  
    for sub_dir in source_dir.iterdir():  
        if sub_dir.is_dir():  
            target_sub_dir = target_dir / sub_dir.name  
            target_sub_dir.mkdir(parents=True, exist_ok=True)  
            print(f'Mkdir [{target_sub_dir}].')  
            replicate_directory_structure(sub_dir, target_sub_dir)  


def integrate_events_file_to_frames_file_by_fixed_frames_number(  
    loader: Callable,  
    events_np_file: str,  
    output_dir: str,  
    split_by: str,  
    frames_num: int,  
    H: int,  
    W: int,  
    print_save: bool = False  
 ) -> None:  
    # Load events from file using the provided loader function  
    events = loader(events_np_file)  
      
    # Extract event data  
    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))  
      
    # Calculate segment indices  
    if split_by == 'number':  
        N = t.size  
        di = N // frames_num  
        j_l = np.arange(frames_num) * di  
        j_r = j_l + di  
        j_r[-1] = N  # Ensure the last segment includes all remaining events  
    elif split_by == 'time':  
        dt = (t[-1] - t[0]) // frames_num  
        j_l = np.zeros(frames_num, dtype=int)  
        j_r = np.zeros(frames_num, dtype=int)  
        idx = np.arange(N)  
        for i in range(frames_num):  
            t_l = dt * i + t[0]  
            t_r = t_l + dt  
            mask = np.logical_and(t >= t_l, t < t_r)  
            idx_masked = idx[mask]  
            j_l[i] = idx_masked[0] if idx_masked.size > 0 else j_l[i-1] + 1  # Handle case with no events in segment  
            j_r[i] = idx_masked[-1] + 1 if idx_masked.size > 0 else j_r[i-1]  # Ensure non-decreasing j_r  
        j_r[-1] = N  # Ensure the last segment includes all remaining events  
    else:  
        raise NotImplementedError("split_by must be 'number' or 'time'")  
      
    # Initialize frames array  
    frames = np.zeros([frames_num, 2, H, W])  
      
    # Integrate events into frames  
    for i in range(frames_num):  
        frame = np.zeros(shape=[2, H * W])  
        x_segment = x[j_l[i]:j_r[i]].astype(int)  
        y_segment = y[j_l[i]:j_r[i]].astype(int)  
        p_segment = p[j_l[i]:j_r[i]]  
          
        mask_pos = p_segment == 0  
        mask_neg = np.logical_not(mask_pos)  
          
        for c, mask in enumerate([mask_pos, mask_neg]):  
            position = y_segment[mask] * W + x_segment[mask]  
            events_number_per_pos = np.bincount(position, minlength=H * W)  
            frame[c][np.arange(events_number_per_pos.size)] += events_number_per_pos  
          
        frames[i] = frame.reshape((2, H, W))  

    return frames


class TrainDataset(Dataset):
    def __init__(self, origin_dataset, train_idx):
        self.origin_dataset = origin_dataset
        self.train_idx = train_idx

    def __getitem__(self, index):
        return self.origin_dataset[self.train_idx[index]]

    def __len__(self):
        return len(self.train_idx)

class TestDataset(Dataset):
    def __init__(self, origin_dataset, test_idx):
        self.origin_dataset = origin_dataset
        self.test_idx = test_idx

    def __getitem__(self, index):
        return self.origin_dataset[self.test_idx[index]]

    def __len__(self):
        return len(self.test_idx)

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

class TransformedDataset(torch.utils.data.Dataset):  
    def __init__(self, original_dataset, transform):  
        self.original_dataset = original_dataset  
        self.transform = transform  
  
    def __getitem__(self, index):  
        sample = self.original_dataset[index]  
        data, target = sample  
        if self.transform is not None:  
            data = self.transform(data)
        return data, target  
  
    def __len__(self):  
        return len(self.original_dataset)  
    
    def get_gt_labels(self):
        samples = self.original_dataset
        labels=[]
        for index in range(len(samples)):
            _, target = samples[index]
            labels.append(target)
        return np.array(labels)
    
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
    





__all__ = ['rm_suffix', 'check_integrity', 'download_and_extract_archive']


def rm_suffix(s, suffix=None):
    if suffix is None:
        return s[:s.rfind('.')]
    else:
        return s[:s.rfind(suffix)]


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_url_to_file(url, fpath):
    with urllib.request.urlopen(url) as resp, open(fpath, 'wb') as of:
        shutil.copyfileobj(resp, of)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from.
        root (str): Directory to place downloaded file in.
        filename (str | None): Name to save the file under.
            If filename is None, use the basename of the URL.
        md5 (str | None): MD5 checksum of the download.
            If md5 is None, download without md5 check.
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f'Using downloaded and verified file: {fpath}')
    else:
        try:
            print(f'Downloading {url} to {fpath}')
            download_url_to_file(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      f' Downloading {url} to {fpath}')
                download_url_to_file(url, fpath)
            else:
                raise e
        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError('File not found or corrupted.')


def _is_tarxz(filename):
    return filename.endswith('.tar.xz')


def _is_tar(filename):
    return filename.endswith('.tar')


def _is_targz(filename):
    return filename.endswith('.tar.gz')


def _is_tgz(filename):
    return filename.endswith('.tgz')


def _is_gzip(filename):
    return filename.endswith('.gz') and not filename.endswith('.tar.gz')


def _is_zip(filename):
    return filename.endswith('.zip')

def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(
            to_path,
            os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, 'wb') as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError(f'Extraction of {from_path} not supported')

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(url,
                                 download_root,
                                 extract_root=None,
                                 filename=None,
                                 md5=None,
                                 remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f'Extracting {archive} to {extract_root}')
    extract_archive(archive, extract_root, remove_finished)