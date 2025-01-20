# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import pylynchipsdk as sdk
import threading
import argparse
import sys; sys.path.append('../')
from lynadapter.lyn_sdk_model_multidevice import ApuRun
from lynadapter import lyn_sdk_model_multidevice
import sys
from utils.config import file2dict,pretty_text
import numpy as np
import os
import torch
from queue import Queue
import time
from torch.utils.data import DataLoader
from backbones import *
from datasets import *
from datasets.base_dataset import Compose
from datasets.bidl_dvsgesture import DVS128Gesture
from datasets.bidl_cifar10dvs import CIFAR10DVS
from datasets.bidl_dvsmnist import DvsMnist
from datasets.utils import split_to_train_test_set,TransformedDataset
from tqdm import tqdm
lyn_sdk_model_multidevice.via_p2p = False
lyn_sdk_model_multidevice.callback_trace = False
lyn_sdk_model_multidevice.time_record = False
time_record = lyn_sdk_model_multidevice.time_record

def get_data(data_name, data_set, cfg):
    """
    Load the test dataset based on the given data name and set.
 
    Parameters:
    data_name (str): The name of the dataset.
    data_set (str): The dataset set to load (e.g., 'test', 'val').
    cfg (dict): The configuration dictionary.
 
    Returns:
    tuple: A tuple containing the test dataset and the number of classes.
    """
    if 'dvsgesture' in data_name:
        data_root = os.path.join(cfg["data"]["data_root"], 'DvsGesture')
        T = cfg["model"]["backbone"]["timestep"]
        test_compose = Compose(cfg["data"]["test"]["pipeline"])
        test_dataset = DVS128Gesture(root=data_root, train=False, data_type='frame', frames_number=T, split_by='number')
        test_dataset = TransformedDataset(test_dataset.testdata, test_compose)
        classes_num = 11
 
    elif 'cifar10dvs' in data_name:
        data_root = os.path.join(cfg["data"]["data_root"], 'DVSCIFAR10')
        T = 10
        test_compose = Compose(cfg["data"]["test"]["pipeline"])
        dataset = CIFAR10DVS(root=data_root, data_type='frame', frames_number=T, split_by='number',test_only=True)
        test_dataset = TransformedDataset(dataset.testdata, test_compose)
        classes_num = 10
 
    elif 'dvsmnist' in data_name:
        data_root = os.path.join(cfg["data"]["data_root"], 'DvsMnist')
        T = 20
        test_compose = Compose(cfg["data"]["test"]["pipeline"])
        dataset = DvsMnist(root=data_root, data_type='frame', frames_number=T, split_by='number',test_only=True)
        test_dataset = TransformedDataset(dataset.testdata, test_compose)
        classes_num = 10
 
    else:
        val_data = cfg["data"][data_set]
        val_data_type = val_data.pop("type")
        try:
            test_dataset = eval(val_data_type)(**val_data)
        except Exception as e:
            raise ValueError(f"Failed to load dataset of type {val_data_type}: {e}")
        if val_data_type == 'imdb':
            classes_num=1
        else:
            classes_num = len(test_dataset.CLASSES)

    return test_dataset, classes_num

os.chdir('../')
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./applications/classification/dvs/cifar10-dvs/resnetlif50-cifar10dvs/resnetlif50-itout-b8x1-cifar10dvs_mp.py', help="test config file path")
args = parser.parse_args()
application_abbr = { "luna16cls": "three_D/luna16cls/clif3fc3lc/",
                         "clif3fc3dm_itout-b16x1-dvsmnist":"dvs/dvs_mnist/clif3fc3dm/",
                         "clifplus3fc3dm_itout-b16x1-dvsmnist":"dvs/dvs_mnist/clif3fc3dm/",
                         "clif5fc2dm_itout-b16x1-dvsmnist":"dvs/dvs_mnist/clif5fc2dm/",
                         "clif5fc2dm_it-b16x1-dvsmnist":"dvs/dvs_mnist/clif5fc2dm/",
                         "clif3flif2dg_itout-b16x1-dvsgesture":"dvs/dvs_gesture/clif3flif2dg/",
                         "clifplus3flifplus2dg_itout-b16x1-dvsgesture":"dvs/dvs_gesture/clif3flif2dg/",
                         "clif7fc1dg_itout-b16x1-dvsgesture":"dvs/dvs_gesture/clif7fc1dg/",
                         "clif7fc1dg_it-b16x1-dvsgesture":"dvs/dvs_gesture/clif7fc1dg/",
                         "clif5fc2cd_itout-b64x1-cifar10dvs": "dvs/cifar10dvs/vgg_cifar10dvs/clif5fc2cd/",
                         "clif5fc2cd_it-b64x1-cifar10dvs":"dvs/cifar10dvs/vgg_cifar10dvs/clif5fc2cd/",
                         "clif7fc1cd_itout-b64x1-cifar10dvs":"dvs/cifar10dvs/vgg_cifar10dvs/clif7fc1cd/",
                         "clif7fc1cd_it-b64x1-cifar10dvs":"dvs/cifar10dvs/vgg_cifar10dvs/clif7fc1cd/",
                         "clifplus5fc2cd_itout-b64x1-cifar10dvs": "dvs/cifar10dvs/vgg_cifar10dvs/clif5fc2cd/",
                         "attentionVGG-it-b64x1-cifar10dvs": "dvs/cifar10dvs/vgg_cifar10dvs/attentionVGG/",
                         "resnetlif18-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif18_cifar10dvs/",
                         "resnetlif50-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif50_cifar10dvs/",
                         "resnetlif50-itout-b8x1-cifar10dvs-mp": "dvs/cifar10dvs/resnetlif50_cifar10dvs/",
                         "resnetlif50-lite-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif50_lite_cifar10dvs/",
                         "resnetlif18-lite-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif18_lite_cifar10dvs/",
                         "resnetlif50-it-b16x1-cifar10dvs": "dvs/cifar10dvs/resnetlif50_cifar10dvs/",
                         "rgbgesture": "videodiff/rgbgesture/clif3flif2rg/",
                         "resnetlif18-itout-b20x4-16-jester": "video/jester/resnetlif18_t16/",
                         "resnetlif18-itout-b16x4-8-jester": "video/jester/resnetlif18_t8/",
                         "resnetlif18-lite-itout-b20x4-16-jester": "video/jester/resnetlif18_lite_t16/",
                         "esimagenet": "dvs/esimagenet/resnetlif18ES/",
                         "imdb":"text/imdb/fasttextIM/",
                         "mnist":"spikegen/clif3fc3mn/"}
data_name = args.config
print('args.config = ', args.config)
args.config = "./applications/classification/" + application_abbr[data_name] + args.config + '.py'  
filename = os.path.basename(args.config)
cfg = file2dict(args.config) 
print('args.lynxi_devices = ', cfg['lynxi_devices'])
count, ret = sdk.lyn_get_device_count()
assert ret == 0, 'lyn_get_device_count fail'
print('device count = {}'.format(count))
assert count > 0, "no device detected in environment"
devices = cfg['lynxi_devices']
dp_num = len(devices)
mp_num = len(devices[0])
print('dp_num = {}, mp_num = {}'.format(dp_num, mp_num))
device_num = dp_num * mp_num
assert device_num > 0 and device_num <= count and max(map(max, cfg['lynxi_devices'])) <= count - 1

opath = './model_files/classification/'
datasets_abbr = {"dvsmnist": "Dm",
                    "luna16cls": "Lc",
                    "dvsgesture": "Dg",
                    "cifar10dvs": "Cd",
                    "rgbgesture": "Rg",
                    "jester": "Jt"}
if mp_num == 1:
    cfg_mode = cfg['model']
elif mp_num > 1:
    cfg_mode = cfg['model_0']
if cfg_mode["backbone"]['type'] not in ['ResNetLifItout_MP', 'ResNetLifItout'] :
    network = cfg_mode["backbone"]['type']
else:
    if mp_num == 1:
        config_name_list = filename.split('/')[-1].split('-')
        dataset_key = config_name_list[3].split('.')[0]
        network = config_name_list[0].capitalize() + datasets_abbr[dataset_key] + config_name_list[1].capitalize()
    elif mp_num > 1:
        config_name_list = filename.split('/')[-1].split('-')
        dataset_key = config_name_list[3].split('.')[0].strip('_mp')
        network = config_name_list[0].capitalize() + datasets_abbr[dataset_key] + config_name_list[1].capitalize() + '_MP'
_base_ = os.path.join(opath, network)
mf_list = []
if mp_num == 1:
        model_path = os.path.join(f'{_base_}', "Net_0")
        print('model_path = ', model_path)
        assert os.path.exists(model_path)
        mf_list.append(model_path)
elif mp_num > 1:
    for i in range(mp_num):
        model_path = _base_ + '/model_{}'.format(i) + '/Net_0'
        print('model_path = ', model_path)
        assert os.path.exists(model_path)
        mf_list.append(model_path)

def input_preprocess_thread(*args):
    thread_id = args[0]
    print('thread id {}: {}'.format(thread_id, sys._getframe().f_code.co_name))
    q_list = args[1]
    data_loader = args[2]
    pre_filled = args[3]
    idex=args[4]
    for i, data in enumerate(data_loader):
        if i < pre_filled: continue
        data_img = data[idex]
        data_img = np.array(data_img).astype(np.float32)
        tmp_spilt_list = np.array_split(data_img, dp_num, 1)
        for k in range(dp_num):
            q_list[k].put(tmp_spilt_list[k])

start_time_arr = np.zeros((dp_num, mp_num))
end_time_arr = np.zeros((dp_num, mp_num))
def apu_infer_thread(*args):
    thread_id = args[0]
    print('thread id {}: {}'.format(thread_id, sys._getframe().f_code.co_name))
    group_id = args[1]
    device_id = args[2]
    apurun_list = args[3]
    times = args[4]
    dataset = args[5]
    q_list = args[6]
 
    model_backbone_type = cfg_mode["backbone"]["type"]
    if "FastTextItout" in model_backbone_type:
        classes_num = 1
    else:
        classes_num = cfg_mode["backbone"]["nclass"]

    results = []
    for i in tqdm(range(times)):
        if i == 0 and time_record == True:
            start_time_arr[group_id][device_id] = time.time()
        if device_id == mp_num - 1 and group_id == dp_num - 1:
            iter_start_time = time.time()
        if device_id == 0:
            img = q_list[group_id].get(block=True)
            apurun_list[group_id][device_id].run(img)
        else:
            apurun_list[group_id][device_id].run()
        if device_id == mp_num - 1 and group_id == dp_num - 1:
            output = apurun_list[group_id][device_id].get_output()[:, 0:classes_num]
            results.append(output)
        if device_id == mp_num - 1 and group_id == dp_num - 1:
            if time_record:
                iter_end_time = time.time()
                iter_infer_time = apurun_list[group_id][device_id].iter_infer_time
                iter_time = (iter_end_time - iter_start_time) * 1000
                time_per = iter_infer_time / iter_time
                sys.stdout.write(', iter_infer_time: {:.2f} ms, iter_time: {:.2f} ms, perenctage of inference: {:.2%} \r'\
                    .format(iter_infer_time, iter_time, time_per))
        if device_id == 0:
            q_list[group_id].task_done()
        if i == times - 1 and time_record == True:
            end_time_arr[group_id][device_id] = time.time()
    if device_id == mp_num - 1 and group_id == dp_num - 1:
        if "FastTextItout" in model_backbone_type:
            eval_results = dataset.evaluate(torch.sigmoid(torch.tensor(results)))
        else:
            eval_results = dataset.evaluate(results)
        for k, v in eval_results.items():
            print(f'\n{k} : {v:.2f}')

if __name__ == "__main__":
    thread_id = 0
    apurun_list = []
    thread_list = []
    q_list = []
    timesteps_list = []
    spilt_list = []
    
    # build dataset
    dataset, classes_num = get_data(data_name, "test", cfg)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        num_workers=1,        
        shuffle=False)
    
    dl_len = len(data_loader)
    print('dataloader length = ', dl_len)

    idex = 0 if "dvsgesture" in data_name or "cifar10dvs" in data_name or "dvsmnist" in data_name else "img"

    # create queue for each device inference thread
    for i in range(dp_num):
        q = Queue(maxsize=100)
        q.queue.clear()
        q_list.append(q)

    get_split = True
    pre_filled = 50
    # fill pre_filled number into queue first 
    for i, data in enumerate(data_loader):
        if i == pre_filled: break
        data_img = data[idex]
        data_img = np.array(data_img).astype(np.float32)
        assert dp_num <= data_img.shape[1], "failed to split timesteps, since target split number is " + \
            "more than timesteps of input data, input shape = {}".format(data_img.shape)
        tmp_spilt_list = np.array_split(data_img, dp_num, 1)
        for k in range(dp_num):
            if get_split:
                timesteps_list.append(tmp_spilt_list[k].shape[1])
                spilt_list.append(tmp_spilt_list[k].shape)
            q_list[k].put(tmp_spilt_list[k])
        if get_split: get_split = False

    print('timesteps are divided by the number of devices:')
    for i in range(dp_num):
        print(spilt_list[i])

    # create preprocess thread
    thread_list.append(
        threading.Thread(target=input_preprocess_thread, \
            args=(thread_id, q_list, data_loader, pre_filled,idex)))
    thread_id += 1

    # create apuruns
    apurun_num = 0
    for g in range(dp_num):
        tmp_list = []
        for i in range(mp_num):
            print('create ApuRun object {}'.format(apurun_num))
            apurun = ApuRun(g, i, devices, mf_list[i], timesteps_list[g])
            tmp_list.append(apurun)
            apurun_num += 1
        apurun_list.append(tmp_list)

    # create inference threads
    for g in range(dp_num):
        for i in range(mp_num):
            thread_list.append(threading.Thread(target=apu_infer_thread, \
                args=(thread_id, g, i, apurun_list, dl_len, dataset, q_list)))
            thread_id += 1

    inference_start = time.time()
    for t in thread_list:
        t.start()

    for i in range(dp_num):
        q_list[i].join()

    for t in thread_list:
        t.join()
    inference_end = time.time()
    fps = (dl_len * timesteps_list[0]) / (inference_end - inference_start)
    print('apu test speed = {:.2f} fps'.format(fps))

    # calculate perenctage of inference time
    if time_record:
        run_time_cost = end_time_arr - start_time_arr
        print('\n### perenctage of inference time ###', end='\n\n')
        for g in range(dp_num):
            print('group_id {}: '.format(g), end='')
            for i in range(mp_num):
                time_per = apurun_list[g][i].infer_time / (run_time_cost[g][i]*1000)
                print('{:.2%}'.format(time_per), end=' ')
            print('\n')
        print('####################################\n')

    # release sdk resource 
    apurun_num = 0
    for g in range(dp_num):
        for i in range(mp_num):
            print('release sdk resource {}'.format(apurun_num))
            apurun_list[g][i].release()
            apurun_num += 1
