# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import argparse
import os
from tqdm import tqdm  
import numpy as np
import torch
import sys
sys.path.append("../")
from functools import partial 
from torch.utils.data import DataLoader
from applications.classification import *
from datasets import *
from datasets.base_dataset import Compose
from datasets.bidl_dvsgesture import DVS128Gesture
from datasets.bidl_cifar10dvs import CIFAR10DVS
from datasets.bidl_dvsmnist import DvsMnist
from datasets.utils import split_to_train_test_set,TransformedDataset
from losses.accuracy import *
import time
from layers import lif,lifplus
from utils import globals
from utils.collate import collate
from utils.config import file2dict
import json
import ast
import re
globals._init()

torch._dynamo.config.disable = True
  

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--use_lyngor', type=int, help='use lyngor flag: 1 or 0, means if use lyngor or not')
    parser.add_argument('--use_legacy', type=int, help='use legacy flag: 1 or 0, means if use legacy or not')
    parser.add_argument('--multinet', default=0, type=int, help='multi net flag: 1 or 0, means if multi net or not')
    parser.add_argument('--c', default=0, type=int, help='compile only flag: 1 or 0, means if compile only or not')
    parser.add_argument('--v', default=0, type=int, help='compile version flag: 1 or 0, means compile v1 or v0')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--b', default=1, type=int,help='test batch size')
    parser.add_argument('--device', default=0,type=str, help='APU Device IDs as a comma-separated list or single value. egg: 1 for apu_1, [1,2] for apu_1 and apu_2')
    parser.add_argument('--mode', default=0, type=int, help='to specify where the data loaded.0 for automode, 1 for global buffer, 2 for ddr')
    parser.add_argument('--post_mode', default=None, type=int, help='to specify where to use post_mode in compile')
    parser.add_argument('--profiler', default=False, type=bool, help='Switch for lyngor profiler analynsis. ONLY function when model needs to be compiled on APU.')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--gpu-ids',
        default=0,
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
             f'result file. You can choose some of ({", ".join(out_options)}), '
             'or use "all" to include all above, or use "none" to disable all of '
             'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
             '"accuracy", "precision", "recall", "f1_score", "support" for single '
             'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
             'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.b > 1 or args.mode==1:
        args.v = 1
    return args

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

from layers import lif
from layers import lifplus

lif.spike_func = lambda _: torch.gt(_, 0.).to(_.dtype)
lifplus.spike_func = lambda _: torch.gt(_, 0.).to(_.dtype)

def load_config(args, USE_LYNGOR):
    """
    Load the configuration and initialize the model.
 
    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    USE_LYNGOR (bool): Flag indicating whether to use Lyngor APU.
 
    Returns:
    tuple: A tuple containing the model, configuration dictionary, input size, timestep, network name, and model backbone type.
    """
    # Load the configuration file
    filename = os.path.basename(args.config)
    cfg = file2dict(args.config)
    args.metrics = 'accuracy'  # Default metrics to accuracy
 
    # Set cudnn benchmark mode if specified in the configuration
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
 
    # Ensure either metrics or output path is specified
    assert args.metrics or args.out, 'Please specify at least one of output path and evaluation metrics.'
 
    # Set global values based on arguments
    if args.use_lyngor == 1:
        globals.set_value('ON_APU', True)
        globals.set_value('FIT', True)
        if 'soma_params' in cfg["model"]["backbone"] and cfg["model"]["backbone"]['soma_params'] == 'channel_share':
            globals.set_value('FIT', False) 
    if args.multinet == 1:
        globals.set_value('MULTINET', True)
    else:
        globals.set_value('MULTINET', False)
    globals.set_value('MODE', args.mode)
 
    # Load the model backbone
    if "it_batch" in cfg["model"]["backbone"]:
        cfg["model"]["backbone"]["it_batch"]=1
    backbone_dict = cfg["model"]["backbone"]
    model_backbone_type = backbone_dict.pop('type')
    model = eval(model_backbone_type)(**backbone_dict)
 
    # Adjust samples per GPU based on arguments and Lyngor usage
    cfg["data"]["samples_per_gpu"] = args.b if USE_LYNGOR else cfg["data"]["samples_per_gpu"]
 
    # Determine input size and timestep
    t = cfg["model"]["backbone"]["timestep"]
    if model_backbone_type in ["FastTextItout", "FastTextlifplusItout"]:
        c = cfg["model"]["backbone"].get("vocab_size", None)
        in_size = [((1, c),)] if c is not None else None
    else:
        c = cfg["model"]["backbone"]["input_channels"]
        h = cfg["model"]["backbone"]["h"]
        w = cfg["model"]["backbone"]["w"]
        in_size = [((t, 1, c, h, w),)] if "it-" in args.config else [((1, c, h, w),)]
 
    # Determine network name based on configuration and dataset
    datasets_abbr = {
        "dvsmnist": "Dm",
        "luna16cls": "Lc",
        "dvsgesture": "Dg",
        "cifar10dvs": "Cd",
        "rgbgesture": "Rg",
        "jester": "Jt",
        "esimagenet": "Es",
        "imdb": "Imdb",
        "mnist": "mn"
    }
    config_name_list = filename.split('/')[-1].split('-')
    dataset_key = config_name_list[-1].split('.')[0]
    network_base = config_name_list[0].capitalize() + datasets_abbr[dataset_key]
    if model_backbone_type in ['ResNetLifItout', 'ResNetLifReluItout']:
        network = network_base + config_name_list[1].capitalize()
    else:
        network = network_base
 
    return model, cfg, in_size, t, network, model_backbone_type


def load_data(model_backbone_type, data_name, cfg, USE_LYNGOR, chip_id=[0]):
    """
    Load the dataset and create a data loader for inference.
 
    Parameters:
    model_backbone_type (str): The type of model backbone.
    data_name (str): The name of the dataset.
    cfg (dict): Configuration dictionary.
    USE_LYNGOR (bool): Flag indicating whether to use Lyngor APU.
    chip_id (list): List of APU chip IDs to use.
 
    Returns:
    tuple: A tuple containing the data loader, number of classes, dataset object, and None.
    """
    # Determine the dataset to use based on conditions
    if model_backbone_type in ["FastTextItout", "FastTextlifplusItout"] and not USE_LYNGOR:
        data_set = "val"
    elif model_backbone_type in ["FastTextItout", "FastTextlifplusItout"]:
        data_set = "test"
    elif cfg["dataset_type"] == "Jester20bn" and not USE_LYNGOR:
        data_set = "test_gpu"
    else:
        data_set = "test"
 
    # Load the dataset and number of classes
    val_datasets, classes_num = get_data(data_name, data_set, cfg)
 
    # Adjust the samples per GPU if using Lyngor APU
    if USE_LYNGOR:
        cfg["data"]["samples_per_gpu"] *= len(chip_id)
    else:
        cfg["data"]["samples_per_gpu"] = 1
    # Create the data loader
    data_loader = DataLoader(
        val_datasets,
        batch_size=cfg["data"]["samples_per_gpu"],
        sampler=None,
        num_workers=cfg["data"]["workers_per_gpu"],
        collate_fn=partial(collate, samples_per_gpu=cfg["data"]["samples_per_gpu"]),
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )
 
    return data_loader, classes_num, val_datasets, None

    
def apu_infer_multi(model_backbone_type, model_path, data_loader, dataset, chip_id, t, classes_num, args, data_name):
    """
    Perform multi-APU inference for a given model.
 
    Parameters:
    model_backbone_type (str): The type of model backbone.
    model_path (str): The path to the model directory.
    data_loader (torch.utils.data.DataLoader): The data loader for inference.
    dataset (Dataset): The dataset object used for evaluation.
    chip_id (list): List of APU chip IDs to use for inference.
    t (int): Number of timesteps or frames.
    classes_num (int): Number of classes in the dataset.
    args (argparse.Namespace): Parsed command-line arguments.
    data_name (str): The name of the dataset.
 
    Returns:
    dict: A dictionary containing evaluation results.
    """
    batch_size = args.b
    sys.path.append("./")
    from lynadapter.lyn_sdk_model import ApuRun
    if "Itout" not in model_backbone_type:
        t_in_apu = 1
    else:
        t_in_apu = t
    model_path = os.path.join(model_path, "Net_0")
    path_apu_json = os.path.join(model_path, "apu_0/apu_x/apu.json")
    # Load APU configuration
    with open(path_apu_json) as file:
        json_data = json.load(file)
    if json_data['apu_x']['batch_size'] != json_data['apu_x']['n_slice']:
        raise ValueError('Models with unequal numbers of batch and n_slice are not supported yet.  Please reduce the batchsize number！')
 
    # Initialize APU runs
    aruns = []
    for chip in chip_id:
        aruns.append(ApuRun(chip, model_path, t_in_apu))
 
    # Determine data index based on dataset type
    idex = 0 if "dvsgesture" in data_name or "cifar10dvs" in data_name or "dvsmnist" in data_name else "img"
 
    # Measure APU run time
    apu_run_time = time.time()
    pp_time_cost = 0
    pp_per_start = time.time()
 
    # Loop through the data loader
    for i, data in enumerate(tqdm(data_loader)):
        pp_per_end = time.time()
        pp_time_cost += pp_per_end - pp_per_start
 
        # Preprocess data
        data_img = data[idex].transpose(1, 0)
        for c in range(len(chip_id)):
            aruns[c].run(data_img[:, batch_size * c:batch_size * (c + 1), ...].numpy())
        pp_per_start = time.time()
 
    # Collect APU outputs
    outputs = []  
    for c in range(len(chip_id)):
        outputs.append(aruns[c].get_output())
    apu_finish_time = time.time()
    # Process and concatenate outputs
    results = []
    for i in tqdm(range(len(data_loader))):
        for c in range(len(chip_id)):
            o = outputs[c][i][0][-1]
            o1 = o.reshape(-1, classes_num)
            results.append(o1)
 
    results = np.concatenate(results, axis=0)
 
    # Evaluate the results
    if model_backbone_type in ["FastTextItout", "FastTextlifplusItout"]:
        eval_results = dataset.evaluate(torch.sigmoid(torch.tensor(results)))
    else:
        eval_results = dataset.evaluate(results)
 
    # Print evaluation results
    for k, v in eval_results.items():
        print(f'\n{k} : {v:.2f}')
 
    # Calculate and print test speed
    test_speed = len(data_loader) * t * batch_size * len(chip_id) / (apu_finish_time - apu_run_time)
    print(f'APU test speed = {test_speed:.4f} fps')
 
    return eval_results



def cpu_infer(model, data_name, data_loader, args, CLASSES, t):
    """
    Perform model inference on CPU or GPU.
 
    Parameters:
    model (torch.nn.Module): The defined model.
    data_name (str): The name of the dataset.
    data_loader (torch.utils.data.DataLoader): The data loader.
    args (argparse.Namespace): Parsed command-line arguments.
    CLASSES (list): List of class names or labels.
    t (int): Number of timesteps or frames.
 
    Returns:
    dict: A dictionary containing evaluation results.
    """
    # Load the model checkpoint
    state_dict = torch.load(args.checkpoint, map_location='cpu')["state_dict"]
    # Remove the "module." prefix if the model was trained with DataParallel
    new_state_dict1 = {}
    new_state_dict2 = {}
    new_state_dict3 = {}
            
    for key, value in state_dict.items():
        new_key = re.sub(r'^module\.', '', key, count=1)
        new_state_dict1[new_key] = value
    if "resnet" in args.config:
        mapped_state_dict = map_keys(new_state_dict1,args.config)
    else:
        mapped_state_dict = new_state_dict1

    for key, value in mapped_state_dict.items():  
        new_key1 = key.replace("backbone.", "")                             
        new_state_dict2[new_key1] = value
    for key, value in new_state_dict2.items():  
        new_key2 = key.replace("unit.", "")                             
        new_state_dict3[new_key2] = value
    model.load_state_dict(new_state_dict3,strict=False) 
    # Optionally compile the model if using PyTorch 2.0 or later
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)
 
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
 
    # Set up evaluation parameters
    num_epochs = 1
    start_time = time.time()
    preprocessing_time = [0]
    validation_accuracy = [0., 0.]
    topk = (1,) if 'imdb' in data_name or 'luna16cls' in data_name else (1,5)
 
    # Loop through the data loader
    start_prep_time = time.time()
    #pdb.set_trace()
    for batch_idx, data in enumerate(tqdm(data_loader)):
        # Measure preprocessing time
        end_prep_time = time.time()
        preprocessing_time[0] += end_prep_time - start_prep_time
 
        # Extract inputs and labels based on dataset type
        if "dvsgesture" in data_name or "cifar10dvs" in data_name or "dvsmnist" in data_name:
            inputs, labels = data[0], data[1]
        else:
            inputs, labels = data["img"], data["gt_label"]
 
        # Move data to the correct device
        inputs, labels = inputs.to(device), labels.to(device)
 
        # Perform inference
        with torch.no_grad():
            result = model(inputs)
            acc = accuracy(result, labels, topk=topk)
 
        # Accumulate accuracy metrics
        for i in range(len(acc)):
            validation_accuracy[i] += acc[i]
 
        # Update preprocessing time start
        start_prep_time = time.time()
 
    # Calculate final evaluation results
    eval_results = {
        f'accuracy_top-{k}': validation_accuracy[i].item() / (batch_idx + 1)
        for i, k in enumerate(topk)
    }
 
    # Print evaluation results
    
    for key, value in eval_results.items():
        print(f'\n{key} : {value:.2f}')
    
    # Calculate and print test speed
    end_time = time.time()
    test_speed = len(data_loader.dataset) * t * num_epochs * args.b / (end_time - start_time)
    print(f'test speed = {test_speed:.4f} fps')
 
    return eval_results


def parse_device_list(device_str):
    try:
        device_list = ast.literal_eval(device_str)
        if not isinstance(device_list, list):
            raise ValueError
    except (ValueError, SyntaxError):
        device_list = [int(device_str)]
    return device_list

def map_keys(state_dict,config):
    
    mapped_state_dict = {}
    
    if "it-" in config:
        patterns = [
            (re.compile(r'^conv.(\d+)\.(.+)$'), lambda m: f'conv.module.{m.group(1)}.{m.group(2)}'),
            (re.compile(r'^(.*?)\.conv(\d+)\.(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.conv{m.group(2)}.module.{m.group(3)}.{m.group(4)}'),
            (re.compile(r'^(.*?)\.downsample.(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.downsample.module.{m.group(2)}.{m.group(3)}'),
            #(re.compile(r'^([^.]+)\.lif.(.+)$'), lambda m: f'{m.group(1)}.lif.lif.{m.group(2)}'),
            #(re.compile(r'^(.*?)\.lif(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.lif{m.group(2)}.lif.{m.group(3)}')
        ]
    elif "cifar10" in config:
        return state_dict
    else:
        patterns = [
            (re.compile(r'^([^.]+)\.lif.(.+)$'), lambda m: f'{m.group(1)}.lif.lif.{m.group(2)}'),
            (re.compile(r'^(.*?)\.lif(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.lif{m.group(2)}.lif.{m.group(3)}')
        ]
    
    
    for key, value in state_dict.items():
        for pattern, mapper in patterns:
            match = pattern.match(key)
            if match:
                mapped_key = mapper(match)
                mapped_state_dict[mapped_key] = value
                break  
        else:
            
            mapped_state_dict[key] = value
    
    return mapped_state_dict

def main():
    os.chdir('../')
    args = parse_args()
    assert args.use_lyngor in (0, 1), 'use_lyngor must in (0, 1)'
    assert args.use_legacy in (0, 1), 'use_legacy must in (0, 1)'
    assert args.c in (0, 1), 'c must in (0, 1)'
    assert args.v in (0, 1), 'v must in (0, 1)'
    args.device = parse_device_list(args.device)
    assert isinstance(args.device, list), 'apu chip id must be list'
    USE_LYNGOR = True if args.use_lyngor == 1 else False
    USE_LEGACY = True if args.use_legacy == 1 else False
    COMPILE_ONLY = True if args.c == 1 else False
    post_mode = args.post_mode
    profiler = args.profiler
        
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
    data_name = args.config.split("-")[-1]
    if data_name in ["jester", "cifar10dvs","dvsgesture","dvsmnist"]:
        data_name = args.config
    
    args.config = "./applications/classification/" + application_abbr[data_name] + args.config + '.py' 
    model_file_path = "./model_files/classification/" 
    
    if "it-" in args.config:
        checkpoint_file_path = "./weight_files/classification/" + application_abbr[data_name].split("/")[-2]
    else:
        checkpoint_file_path = "./weight_files/classification/" + application_abbr[data_name].split("/")[-2]
    

    if args.checkpoint:        
        if "plus" in args.config:
            args.checkpoint = checkpoint_file_path + "/lifplus/" + args.checkpoint
        else:
            args.checkpoint = checkpoint_file_path + "/lif/" + args.checkpoint

    model,cfg,in_size,t, network,model_backbone_type= load_config(args,USE_LYNGOR)
    if args.b != 1:
        model_file = model_file_path  + network + "_" + str(args.b)
    else:        
        model_file = model_file_path  + network

    if USE_LYNGOR is True:
        if not USE_LEGACY: 
            state_dict = torch.load(args.checkpoint, map_location='cpu')["state_dict"]


            new_state_dict1 = {}
            new_state_dict2 = {}
            new_state_dict3 = {}
            
            for key, value in state_dict.items():
                new_key = re.sub(r'^module\.', '', key, count=1)
                new_state_dict1[new_key] = value
            '''
            if "it-" in args.config:
                mapped_state_dict = map_keys(new_state_dict1)
            else:
            '''
            if "resnet" in args.config:
                mapped_state_dict = map_keys(new_state_dict1,args.config)
            else:
                mapped_state_dict = new_state_dict1

            for key, value in mapped_state_dict.items():  
                new_key1 = key.replace("backbone.", "")                             
                new_state_dict2[new_key1] = value
            for key, value in new_state_dict2.items():  
                new_key2 = key.replace("unit.", "")                             
                new_state_dict3[new_key2] = value
            model.load_state_dict(new_state_dict3,strict=False)      
            sys.path.append("./")
            from lynadapter.lyn_compile import model_compile
            model_compile(model.eval(),model_file,in_size,args.v,args.b,input_type="float16",post_mode=post_mode, profiler=profiler)
        elif COMPILE_ONLY:
            print("COMPILE_ONLY switches on，no need to perform APU inference operations")        
        
        if len(args.device)>1:
            chip_id = [int(item) for item in args.device]
        else:
            chip_id = args.device
        

        if not COMPILE_ONLY:
            data_loader, classes_num,dataset,CLASSES= load_data(model_backbone_type, data_name,cfg,USE_LYNGOR,chip_id) 
            apu_infer_multi(model_backbone_type,model_file,data_loader, dataset, chip_id,t,classes_num,args,data_name)

    else:
        data_loader, classes_num,dataset,CLASSES= load_data(model_backbone_type, data_name,cfg,USE_LYNGOR)
        cpu_infer(model, data_name,data_loader,args,CLASSES,t)


if __name__ == '__main__':

    main()
