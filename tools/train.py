# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import argparse
import os
import os.path as osp
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
import sys
sys.path.append("../")
from utils.config import file2dict,pretty_text
from utils.logging import get_logger
from utils.apis import set_random_seed,worker_init_fn
from utils.warm_up import *
from utils.collate import collate
from applications.classification import *
from datasets import *
from datasets.base_dataset import Compose
from datasets.bidl_dvsgesture import DVS128Gesture
from datasets.bidl_cifar10dvs import CIFAR10DVS
from datasets.bidl_dvsmnist import DvsMnist
from datasets.utils import split_to_train_test_set,TransformedDataset
from torch.utils.data import DataLoader,DistributedSampler
from torch.optim import *
from torch.optim.lr_scheduler import *
from losses.label_smooth_loss import *
from losses.accuracy import *
import torch.multiprocessing as mp
from torch import distributed as dist
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel
from datasets.pipelines.bidl_formating import *
from datasets.pipelines.bidl_loading import *
from datasets.pipelines.bidl_transforms import *
import re 

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='device used for training')
    group_gpus.add_argument('--gpus', type=int, help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

def setup_environment(args):
    """
    Setup the environment variables based on the provided arguments.
 
    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    """
    # Set the CUDA_VISIBLE_DEVICES environment variable to limit GPU access
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_ids))
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # Remove the variable if not set
 
    # Change the working directory to the parent directory
    os.chdir('../')
 
    # Set the LOCAL_RANK environment variable if not already set
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

def configure_logger(cfg):
    """
    Configure the logger based on the provided configuration.
 
    Parameters:
    cfg (dict): Configuration dictionary containing logging parameters.
 
    Returns:
    logging.Logger: The configured logger object.
    """
    # Create a timestamp for the log file name
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # Construct the log file path
    log_file = os.path.join(cfg["work_dir"], f'{timestamp}.log')
    
    # Ensure the work directory exists
    os.makedirs(cfg["work_dir"], exist_ok=True)
    
    # Get the logger with the specified log file and log level
    logger = get_logger(log_file=log_file, log_level=cfg["log_level"])
    
    return logger

def load_config(args):
    """
    Load the configuration file and update it with command-line arguments.
 
    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
 
    Returns:
    tuple: A tuple containing the configuration dictionary and the data name.
    """
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
 
    config_path = os.path.join("./applications/classification/", application_abbr[data_name], args.config + '.py')
    cfg = file2dict(config_path)
    scaler = None
    fp16_cfg = cfg.get('fp16', None) 
    if fp16_cfg:  
        try:  
            from torch.cuda.amp import GradScaler, autocast  
        except ImportError:  
            raise ImportError("Please install torch >= 1.6 to use mixed precision training.")  
        scaler = GradScaler(init_scale=cfg['fp16']['loss_scale']) 
    # Update work directory
    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir
    elif 'work_dir' not in cfg:
        cfg["work_dir"] = os.path.join('./work_dirs', os.path.splitext(os.path.basename(config_path))[0])
 
    # Update resume from path
    if args.resume_from is not None:
        cfg["resume_from"] = args.resume_from
 
    # Update GPU IDs
    if args.gpu_ids is not None:
        cfg["gpu_ids"] = args.gpu_ids
    elif args.gpus is not None:
        cfg["gpu_ids"] = range(args.gpus)
    else:
        cfg["gpu_ids"] = range(1)
 
    return cfg, data_name,scaler

def get_data(data_name, cfg):
    """
    Load the training and testing datasets based on the given data name and configuration.
 
    Parameters:
    data_name (str): The name of the dataset.
    cfg (dict): The configuration dictionary containing dataset parameters.
 
    Returns:
    tuple: A tuple containing the training and testing datasets.
    """
    if 'dvsgesture' in data_name:
        T = 16
        data_root = os.path.join(cfg["data"]["data_root"], 'DvsGesture')
        train_compose = Compose(cfg["data"]["train"]["pipeline"])
        test_compose = Compose(cfg["data"]["test"]["pipeline"])
        train_dataset = DVS128Gesture(root=data_root, train=True, data_type='frame', frames_number=T, split_by='number')
        test_dataset = DVS128Gesture(root=data_root, train=False, data_type='frame', frames_number=T, split_by='number')
        train_dataset=TransformedDataset(train_dataset.traindata, train_compose)
        test_dataset = TransformedDataset(test_dataset.testdata, test_compose)
        nclass = 11
 
    elif 'cifar10dvs' in data_name:
        T = 10
        data_root = os.path.join(cfg["data"]["data_root"], 'DVSCIFAR10')
        train_compose = Compose(cfg["data"]["train"]["pipeline"])
        test_compose = Compose(cfg["data"]["test"]["pipeline"])
        dataset = CIFAR10DVS(root=data_root, data_type='frame', frames_number=T, split_by='number',test_only=False)
        train_dataset = TransformedDataset(dataset.traindata, train_compose)
        test_dataset = TransformedDataset(dataset.testdata, test_compose)
 
    elif 'dvsmnist' in data_name:
        T = 20
        data_root = os.path.join(cfg["data"]["data_root"], 'DvsMnist')
        train_compose = Compose(cfg["data"]["train"]["pipeline"])
        test_compose = Compose(cfg["data"]["test"]["pipeline"])
        dataset = DvsMnist(root=data_root, data_type='frame', frames_number=T, split_by='number',test_only=False)
        train_dataset = TransformedDataset(dataset.traindata, train_compose)
        test_dataset = TransformedDataset(dataset.testdata, test_compose)
 
    else:
        # Load custom training dataset
        train_data = cfg["data"]["train"]
        train_data_type = train_data.pop("type")
        try:
            train_dataset = eval(train_data_type)(**train_data)
        except Exception as e:
            raise ValueError(f"Failed to load training dataset of type {train_data_type}: {e}")
 
        # Load custom testing dataset
        test_data = cfg["data"]["val"]
        test_data_type = test_data.pop("type")
        try:
            test_dataset = eval(test_data_type)(**test_data)
        except Exception as e:
            raise ValueError(f"Failed to load testing dataset of type {test_data_type}: {e}")
 
    return train_dataset, test_dataset

def setup_model(cfg, logger):
    """
    Setup the model based on the provided configuration.
 
    Parameters:
    cfg (dict): Configuration dictionary containing model parameters.
    logger (logging.Logger): Logger object for logging messages.
 
    Returns:
    torch.nn.Module: The initialized model.
    """
    # Load backbone configuration
    backbone_dict = cfg["model"]["backbone"]
    backbone_name = backbone_dict.pop('type')
 
    # Initialize the model
    try:
        model = eval(backbone_name)(**backbone_dict)
    except Exception as e:
        raise ValueError(f"Failed to create model of type {backbone_name}: {e}")
 
    # Resume model from checkpoint if specified
    if cfg.get("resume_from") is not None:
        resume_file_path = os.path.join(cfg["work_dir"], cfg["resume_from"])
        try:
            state_dict = torch.load(resume_file_path, map_location='cpu')["state_dict"]
            new_state_dict={}
            for key, value in state_dict.items():
                new_key = re.sub(r'^module\.', '', key, count=1)
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=True)
            logger.info(f'Resumed model from {resume_file_path}')
        except Exception as e:
            raise ValueError(f"Failed to load model state from {resume_file_path}: {e}")

 
    return model

def setup_data_loaders(args,cfg, data_name):
    """
    Setup the data loaders for training and validation based on the provided configuration and data name.
 
    Parameters:
    cfg (dict): Configuration dictionary containing data loading parameters.
    data_name (str): The name of the dataset to load.
 
    Returns:
    tuple: A tuple containing the training and validation data loaders.
    """
    # Load datasets
    train_dataset, val_dataset = get_data(data_name, cfg)
 
    # Determine distributed data loading parameters
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
 
    # Create samplers if distributed training is enabled
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    init_fn = partial(worker_init_fn, num_workers=cfg["data"]["workers_per_gpu"], rank=rank,seed=args.seed) \
             if args.seed is not None else None
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["samples_per_gpu"],
        sampler=train_sampler,
        collate_fn=partial(collate, samples_per_gpu=cfg["data"]["samples_per_gpu"]),
        num_workers=cfg["data"]["workers_per_gpu"],
        pin_memory=True,
        shuffle=(train_sampler is None),
        worker_init_fn=init_fn
    )
 
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["samples_per_gpu"],
        sampler=val_sampler,
        collate_fn=partial(collate, samples_per_gpu=cfg["data"]["samples_per_gpu"]),
        num_workers=cfg["data"]["workers_per_gpu"],
        pin_memory=False,
        shuffle=False,
        worker_init_fn=init_fn
    )
 
    return train_loader, val_loader

def setup_optimizer_and_scheduler(cfg, model):
    """
    Setup the optimizer and learning rate scheduler based on the provided configuration.
 
    Parameters:
    cfg (dict): Configuration dictionary containing optimizer and scheduler configurations.
    model (torch.nn.Module): The model to optimize.
 
    Returns:
    tuple: A tuple containing the optimizer and scheduler objects.
    """
    # Load optimizer configuration
    optimizer_cfg = cfg.get("optimizer", {})
    optimizer_cfg_type = optimizer_cfg.pop("type", "Adam")
    try:
        optimizer = eval(optimizer_cfg_type)(model.parameters(), **optimizer_cfg)
    except Exception as e:
        raise ValueError(f"Failed to create optimizer of type {optimizer_cfg_type}: {e}")
 
    # Load scheduler configuration
    scheduler_cfg = cfg.get("lr_config", {})
    scheduler_cfg_type = scheduler_cfg.pop("policy", "StepLR")
    try:
        scheduler = eval(scheduler_cfg_type)(optimizer, **scheduler_cfg)
    except Exception as e:
        raise ValueError(f"Failed to create scheduler of type {scheduler_cfg_type}: {e}")
 
    return optimizer, scheduler

def train_model(args, cfg, scaler,model, data_name, train_loader, val_loader, optimizer, scheduler, logger):
    """
    Train the model using the provided data loaders and configuration.
 
    Parameters:
    args (argparse.Namespace): Parsed command-line arguments.
    cfg (dict): Configuration dictionary.
    model (torch.nn.Module): The model to train.
    data_name (str): The name of the dataset.
    train_loader (torch.utils.data.DataLoader): The training data loader.
    val_loader (torch.utils.data.DataLoader): The validation data loader.
    optimizer (torch.optim.Optimizer): The optimizer to use.
    scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
    logger (logging.Logger): The logger to use for logging.
    """
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
 
    # Initialize distributed data parallel if needed
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        if torch.cuda.is_available():
            model = DataParallel(model.cuda(cfg["gpu_ids"][0]), device_ids=cfg['gpu_ids'])
        else:
            model = model.cpu()
 
    # Load loss function
    loss_cfg = cfg["model"]["head"]["loss"]
    loss_cfg_type = loss_cfg.pop("type")
    loss_func = eval(loss_cfg_type)(**loss_cfg)
 
    # Training configuration
    interval = cfg["log_config"]["interval"]
    topk = cfg['model']['head']['topk'] if cfg['model']['head']['topk'] else (1, 5)
    best_top1_acc = 0
    idex = [0, 1] if "dvsgesture" in data_name or "cifar10dvs" in data_name or "dvsmnist" in data_name  else ['img', 'gt_label']
 
    # Resume training if specified
    if cfg["resume_from"] is not None:
        checkpoint = torch.load(cfg["work_dir"] + '/' + cfg["resume_from"], map_location='cpu')
        epochs = cfg["runner"]["max_epochs"] - (checkpoint["epoch"] + 1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if hasattr(scheduler, 'state_dict'):
            scheduler.load_state_dict(checkpoint["scheduler_stat_dict"])
        iters = checkpoint["iters"]
    else:
        epochs = cfg["runner"]["max_epochs"]
        iters = 0
 
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.
        acc_top1 = 0.
        acc_top5 = 0.
 
        for batch_idx, data in enumerate(train_loader):
            inputs, targets = data[idex[0]].to(device), data[idex[1]].to(device)
 
            # Warmup learning rate if specified
            if cfg.get('warmup', None) is not None:
                if iters < cfg["warmup"]["warmup_iters"]+1:
                    lr = warm_up(cur_iters=iters, warmup_iters=cfg["warmup"]["warmup_iters"], warmup_ratio=cfg["warmup"]["warmup_ratio"], lr=cfg["optimizer"]["lr"])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    iters += 1
 
            optimizer.zero_grad()
 
            # Mixed precision training if specified
            if cfg.get('fp16', None):
                with torch.cuda.amp.autocast():
                    pred = model(inputs)
                    loss = loss_func(pred, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                pred = model(inputs)
                loss = loss_func(pred, targets)
                loss.backward()
                optimizer.step()
 
            train_loss += loss.item()
            acc = accuracy(pred.detach(), targets, topk=topk)
            acc_top1 += acc[0].item()
            if topk == (1, 5):
                acc_top5 += acc[1].item()
 
            # Log training progress
            if (batch_idx + 1) % interval == 0:
                log_str = f'Epoch [{epoch+1}/{epochs}]' \
                        f'[{batch_idx+1}/{len(train_loader)}]\t'
                log_str += f"lr: {optimizer.param_groups[0]['lr']:.3e} "
                log_str += f"top-1: {acc_top1/interval:.4f} "
                if topk == (1, 5):
                    log_str += f"top-5: {acc_top5/interval:.4f} "
                log_str += f"loss: {train_loss/interval:.4f} "
                logger.info(log_str)
                train_loss = 0.
                acc_top1 = 0.
                acc_top5 = 0.
 
        scheduler.step()
        logger.info(f'Saving checkpoint at epoch {epoch+1}')
 
        # Validation loop
        model.eval()
        val_acc_top1 = 0.
        val_acc_top5 = 0.
 
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                inputs, targets = data[idex[0]].to(device), data[idex[1]].to(device)
                result = model(inputs)
                acc = accuracy(result, targets, topk=topk)
                val_acc_top1 += acc[0].item()
                if topk == (1, 5):
                    val_acc_top5 += acc[1].item()
 
        top1_acc = val_acc_top1 / (batch_idx + 1)
        top5_acc = val_acc_top5 / (batch_idx + 1)
        log_str = f'Epoch(val) [{epoch+1}/{epochs}]\t'
        log_str += f"top-1: {top1_acc:.4f} "
        if topk == (1, 5):
            log_str += f"top-5: {top5_acc:.4f} "
        logger.info(log_str)
 
        # Save checkpoint
        savetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        state_dict = model.state_dict()
        if "embedding_fc.weight" in state_dict:
            del state_dict["embedding_fc.weight"]
 
        if dist.is_available() and dist.is_initialized():  
            rank = dist.get_rank()
            torch.distributed.barrier()    
        else:  
            rank = 0 

        if rank == 0:  
            checkpoint = {  
                'state_dict': state_dict,  
                'optimizer_state_dict': optimizer.state_dict(),  
                'scheduler_stat_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,  
                'epoch': epoch,  
                'time': savetime,  
                'iters': iters,  
                'seed': args.seed,  
                'config': args.config  
            }  
            if epoch == 0:  
                best_top1_acc = top1_acc  
            else:  
                if top1_acc > best_top1_acc:  
                    best_top1_acc = top1_acc  
                    torch.save(checkpoint, cfg["work_dir"] + '/best.pth')  
            torch.save(checkpoint, cfg["work_dir"] + '/latest.pth')
def main():
    args = parse_args()
    setup_environment(args)
    cfg,data_name,scaler = load_config(args)
    logger = configure_logger(cfg)
    logger.info(f'Config:\n{pretty_text(cfg)}')
    
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    model = setup_model(cfg, logger)
    train_loader, val_loader = setup_data_loaders(args,cfg, data_name)
    optimizer, scheduler = setup_optimizer_and_scheduler(cfg, model)
    train_model(args,cfg, scaler,model, data_name,train_loader, val_loader, optimizer, scheduler, logger)

if __name__ == '__main__':
    main()