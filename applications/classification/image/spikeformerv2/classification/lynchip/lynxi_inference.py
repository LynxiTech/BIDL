# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.

import sys; 
sys.path.append('../')
sys.path.append('../../../../../../')
import argparse
import os
import time
from pathlib import Path
import torch
import util.misc as misc
from util.datasets import build_dataset
import lynxi_models as models
from pass_function import optSpikeTransformer
from evaluate import evaluate
import re
from utils import globals

def get_args_parser():
    # important params
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)  # 20/30(T=4)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--data_path", default="/raid/ligq/imagenet1-k/", type=str, help="dataset path"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="spikformer_8_384_CAFormer",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--model_mode",
        default="ms",
        type=str,
        help="Mode of model to train",
    )

    parser.add_argument(
        "--compile",
        type=bool,
        default=False,
    )

    # parser.add_argument(
    #     "--isCount",
    #     type=bool,
    #     default=False,
    # )

    parser.add_argument("--input_size",
        default=224,
        type=int,
        help="size of input images",
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default='',
        help="lynxi model path",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=6e-4,
        metavar="LR",  # 1e-5,2e-5(T=4)
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    parser.add_argument("--time_steps", default=4, type=int)

    # Dataset parameters

    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="/raid/ligq/htx/spikemae/output_dir",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cpu:0", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distillation parameters
    parser.add_argument(
        "--kd",
        action="store_true",
        default=True,
        help="kd or not",
    )
    parser.add_argument(
        "--teacher_model",
        default="caformer_b36_in21ft1k",
        type=str,
        metavar="MODEL",
        help='Name of teacher model to train (default: "caformer_b36_in21ft1k"',
    )
    parser.add_argument(
        "--distillation_type",
        default="none",
        choices=["none", "soft", "hard"],
        type=str,
        help="",
    )
    parser.add_argument("--distillation_alpha", default=0.5, type=float, help="")
    parser.add_argument("--distillation_tau", default=1.0, type=float, help="")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser

def main(args):

    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    dataset_val = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    if args.model_mode == "ms":
        model = models.__dict__[args.model](kd=args.kd)
    elif args.model_mode == "sew":
        model = models_sew.__dict__[args.model]()
    model.T = args.time_steps
    ck_path = globals.get_value('ck_path')
    checkpoint = torch.load(ck_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    if True:
        isCompile = globals.get_value('isCompile')
        if isCompile:
            import lyngor as lyn
            # lyn.debug()
            generated_path = globals.get_value('generated_path')
            print('------------ Compile Start ------------')
            lyn_model = lyn.DLModel()
            lyn_model.load(model, model_type="pytorch", inputs_dict={"input0":(1,3,224,224)},
                        in_type="float16",
                        out_type="float16")
            offline_builder = lyn.Builder(target="apu")
            offline_builder.build(lyn_model.graph, lyn_model.param, out_path=generated_path, custom_opt=[optSpikeTransformer()])
            print('------------ Compile End ------------')
        start_t = time.time()
        test_stats = evaluate(data_loader_val, model, device)
        end_t = time.time()
        total_t = end_t - start_t
        img_acc = len(data_loader_val)
        fps = img_acc * model.T / total_t
        print('===> fps = ', fps)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

root_path = '../../../../../../'
img_path = root_path + "data/spikeformerv2/few_ILSVRC2012_img/"
ck_path = root_path + 'weight_files/classification/spikeformerv2/'
generated_path = root_path + "model_files/classification/spikeformerv2/"

globals._init()
globals.set_value('isHook', False)
globals.set_value('isModified', True)
globals.set_value('img_path', img_path)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    globals.set_value('isCompile', args.compile)
    globals.set_value('ck_path', ck_path+'55M_kd_T{}.pth'.format(args.time_steps))
    globals.set_value('generated_path', generated_path+'55M_kd_T{}'.format(args.time_steps))
    args.data_path = img_path

    if re.search(r'apu:(\d+)', args.device):
        apu_id = int(re.search(r'apu:(\d+)', args.device).group(1))
        args.device = 'cpu'
        globals.set_value('apu_id', apu_id)
        globals.set_value('isApu', True)

    main(args)
