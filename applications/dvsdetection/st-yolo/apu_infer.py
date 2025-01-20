# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import os
import time
from warnings import warn
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Tuple, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from tqdm import tqdm

from functools import partial
from config.modifier import dynamically_modify_train_config
from data.genx_utils.dataset_streaming import build_streaming_dataset
from data.utils.types import DataType, DatasetMode, DatasetSamplingMode, ObjDetOutput
import sys
sys.path.append("../../../")
from lynadapter.lyn_sdk_model import ApuRun_Single
from models.detection.yolox.utils import postprocess
from modules.utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from modules.data.genx_apu import get_dataloader_kwargs

root = os.getcwd()
for _ in range(3):
    root = os.path.dirname(root)  

def parse_args():
    parser = argparse.ArgumentParser(description='specify model to compile and infer')
    parser.add_argument('--model', default='yolov5s', type=str,help='specify a model in yolov5n, yolov5s and yolov5x')
    parser.add_argument('--c', default=0, type=int, help='compile flag: 1 or 0, means if compile or not')
    parser.add_argument('--device', default=0, type=int, help='APU Device IDs to run model on.')
    args = parser.parse_args()
    return args

def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    batch_size_eval = config.batch_size.eval
    num_workers_generic = config.hardware.get('num_workers', None)
    num_workers_eval = config.hardware.num_workers.get('eval', num_workers_generic)
    dataset_config = config.dataset
    eval_sampling_mode = 'stream'

    build_eval_dataset = partial(build_streaming_dataset,
                                 batch_size=batch_size_eval,
                                 num_workers=num_workers_eval)

    test_dataset = build_eval_dataset(dataset_mode=DatasetMode.TESTING, dataset_config=dataset_config)


    test_dataloader = DataLoader(**get_dataloader_kwargs(
        dataset=test_dataset, sampling_mode=eval_sampling_mode, dataset_mode=DatasetMode.TESTING,
        dataset_config=dataset_config, batch_size=batch_size_eval,
        num_workers=num_workers_eval))

    ### initialize
    dataset_name = config.dataset.name
    mode = Mode.TEST
    mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
    mode_2_batch_size: Dict[Mode, Optional[int]] = {}
    mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
    mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}
    dataset_eval_sampling = config.dataset.eval.sampling
    assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
    mode_2_psee_evaluator[mode] = PropheseeEvaluator(
        dataset=dataset_name, downsample_by_2=config.dataset.downsample_by_factor_2)
    mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
    mode_2_hw[mode] = None
    mode_2_batch_size[mode] = None
    ### initialize finish

    path_backbone = os.path.join(root+config.path_backbone, "Net_0")
    arun = ApuRun_Single(device, path_backbone)

    path_head = os.path.join(root+config.path_head, "Net_0")
    arun2 = ApuRun_Single(device, path_head)

    num_iterations = len(test_dataset)
    n = 0
    t_total = 0
    for val_batch in tqdm(test_dataloader):
        n += 1
        t1 = time.time()
        data = val_batch['data']
        worker_id = val_batch['worker_id']
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if mode_2_batch_size[mode] is None:
            mode_2_batch_size[mode] = batch_size
        else:
            assert mode_2_batch_size[mode] == batch_size

        backbone_feature_selector = BackboneFeatureSelector()
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        in_res_hw = tuple(config.model.backbone.in_res_hw)
        input_padder = InputPadderFromShape(desired_hw=in_res_hw)
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=torch.float32)
            ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
            if mode_2_hw[mode] is None:
                mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert mode_2_hw[mode] == ev_tensors.shape[-2:]

            data_img = np.array(ev_tensors).astype(np.float32)

            if is_first_sample:
                output = arun.run(data_img, 0)  # reset
            else:
                output = arun.run(data_img, 1)

            backbone_features = {}
            backbone_features[1] = torch.from_numpy(output[0])
            backbone_features[2] = torch.from_numpy(output[1])
            backbone_features[3] = torch.from_numpy(output[2])
            backbone_features[4] = torch.from_numpy(output[3])

            if collect_predictions:
                current_labels, valid_batch_indices = sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
                # Store backbone features that correspond to the available labels.
                if len(current_labels) > 0:
                    backbone_feature_selector.add_backbone_features(backbone_features=backbone_features,
                                                                    selected_indices=valid_batch_indices)

                    obj_labels.extend(current_labels)
                    ev_repr_selector.add_event_representations(event_representations=ev_tensors,
                                                               selected_indices=valid_batch_indices)
        if len(obj_labels) == 0:
            outputs = {ObjDetOutput.SKIP_VIZ: True}
        else:
            selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
            n_batch = selected_backbone_features[1].size(0)
            for i in range(n_batch):
                f2 = np.array(selected_backbone_features[2][i:i + 1, ...])
                f3 = np.array(selected_backbone_features[3][i:i + 1, ...])
                f4 = np.array(selected_backbone_features[4][i:i + 1, ...])
                data_in = np.concatenate((f4.reshape(-1), f3.reshape(-1), f2.reshape(-1)))
                
                output = arun2.run(data_in, i)
                output = torch.from_numpy(output[0])
                if i == 0:
                    predictions = output
                else:
                    predictions = torch.cat((predictions, output), dim=0)

            pred_processed = postprocess(prediction=predictions,
                                         num_classes=config.model.head.num_classes,
                                         conf_thre=config.model.postprocess.confidence_threshold,
                                         nms_thre=config.model.postprocess.nms_threshold)

            loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

            mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

        t2 = time.time()
        fps = 11/(t2-t1)
        t_total += (t2-t1)

        if n >= num_iterations:   
            print(f'Average frame rate is {11 * n / t_total}fps')
            break
    arun.apu_unload()
    print('test_dataloader traverse finished')
    psee_evaluator = mode_2_psee_evaluator[mode]
    batch_size = mode_2_batch_size[mode]
    hw_tuple = mode_2_hw[mode]
    if psee_evaluator is None:
        warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
        return
    assert batch_size is not None
    assert hw_tuple is not None
    if psee_evaluator.has_data():
        metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                 img_width=hw_tuple[1])
        assert metrics is not None

        prefix = f'{mode_2_string[mode]}/'
        log_dict = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                value = torch.tensor(v)
            elif isinstance(v, np.ndarray):
                value = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                value = v
            else:
                raise NotImplementedError
            assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
            print(f'{prefix}{k} :  {value}')
            # log_dict[f'{prefix}{k}'] = value.to(torch.device)
        psee_evaluator.reset_buffer()
        os._exit(0)
    else:
        warn(f'psee_evaluator has not data in {mode=}', UserWarning, stacklevel=2)


if __name__ == '__main__':
    args = parse_args()
    if args.c:
        from compile_backbone import compile_backbone
        from compile_head import compile_head
        compile_backbone(yolo_dict=args.model)
        compile_head(yolo_dict=args.model)
    global device
    device = args.device
    config_name = 'val_' + args.model
    initialize(config_path="config", job_name="main")
    cfg = compose(config_name=config_name)
    main(cfg)
