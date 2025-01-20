# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import os
import time
from collections import OrderedDict

from models.detection.yolox_extension.models.yolo_detector import YoloV5Detector
from utils.evaluation.prophesee.visualize.vis_utils import draw_bboxes, LABELMAP_GEN1, draw_bboxes_bbv

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
from models.detection.yolox.utils import postprocess
from modules.utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee, to_prophesee_show
from utils.padding import InputPadderFromShape
from modules.data.genx import get_dataloader_kwargs
from einops import rearrange, reduce


def ev_repr_to_img(x: np.ndarray):
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img


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
    mode_2_rnn_states: Dict[Mode, RNNStates] = {
        Mode.TRAIN: RNNStates(),
        Mode.VAL: RNNStates(),
        Mode.TEST: RNNStates(),
    }
    ### initialize finish

    model = YoloV5Detector(config.model)
    checkpoint_file = config.checkpoint
    params = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    temp_dict = dict(list(params['state_dict'].items()))
    state_dict = OrderedDict((k[4:], v) for k, v in temp_dict.items())
    model.load_state_dict(state_dict)
    model.training = False
    model.yolox_head.training = False

    n = 0
    for val_batch in tqdm(test_dataloader):
        data = val_batch['data']
        worker_id = val_batch['worker_id']
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if mode_2_batch_size[mode] is None:
            mode_2_batch_size[mode] = batch_size
        else:
            assert mode_2_batch_size[mode] == batch_size

        prev_states = mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        in_res_hw = tuple(config.model.backbone.in_res_hw)
        input_padder = InputPadderFromShape(desired_hw=in_res_hw)
        for tidx in range(sequence_len):
            n += 1
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=torch.float32)
            ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
            if mode_2_hw[mode] is None:
                mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = model.forward_backbone(x=ev_tensors, previous_states=prev_states)
            prev_states = states

            predictions, losses = model.forward_detect(backbone_features=backbone_features)

            pred_processed = postprocess(prediction=predictions,
                                         num_classes=config.model.head.num_classes,
                                         conf_thre=config.model.postprocess.confidence_threshold,
                                         nms_thre=config.model.postprocess.nms_threshold)

            yolox_preds_proph = to_prophesee_show(pred_processed)
            ev_img = ev_repr_to_img(ev_tensors[0].cpu().numpy())
            prediction_img = ev_img.copy()
            draw_bboxes_bbv(prediction_img, yolox_preds_proph[0], save_id=n,
                            labelmap=LABELMAP_GEN1, is_pred=True)
        mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

    print('test_dataloader traverse finished')


if __name__ == '__main__':
    initialize(config_path="config", job_name="main")
    cfg = compose(config_name="gpu_show")
    main(cfg)
