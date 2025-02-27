# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

import copy
from abc import ABCMeta, abstractmethod
import numpy as np
from torch.utils.data import Dataset
from collections.abc import Sequence
from losses.accuracy import *
from .pipelines.bidl_formating import *
from .pipelines.bidl_loading import *
from .pipelines.bidl_transforms import *

class Compose:
    """Compose a data pipeline with a sequence of transforms.
 
    Args:
        transforms (Sequence[dict | callable]): A sequence of either
            configuration dictionaries for transforms or transform objects.
    """
 
    def __init__(self, transforms):
        """Initialize the Compose object."""
        assert isinstance(transforms, Sequence), 'transforms must be a sequence'
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform_type = transform.pop("type")
                try:
                    transform_obj = eval(transform_type)(**transform)
                except Exception as e:
                    raise ValueError(f'Failed to create transform of type {transform_type}: {e}')
                self.transforms.append(transform_obj)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
 
    def __call__(self, data):
        """Apply the transforms to the data.
 
        Args:
            data (dict): The data to be transformed.
 
        Returns:
            dict: The transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
 
    def __repr__(self):
        """Return a string representation of the Compose object."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


class BaseDataset(metaclass=ABCMeta):
    """Base class for datasets.
 
    Args:
        data_prefix (str): Prefix of the data path.
        pipeline (list): List of dictionaries, where each element represents
            an operation defined in `mmcls.datasets.pipelines`.
        classes (Sequence[str] | str | None): Class names. If `None`, use
            default `CLASSES` defined by the dataset. If a string, it is
            treated as a file name containing class names, one per line. If
            a tuple or list, it overrides the default `CLASSES`.
        ann_file (str | None): Annotation file path. When `None`, the
            subclass is expected to read annotations based on `data_prefix`.
        test_mode (bool): Whether the dataset is in test mode.
    """
 
    CLASSES = None
 
    def __init__(self, data_prefix, pipeline, classes=None, ann_file=None, test_mode=False):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()
 
    @abstractmethod
    def load_annotations(self):
        """Load annotations for the dataset."""
        pass
 
    @property
    def class_to_idx(self):
        """Mapping from class name to class index.
 
        Returns:
            dict: Mapping from class name to class index.
        """
        return {cls: idx for idx, cls in enumerate(self.CLASSES)}
 
    def get_gt_labels(self):
        """Get ground-truth labels for all images.
 
        Returns:
            np.ndarray: Ground-truth labels.
        """
        return np.array([data['gt_label'] for data in self.data_infos])
 
    def get_cat_ids(self, idx):
        """Get category ID for a given index.
 
        Args:
            idx (int): Index of the data.
 
        Returns:
            int: Category ID.
        """
        return self.data_infos[idx]['gt_label'].astype(np.int)
 
    def prepare_data(self, idx):
        """Prepare data for a given index.
 
        Args:
            idx (int): Index of the data.
 
        Returns:
            dict: Prepared data.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)
 
    def __len__(self):
        """Get the number of samples in the dataset.
 
        Returns:
            int: Number of samples.
        """
        return len(self.data_infos)
 
    def __getitem__(self, idx):
        """Get data for a given index.
 
        Args:
            idx (int): Index of the data.
 
        Returns:
            dict: Data for the given index.
        """
        return self.prepare_data(idx)
 
    @classmethod
    def get_classes(cls, classes=None):
        """Get class names for the dataset.
 
        Args:
            classes (Sequence[str] | str | None): Class names. See class
                docstring for details.
 
        Returns:
            tuple[str] | list[str]: Class names.
        """
        if classes is None:
            return cls.CLASSES
        elif isinstance(classes, (tuple, list)):
            return classes
        elif isinstance(classes, str):
            # Load class names from a file
            with open(classes, 'r') as f:
                return [line.strip() for line in f]
        else:
            raise ValueError(f'Unsupported type {type(classes)} for classes.')
 
    def evaluate(self, results, metric='accuracy', metric_options=None, logger=None):
        """Evaluate the dataset.
 
        Args:
            results (list): Testing results.
            metric (str | list[str]): Metrics to evaluate. Default is
                `'accuracy'`.
            metric_options (dict, optional): Options for metrics. Allowed
                keys are `'topk'`, `'thrs'`, and `'average_mode'`. Defaults
                to `None`.
            logger (logging.Logger | str, optional): Logger for printing
                evaluation information. Defaults to `None`.
 
        Returns:
            dict: Evaluation results.
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        
        allowed_metrics = {
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        }
        eval_results = {}
        results = np.vstack(results)
 
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        gt_labels = gt_labels[:num_imgs]
        assert len(gt_labels) == num_imgs, 'Testing results should have ' \
            'the same length as ground-truth labels.'
 
        invalid_metrics = set(metrics) - allowed_metrics
        if invalid_metrics:
            raise ValueError(f'Unsupported metrics: {invalid_metrics}')
 
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
                    f'accuracy_top-{k}': a.item()
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc.item()}
 
            if thrs is not None:
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(eval_results_)
 
        return eval_results