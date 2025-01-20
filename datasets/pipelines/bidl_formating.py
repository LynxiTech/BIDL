# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image



def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not torch.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
    
class ImageToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'



class Transpose(object):

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'



class ToPIL(object):

    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = Image.fromarray(results['img'])
        return results



class ToNumpy(object):

    def __init__(self):
        pass

    def __call__(self, results):
        results['img'] = np.array(results['img'], dtype=np.float32)
        return results


class WrapFieldsToLists(object):
    """Wrap fields of the data dictionary into lists for evaluation.

    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.

    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapIntoLists')
        >>> ]
    """

    def __call__(self, results):
        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ToNdarr:

    def __init__(self, keys, dtype):
        self.keys = keys
        self.dtype = dtype

    def __call__(self, results):
        for key in self.keys:
            _dat = results[key]
            _pack = results.get('pack')
            if _pack is not None:
                assert len(_dat.shape) == 1
                _dat = np.unpackbits(_dat).reshape(_pack)
                results.pop('pack')
            results[key] = _dat.astype(self.dtype)
        return results


class ToTensorType:

    def __init__(self, keys, dtype):
        self.keys = keys
        self.dtype = dtype

    def __call__(self, results):
        results = torch.from_numpy(results.astype(self.dtype))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
    
class ToTensorTypeDict:
    """
    Only support dimension order (t,c,h,w)!
    For Dict data
    """
    def __init__(self, keys, dtype):
        self.keys = keys
        self.dtype = dtype

    def __call__(self, results):
        for key in self.keys:
            results[key] = torch.from_numpy(results[key].astype(self.dtype))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
    
class ToOneHot:
    def __init__(self, keys,param):
        self.keys = keys
        self.param = param


    def __call__(self, results):
        for key in self.keys:
            if len(results[key].shape)==0:
                one_hot = torch.zeros(self.param)
                one_hot.scatter_(dim=0, index=results[key].unsqueeze(dim=0),src=torch.ones(self.param))
            else:                
                one_hot = torch.zeros(results[key].shape[0],self.param).long()
                one_hot.scatter_(dim=1, index=results[key].unsqueeze(dim=1),src=torch.ones(results[key].shape[0],self.param).long())

            results[key] = one_hot

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
