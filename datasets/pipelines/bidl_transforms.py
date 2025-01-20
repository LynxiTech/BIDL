# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

import math
import random
import copy
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None
import cv2
import torch
#from torchvision.transforms import functional_pil as F_pil
from enum import Enum
from torchvision.transforms import functional as F
from .auto_augment import Brightness, Contrast, ColorTransform, random_negative
from datasets.base_dataset import *
import torch.nn.functional as FUN
from torch.distributions import Normal
if Image is not None:
    pillow_interp_codes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'box': Image.BOX,
        'lanczos': Image.LANCZOS,
        'hamming': Image.HAMMING
    }

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}
def bbox_clip(bboxes, img_shape):
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple[int]): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
    cmin[0::2] = img_shape[1] - 1
    cmin[1::2] = img_shape[0] - 1
    clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
    return clipped_bboxes


def bbox_scaling(bboxes, scale, clip_shape=None):
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple[int], optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes

def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))
    
def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale
    
def imcrop(img, bboxes, scale=1.0, pad_fill=None):
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no padding.
        pad_fill (Number | list[Number]): Value to be filled for padding.
            Default: None, which means no padding.

    Returns:
        list[ndarray] | ndarray: The cropped image patches.
    """
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            if chn == 1:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
            else:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
            patch = np.array(
                pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start + w,
                  ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches
    
class RandomCrop(object):
    """Crop the given Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Defaults to "constant". Should
            be one of the following:

            - constant: Pads with a constant value, this value is specified \
                with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: Pads with reflection of image without repeating the \
                last value on the edge. For example, padding [1, 2, 3, 4] \
                with 2 elements on both sides in reflect mode will result \
                in [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: Pads with reflection of image repeating the last \
                value on the edge. For example, padding [1, 2, 3, 4] with \
                2 elements on both sides in symmetric mode will result in \
                [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 pad_val=0,
                 padding_mode='constant'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        ymin = random.randint(0, height - target_height)
        xmin = random.randint(0, width - target_width)
        return ymin, xmin, target_height, target_width
    '''
    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.padding is not None:
                img = mmcv.impad(
                    img, padding=self.padding, pad_val=self.pad_val)

            # pad the height if needed
            if self.pad_if_needed and img.shape[0] < self.size[0]:
                img = mmcv.impad(
                    img,
                    padding=(0, self.size[0] - img.shape[0], 0,
                             self.size[0] - img.shape[0]),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.shape[1] < self.size[1]:
                img = mmcv.impad(
                    img,
                    padding=(self.size[1] - img.shape[1], 0,
                             self.size[1] - img.shape[1], 0),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            ymin, xmin, height, width = self.get_params(img, self.size)
            results[key] = mmcv.imcrop(
                img,
                np.array([
                    xmin,
                    ymin,
                    xmin + width - 1,
                    ymin + height - 1,
                ]))
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(size={self.size}, padding={self.padding})')
    '''

class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        size (sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maxinum number of attempts before falling back to
            Central Crop. Defaults to 10.
        efficientnet_style (bool): Whether to use efficientnet style Random
            ResizedCrop. Defaults to False.
        min_covered (Number): Minimum ratio of the cropped area to the original
             area. Only valid if efficientnet_style is true. Defaults to 0.1.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet_style is true.
            Defaults to 32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 max_attempts=10,
                 efficientnet_style=False,
                 min_covered=0.1,
                 crop_padding=32,
                 interpolation='bilinear',
                 backend='cv2'):
        if efficientnet_style:
            assert isinstance(size, int)
            self.size = (size, size)
            assert crop_padding >= 0
        else:
            if isinstance(size, (tuple, list)):
                self.size = size
            else:
                self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received scale {scale} and rato {ratio}.')
        assert min_covered >= 0, 'min_covered should be no less than 0.'
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.efficientnet_style = efficientnet_style
        self.min_covered = min_covered
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio, max_attempts=10):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maxinum number of attempts before falling back
                to central crop. Defaults to 10.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    # https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py # noqa
    @staticmethod
    def get_params_efficientnet_style(img,
                                      size,
                                      scale,
                                      ratio,
                                      max_attempts=10,
                                      min_covered=0.1,
                                      crop_padding=32):
        """Get parameters for ``crop`` for a random sized crop in efficientnet
        style.

        Args:
            img (ndarray): Image to be cropped.
            size (sequence): Desired output size of the crop.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maxinum number of attempts before falling back
                to central crop. Defaults to 10.
            min_covered (Number): Minimum ratio of the cropped area to the
                original area. Only valid if efficientnet_style is true.
                Defaults to 0.1.
            crop_padding (int): The crop padding parameter in efficientnet
                style center crop. Defaults to 32.

        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height, width = img.shape[:2]
        area = height * width
        min_target_area = scale[0] * area
        max_target_area = scale[1] * area

        for _ in range(max_attempts):
            aspect_ratio = random.uniform(*ratio)
            min_target_height = int(
                round(math.sqrt(min_target_area / aspect_ratio)))
            max_target_height = int(
                round(math.sqrt(max_target_area / aspect_ratio)))

            if max_target_height * aspect_ratio > width:
                max_target_height = int((width + 0.5 - 1e-7) / aspect_ratio)
                if max_target_height * aspect_ratio > width:
                    max_target_height -= 1

            max_target_height = min(max_target_height, height)
            min_target_height = min(max_target_height, min_target_height)

            # slightly differs from tf inplementation
            target_height = int(
                round(random.uniform(min_target_height, max_target_height)))
            target_width = int(round(target_height * aspect_ratio))
            target_area = target_height * target_width

            # slight differs from tf. In tf, if target_area > max_target_area,
            # area will be recalculated
            if (target_area < min_target_area or target_area > max_target_area
                    or target_width > width or target_height > height
                    or target_area < min_covered * area):
                continue

            ymin = random.randint(0, height - target_height)
            xmin = random.randint(0, width - target_width)
            ymax = ymin + target_height - 1
            xmax = xmin + target_width - 1

            return ymin, xmin, ymax, xmax

        # Fallback to central crop
        img_short = min(height, width)
        crop_size = size[0] / (size[0] + crop_padding) * img_short

        ymin = max(0, int(round((height - crop_size) / 2.)))
        xmin = max(0, int(round((width - crop_size) / 2.)))
        ymax = min(height, ymin + crop_size) - 1
        xmax = min(width, xmin + crop_size) - 1

        return ymin, xmin, ymax, xmax

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.efficientnet_style:
                get_params_func = self.get_params_efficientnet_style
                get_params_args = dict(
                    img=img,
                    size=self.size,
                    scale=self.scale,
                    ratio=self.ratio,
                    max_attempts=self.max_attempts,
                    min_covered=self.min_covered,
                    crop_padding=self.crop_padding)
            else:
                get_params_func = self.get_params
                get_params_args = dict(
                    img=img,
                    scale=self.scale,
                    ratio=self.ratio,
                    max_attempts=self.max_attempts)
            ymin, xmin, ymax, xmax = get_params_func(**get_params_args)
            img = imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
            results[key] = imresize(
                img,
                tuple(self.size[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(size={self.size}'
        repr_str += f', scale={tuple(round(s, 4) for s in self.scale)}'
        repr_str += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', min_covered={self.min_covered}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str



class RandomFlip(object):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility and flip direction.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_prob=0.5, direction='horizontal'):
        assert 0 <= flip_prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        results['flip'] = flip
        results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = imflip(
                    results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


class Resize(object):
    """Resize images.

    Args:
        size (int | tuple): Images scales for resizing (h, w).
            When size is int, the default behavior is to resize an image
            to (size, size). When size is tuple and the second value is -1,
            the short edge of an image is resized to its first value.
            For example, when size is 224, the image is resized to 224x224.
            When size is (224, -1), the short side is resized to 224 and the
            other side is computed based on the short side, maintaining the
            aspect ratio.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            More details can be found in `mmcv.image.geometric`.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(self, size, interpolation='bilinear', backend='cv2'):
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
        self.resize_w_short_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_short_side = True
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            
            if not isinstance(img, torch.Tensor):
                
                img = F.to_tensor(img).squeeze(0).numpy() 
                          
                # pil_interpolation = pil_modes_mapping[InterpolationMode.BILINEAR]
                # img =  F_pil.resize(img, size=self.size, interpolation=pil_interpolation,max_size=None)
               
                # results[key] =  F.to_tensor(img)
                # results['img_shape'] = F.to_tensor(img).shape
                # return
            ignore_resize = False
            if self.resize_w_short_side:
                h, w = img.shape[:2]
                short_side = self.size[0]
                if (w <= h and w == short_side) or (h <= w
                                                    and h == short_side):
                    ignore_resize = True
                else:
                    if w < h:
                        width = short_side
                        height = int(short_side * h / w)
                    else:
                        height = short_side
                        width = int(short_side * w / h)
            else:
                height, width = self.size
            if not ignore_resize:
                img = imresize(
                    img,
                    size=(width, height),
                    interpolation=self.interpolation,
                    return_scale=False,
                    backend=self.backend)                
                results[key] = torch.from_numpy(img)
                results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str



class CenterCrop(object):
    r"""Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping with the format
            of (h, w).
        efficientnet_style (bool): Whether to use efficientnet style center
            crop. Defaults to False.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet style is True. Defaults to
            32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Only valid if
            ``efficientnet_style`` is True. Defaults to 'bilinear'.
        backend (str): The image resize backend type, accepted values are
            `cv2` and `pillow`. Only valid if efficientnet style is True.
            Defaults to `cv2`.


    Notes:
        - If the image is smaller than the crop size, return the original
          image.
        - If efficientnet_style is set to False, the pipeline would be a simple
          center crop using the crop_size.
        - If efficientnet_style is set to True, the pipeline will be to first
          to perform the center crop with the ``crop_size_`` as:

        .. math::
            \text{crop\_size\_} = \frac{\text{crop\_size}}{\text{crop\_size} +
            \text{crop\_padding}} \times \text{short\_edge}

        And then the pipeline resizes the img to the input crop size.
    """

    def __init__(self,
                 crop_size,
                 efficientnet_style=False,
                 crop_padding=32,
                 interpolation='bilinear',
                 backend='cv2'):
        if efficientnet_style:
            assert isinstance(crop_size, int)
            assert crop_padding >= 0
            assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                     'lanczos')
            if backend not in ['cv2', 'pillow']:
                raise ValueError(
                    f'backend: {backend} is not supported for '
                    'resize. Supported backends are "cv2", "pillow"')
        else:
            assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                                  and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.efficientnet_style = efficientnet_style
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            # img.shape has length 2 for grayscale, length 3 for color
            img_height, img_width = img.shape[:2]

            # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L118 # noqa
            if self.efficientnet_style:
                img_short = min(img_height, img_width)
                crop_height = crop_height / (crop_height +
                                             self.crop_padding) * img_short
                crop_width = crop_width / (crop_width +
                                           self.crop_padding) * img_short

            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = imcrop(img, bboxes=np.array([x1, y1, x2, y2]))

            if self.efficientnet_style:
                img = imresize(
                    img,
                    tuple(self.crop_size[::-1]),
                    interpolation=self.interpolation,
                    backend=self.backend)
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(crop_size={self.crop_size}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str





class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness.
            brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast.
            contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation.
            saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation].
    """

    def __init__(self, brightness, contrast, saturation):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, results):
        brightness_factor = random.uniform(0, self.brightness)
        contrast_factor = random.uniform(0, self.contrast)
        saturation_factor = random.uniform(0, self.saturation)
        color_jitter_transforms = [
            dict(
                type='Brightness',
                magnitude=brightness_factor,
                prob=1.,
                random_negative_prob=0.5),
            dict(
                type='Contrast',
                magnitude=contrast_factor,
                prob=1.,
                random_negative_prob=0.5),
            dict(
                type='ColorTransform',
                magnitude=saturation_factor,
                prob=1.,
                random_negative_prob=0.5)
        ]
        random.shuffle(color_jitter_transforms)
        transform = Compose(color_jitter_transforms)
        return transform(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation})'
        return repr_str
    

    
    

def _handle_padding_params(self, padding, nested):
    if isinstance(padding, int):
        if nested:
            self.padding = ((padding,) * 2,) * 2
        else:
            self.padding = (padding,) * 4
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        if nested:
            self.padding = ((padding[0],) * 2, (padding[1],) * 2)
        else:
            self.padding = (padding[0],) * 2 + (padding[1],) * 2
    elif isinstance(padding, (list, tuple)) and len(padding) == 4:
        if nested:
            self.padding = tuple(padding[:2]) + tuple(padding[2:])
        else:
            self.padding = padding
    elif padding is None:
        self.padding = None
    else:
        raise NotImplemented



class CropPadByRatioVideo:
    """
    Only support dimension order (t,c,h,w)!
    """

    def __init__(self, croppad, allowed_ratio, min_padd=0., max_padd=0., pad_mode='wrap'):
        """
        Args:
            croppad: of length 4, in percentage
            allowed_ratio: width-height-ratio. Crop/pad the image if its ratio is out of this range.
        """
        assert croppad is None or len(croppad) == 4
        assert len(allowed_ratio) == 2
        self.croppad = croppad
        self.allowed_ratio = allowed_ratio
        assert min_padd <= max_padd
        self.min_padd = min_padd
        self.max_padd = max_padd
        if isinstance(pad_mode, (tuple, list)):
            assert all([_ in self.PAD_MODES for _ in pad_mode])
        self.pad_mode = pad_mode

    PAD_MODES = (
        'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'symmetric', 'wrap',
        'empty')

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            frames = results[key]

            h, w = frames.shape[-2:]
            ratio = w / h
            if not (self.allowed_ratio[0] <= ratio <= self.allowed_ratio[1]):
                if self.croppad is None:
                    if ratio < self.allowed_ratio[0]:
                        dw = round((h * self.allowed_ratio[0] - w) / 2)
                        padding = [dw, 0, dw, 0]
                    else:
                        assert ratio > self.allowed_ratio[1]
                        dh = round((w / self.allowed_ratio[1] - h) / 2)
                        padding = [0, dh, 0, dh]
                else:
                    side = min(h, w)
                    padding_ = [_ * side for _ in self.croppad]
                    m_p = [[round(-min(_, 0)), round(max(_, 0))] for _ in padding_]
                    m, p = list(zip(*m_p))
                    y1, x1, y2, x2 = m[1], m[0], h - m[3], w - m[2]
                    frames = frames[:, :, y1:y2, x1:x2]

                    h1, w1 = frames.shape[-2:]
                    h2, w2 = y2 - y1, x2 - x1
                    ratio2 = w2 / h2
                    if ratio2 < self.allowed_ratio[0]:
                        dw2 = round(h2 * self.allowed_ratio[0] - w1)
                        _pw_ = dw2 / sum(p[0::2])
                        padding = [round(_pw_ * p[0]), p[1], round(_pw_ * p[2]), p[3]]
                    elif ratio2 > self.allowed_ratio[1]:
                        dh2 = round(w2 / self.allowed_ratio[1] - h1)
                        _ph_ = dh2 / sum(p[1::2])
                        padding = [p[0], round(_ph_ * p[1]), p[2], round(_ph_ * p[3])]
                    else:
                        padding = list(p)
            else:
                padding = [round(self.min_padd * min(h, w))] * 4

            padding = [round(min(_, self.max_padd * min(w, h))) for _ in padding]

            pad_size = ((0, 0),) * 2 + ((padding[1], padding[3]), (padding[0], padding[2]))
            if isinstance(self.pad_mode, (list, tuple)):
                modes = self.PAD_MODES + tuple(self.pad_mode)
                pad_mode = np.random.choice(modes)
            else:
                pad_mode = self.pad_mode
            frames3 = np.pad(frames, pad_size, mode=pad_mode)
            results[key] = frames3

        return results

class RandomCropVideoDict:
    """
    Only support dimension order (t,c,h,w)!
    For Dict data
    """

    def __init__(self, *args, **kwargs):
        RandomCrop.__init__(self, *args, **kwargs)
        _handle_padding_params(self, self.padding, False)

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            frames = results[key]  # (t,c,h,w)
            if self.pad_if_needed is True:
                hi, wi = frames.shape[-2:]
                h1, w1 = hi + self.padding[1] + self.padding[3], wi + self.padding[0] + self.padding[2]
                dh, dw = self.size[0] - h1, self.size[1] - w1
                padding = ((self.padding[1] + dh, self.padding[3] + dh), (self.padding[0] + dw, self.padding[2] + dw))
            else:
                padding = ((self.padding[1], self.padding[3]), (self.padding[0], self.padding[2]))

            if padding is not None:
                pad_size = ((0, 0),) * 2 + padding
                frames = np.pad(frames, pad_size, self.padding_mode, constant_values=self.pad_val)

            img = frames[0].transpose([1, 2, 0])
            ymin, xmin, height, width = RandomCrop.get_params(img, self.size)
            frames2 = frames[..., ymin:ymin + height, xmin:xmin + width]

            results[key] = frames2

        return results
    
class RandomCropVideo:
    """
    Only support dimension order (t,c,h,w)!
    """

    def __init__(self, *args, **kwargs):
        RandomCrop.__init__(self, *args, **kwargs)
        _handle_padding_params(self, self.padding, False)

    def __call__(self, results):
        
        frames = results  # (t,c,h,w)
        if self.pad_if_needed is True:
                hi, wi = frames.shape[-2:]
                h1, w1 = hi + self.padding[1] + self.padding[3], wi + self.padding[0] + self.padding[2]
                dh, dw = self.size[0] - h1, self.size[1] - w1
                padding = ((self.padding[1] + dh, self.padding[3] + dh), (self.padding[0] + dw, self.padding[2] + dw))
        else:
                padding = ((self.padding[1], self.padding[3]), (self.padding[0], self.padding[2]))

        if padding is not None:
                pad_size = ((0, 0),) * 2 + padding
                frames = np.pad(frames, pad_size, self.padding_mode, constant_values=self.pad_val)

        img = frames[0].transpose([1, 2, 0])
        ymin, xmin, height, width = RandomCrop.get_params(img, self.size)
        frames2 = frames[..., ymin:ymin + height, xmin:xmin + width]

        results = frames2

        return results


class RandomFlipVideoDict:
    """
    Only support dimension order (t,c,h,w)!
    For Dict data
    """

    __init__ = RandomFlip.__init__

    def __call__(self, results):
        flip = True if np.random.rand() < self.flip_prob else False
        results['flip'] = flip
        results['flip_direction'] = self.direction

        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                frames = results[key]  # (t,c,h,w)

                direction = results['flip_direction']
                if direction == 'horizontal':
                    frames2 = np.flip(frames, axis=2)
                elif direction == 'vertical':
                    frames2 = np.flip(frames, axis=3)
                else:
                    raise NotImplemented

                results[key] = frames2

        return results


class RandomFlipVideo:
    """
    Only support dimension order (t,c,h,w)!
    """

    __init__ = RandomFlip.__init__

    def __call__(self, results):
        flip = True if np.random.rand() < self.flip_prob else False
        if flip:
            frames = results  # (t,c,h,w)
            direction = self.direction
            if direction == 'horizontal':
                frames2 = np.flip(frames, axis=2)
            elif direction == 'vertical':
                frames2 = np.flip(frames, axis=3)
            else:
                raise NotImplemented
            results = frames2

        return results


class RandomResizedCropVideo:
    """
    Only support dimension order (t,c,h,w)!
    """

    def __init__(self, center=(0, 0), *args, **kwargs):
        RandomResizedCrop.__init__(self, *args, **kwargs)
        self.center = center

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            frames = results[key]
            if not any(self.center):
                img = frames[0].transpose([1, 2, 0])
                if self.efficientnet_style:
                    get_params_func = RandomResizedCrop.get_params_efficientnet_style
                    get_params_args = dict(
                        img=img, size=self.size, scale=self.scale, ratio=self.ratio, max_attempts=self.max_attempts,
                        min_covered=self.min_covered, crop_padding=self.crop_padding)
                else:
                    get_params_func = RandomResizedCrop.get_params
                    get_params_args = dict(
                        img=img, scale=self.scale, ratio=self.ratio, max_attempts=self.max_attempts)
                ymin, xmin, ymax, xmax = get_params_func(**get_params_args)
            else:
                ymin, xmin, ymax, xmax = self.get_center_crop_params(*frames.shape[-2:], self.scale, self.ratio,
                    self.center, self.max_attempts)

            frames2 = frames[..., ymin:ymax + 1, xmin:xmax + 1]           
            frames2 = np.stack(
                [imresize(_.transpose([1, 2, 0]), self.size[::-1], False, self.interpolation,
                    backend=self.backend) for _ in frames2]
            ).transpose([0, 3, 1, 2])

            results[key] = frames2

        return results

    @staticmethod
    def get_center_crop_params(height, width, scale, ratio, center, max_attempts=10):
        area = height * width
        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                if center[0]:
                    ymin = (height - target_height) // 2
                else:
                    ymin = random.randint(0, height - target_height)
                if center[1]:
                    xmin = (width - target_width) // 2
                else:
                    xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

class ResizeVideo:
    """
    Only support dimension order (t,c,h,w)!
    """
    __init__ = Resize.__init__

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            frames = results[key]
            ignore_resize = False
            if self.resize_w_short_side:
                h, w = frames.shape[-2:]
                short_side = self.size[0]
                if (w <= h and w == short_side) or (h <= w and h == short_side):
                    ignore_resize = True
                else:
                    if w < h:
                        width, height = short_side, int(short_side * h / w)
                    else:
                        height, width = short_side, int(short_side * w / h)
            else:
                height, width = self.size
            if not ignore_resize:
                frames2 = np.stack(
                    [imresize(_.transpose([1, 2, 0]), (width, height), False, self.interpolation,
                        backend=self.backend) for _ in frames]
                ).transpose([0, 3, 1, 2])
                results[key] = frames2
                results['img_shape'] = frames2.shape

        return results

class ResizeDVS:
    """
    Only support dimension order (t,c,h,w)!
    """
    __init__ = Resize.__init__

    def __call__(self, results):
        
        frames = results
        ignore_resize = False
        if self.resize_w_short_side:
            h, w = frames.shape[-2:]
            short_side = self.size[0]
            if (w <= h and w == short_side) or (h <= w and h == short_side):
                ignore_resize = True
            else:
                if w < h:
                    width, height = short_side, int(short_side * h / w)
                else:
                    height, width = short_side, int(short_side * w / h)
        else:
            height, width = self.size
        if not ignore_resize:
            frames2 = np.stack(
                [imresize(_.transpose([1, 2, 0]), (width, height), False, self.interpolation,
                        backend=self.backend) for _ in frames]
            ).transpose([0, 3, 1, 2])
            
            #frames2=torch.stack([F.resize(_, (width, height)) for _ in frames])
            results = frames2

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str

class CenterCropVideo:
    """
    Only support dimension order (t,c,h,w)!
    """
    __init__ = CenterCrop.__init__

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            frames = results[key]
            img_height, img_width = frames.shape[-2:]

            if self.efficientnet_style:
                img_short = min(img_height, img_width)
                crop_height = crop_height / (crop_height + self.crop_padding) * img_short
                crop_width = crop_width / (crop_width + self.crop_padding) * img_short

            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height)  - 1
            x2 = min(img_width, x1 + crop_width)  - 1

            frames2 = frames[:, :, y1:y2 + 1, x1:x2 + 1]
            assert frames2.shape[-2:] == (224, 224) or frames2.shape[-2:] == (112, 112)

            if self.efficientnet_style:
                frames2 = np.stack(
                    [imresize(_.transpose([1, 2, 0]), self.crop_size[::-1], False, self.interpolation,
                        backend=self.backend) for _ in frames2]
                ).transpose([0, 3, 1, 2])
            results[key] = frames2

        results['img_shape'] = frames2.shape

        return results



class BrightnessVideo:
    """
    Only support dimension order (t,c,h,w)!
    """
    __init__ = Brightness.__init__

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            frames = results[key]
            frames_brightened = self.adjust_brightness(frames, factor=1 + magnitude)
            results[key] = frames_brightened
        return results

    @staticmethod
    def adjust_brightness(block, factor):
        assert block.dtype == np.uint8
        factor = np.array(factor, dtype='float32')
        brightened_block = (block * factor).astype('uint8')
        brightened_block = np.clip(brightened_block, 0, 255)
        return brightened_block



class ContrastVideo:
    """
    Only support dimension order (t,c,h,w)!
    """
    __init__ = Contrast.__init__

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            frames = results[key]
            frames_contrasted = self.adjust_contrast(frames, factor=1 + magnitude)
            results[key] = frames_contrasted
        return results

    @staticmethod
    def adjust_contrast(block, factor):
        """The ``c`` in (t, c, h, w) must be RGB."""
        assert block.dtype == np.uint8
        factor = np.array(factor, dtype='float32')
        gray_block = (block[:, 0, :, :] * .299 + block[:, 1, :, :] * .587 + block[:, 2, :, :] * .114).astype('uint8')
        hist, _ = np.histogram(gray_block, 256, (0, 255))
        mean = np.sum(gray_block) / np.sum(hist)
        degenerated = np.array([.299, .587, .114], dtype='float32')[None, :, None, None] * mean.astype('float32')
        contrasted_block = (block * factor + degenerated * (1 - factor)).astype('uint8')
        contrasted_block = np.clip(contrasted_block, 0, 255)
        return contrasted_block



class ColorTransformVideo:
    """
    Only support dimension order (t,c,h,w)!
    """
    __init__ = ColorTransform.__init__

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            block = results[key]
            # img_color_adjusted = mmcv.adjust_color(block, alpha=1 + magnitude)
            img_color_adjusted = self.adjust_color(block, alpha=1 + magnitude)
            results[key] = img_color_adjusted.astype(block.dtype)
        return results

    @staticmethod
    def adjust_color(block, alpha):
        assert block.dtype == np.uint8
        alpha = np.array(alpha, dtype='float32')
        gray_block = (block[:, 0, :, :] * .299 + block[:, 1, :, :] * .587 + block[:, 2, :, :] * .114).astype('uint8')
        colored_block = (block * alpha + gray_block[:, None, :, :] * (1 - alpha)).astype('uint8')
        colored_block = np.clip(colored_block, 0, 255)
        return colored_block



class ColorJitterVideo:
    """
    Only support dimension order (t,c,h,w)!
    """
    __init__ = ColorJitter.__init__

    def __call__(self, results):
        brightness_factor = random.uniform(0, self.brightness)
        contrast_factor = random.uniform(0, self.contrast)
        saturation_factor = random.uniform(0, self.saturation)
        color_jitter_transforms = [
            dict(type='BrightnessVideo', magnitude=brightness_factor, prob=1., random_negative_prob=0.5),
            dict(type='ContrastVideo', magnitude=contrast_factor, prob=1., random_negative_prob=0.5),
            dict(type='ColorTransformVideo', magnitude=saturation_factor, prob=1., random_negative_prob=0.5)
        ]
        random.shuffle(color_jitter_transforms)
        transform = Compose(color_jitter_transforms)
        return transform(results)



class CutOutVideo:
    """
    Only support dimension order (t,c,h,w)!
    """

    def __init__(self, area: float, fill=0, prob=0.5):
        assert 0 < area < 1
        if fill is not None:
            assert isinstance(fill, (int, list, tuple))  # the last two type for per-channel-fill
        assert 0 < prob < 1
        self.area = area
        self.fill = fill
        self.prob = prob

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            frames = results[key]
            t, c, h, w = frames.shape
            flags = (np.random.rand(t) < self.prob)[:, None, None, None]
            if not any(flags):
                break

            area = self.area * h * w
            ratio = area / min(area, h - 1) ** 2, area / max(area / w, 1 + 1) ** 2
            r = np.exp(np.random.uniform(*np.log(ratio)))
            h2, w2 = int(np.sqrt(area / r)), int(np.sqrt(area * r))
            x1, y1 = np.random.randint(0, w - w2), np.random.randint(0, h - h2)

            if self.fill is None:
                fill = np.random.randint(frames.min(), frames.max() + 1, c, dtype='uint8')
            elif isinstance(self.fill, int):
                fill = np.array([self.fill] * c, dtype='uint8')
            else:
                fill = np.array(self.fill, dtype='uint8')

            block = frames[:, :, y1:y1 + h2, x1:x1 + w2]
            block[...] = np.where(flags, fill[None, :, None, None] * np.ones([t, c, h2, w2], dtype='uint8'), block)
            results[key] = frames

        return results


class RandomNoiseDict:
    """
    Only support dimension order (t,c,h,w)!
    """

    def __init__(self, noise,*args, **kwargs):        
        self.noise = noise  
        self.plt = False

    def convert2gray(self,inpt):
        img = copy.deepcopy(inpt)
        img = img.astype(np.uint8)
        x,y = np.where(img!=0)
        img[x,y] = 255
        return img

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            frames = results[key]  # (t,c,h,w)           
            size = frames.shape[-1]*frames.shape[-2]
            noise_num = int(size*self.noise)
             
            points_x  = np.random.randint(0, frames.shape[-1], noise_num)
            points_y = np.random.randint(0, frames.shape[-2], noise_num)

            frames[..., points_x,points_y] += 1. 

            results[key] = frames


        return results
    
class RandomNoise:
    """
    Only support dimension order (t,c,h,w)!
    """

    def __init__(self, noise,*args, **kwargs):        
        self.noise = noise  
        self.plt = False

    def convert2gray(self,inpt):
        img = copy.deepcopy(inpt)
        img = img.astype(np.uint8)
        x,y = np.where(img!=0)
        img[x,y] = 255
        return img

    def __call__(self, results):
        frames = results # (t,c,h,w)           
        size = frames.shape[-1]*frames.shape[-2]
        noise_num = int(size*self.noise)
             
        points_x  = np.random.randint(0, frames.shape[-1], noise_num)
        points_y = np.random.randint(0, frames.shape[-2], noise_num)

        frames[..., points_x,points_y] += 0.5 

        results = frames


        return results
    


class Rate:  

    def __init__(self, time_steps, gain=1, offset=0):
        assert isinstance(time_steps, int) 
        self.time_steps = time_steps
        self.gain = gain
        self.offset = offset

    def _rate_coding(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]      
            if img.ndim==2:
                img = img.unsqueeze(0)            
            time_data = (
                img.repeat(
                    tuple(
                        [self.time_steps]
                        + torch.ones(len(img.size()), dtype=int).tolist()
                    )
                )
                * self.gain
                + self.offset
            )           

            clipped_data = torch.clamp(time_data, min=0, max=1)
            # pass time_data matrix into bernoulli function.
            spike_data = torch.bernoulli(clipped_data)            
            results[key] = spike_data
            results['img_shape'] = spike_data.shape

    def __call__(self, results):
        self._rate_coding(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str



class Temporal:    

    def __init__(self,  time_steps=False, threshold=0.01, tau=1, first_spike_time=0,
            on_target=1, off_target=0, clip=False, normalize=False, linear=False,interpolate=False,
            epsilon=1e-7,):
        assert isinstance(time_steps, int) 
        self.time_steps = time_steps
        self.threshold = threshold
        self.tau = tau
        self.first_spike_time = first_spike_time
        self.on_target = on_target
        self.off_target = off_target
        self.clip = clip
        self.normalize = normalize
        self.linear = linear
        self.epsilon = epsilon

    def _latency_coding(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]      
            if img.ndim==2:
                img = img.unsqueeze(0)
            
            if torch.min(img) < 0 or torch.max(img) > 1:
                raise Exception(
                    f"Elements of ``data`` must be between [0, 1], but input "
                    f"is [{torch.min(img)}, {torch.max(img)}]"
                )
            spike_time, idx = self._latency_code(
                img,
                num_steps=self.time_steps,
                threshold=self.threshold,
                tau=self.tau,
                first_spike_time=self.first_spike_time,
                normalize=self.normalize,
                linear=self.linear,
                epsilon=self.epsilon,)      
            
            

            spike_data = torch.zeros(
                (tuple([self.time_steps] + list(spike_time.size()))),                      
                
            )           
            

            # use rm_idx to remove spikes beyond the range of num_steps
            rm_idx = torch.round(spike_time).long() > self.time_steps - 1
            spike_data = (
                spike_data.scatter(
                    0,
                    torch.round(torch.clamp_max(spike_time, self.time_steps - 1))
                    .long()
                    .unsqueeze(0),
                    1,
                )
                * ~rm_idx
            )

            # Use idx to remove spikes below the threshold
            if self.clip:
                spike_data = spike_data * ~idx  # idx is broadcast in T direction

            data =  torch.clamp(spike_data * self.on_target, self.off_target)
            
            results[key] = data
            results['img_shape'] = data.shape
                
    def _latency_code(self,
            data,
            num_steps=False,
            threshold=0.01,
            tau=1,
            first_spike_time=0,
            normalize=False,
            linear=False,
            epsilon=1e-7,):
   
        idx = data < threshold

        if not linear:
            spike_time = self._latency_code_log(
                data,
                num_steps=num_steps,
                threshold=threshold,
                tau=tau,
                first_spike_time=first_spike_time,
                normalize=normalize,
                epsilon=epsilon,
            )

        elif linear:
            spike_time = self._latency_code_linear(
                data,
                num_steps=num_steps,
                threshold=threshold,
                tau=tau,
                first_spike_time=first_spike_time,
                normalize=normalize,
            )

        return spike_time, idx
    
    def _latency_code_linear(self,
        data,
        num_steps=False,
        threshold=0.01,
        tau=1,
        first_spike_time=0,
        normalize=False,):
        if normalize:
            tau = num_steps - 1 - first_spike_time

        spike_time = (
            torch.clamp_max((-tau * (data - 1)), -tau * (threshold - 1))
        ) + first_spike_time

        # the following code is intended for negative input data.
        # it is more broadly caught in latency code by ensuring 0 < data < 1.
        # Consider disabling ~(0<data<1) input.
        if torch.min(spike_time) < 0 and normalize:
            spike_time = (
                (spike_time - torch.min(spike_time))
                * (1 / (torch.max(spike_time) - torch.min(spike_time)))
                * (num_steps - 1)
            )
        return spike_time


    def _latency_code_log(self,
        data,
        num_steps=False,
        threshold=0.01,
        tau=1,
        first_spike_time=0,
        normalize=False,
        epsilon=1e-7,
    ):

        
        data = torch.clamp(
            data, threshold + epsilon
        )  # saturates all values below threshold.

        spike_time = tau * torch.log(data / (data - threshold))

        if first_spike_time > 0:
            spike_time += first_spike_time

        if normalize:
            spike_time = (spike_time - first_spike_time) * (
                num_steps - first_spike_time - 1
            ) / torch.max(spike_time - first_spike_time) + first_spike_time

        return spike_time   
    


    def __call__(self, results):
        self._latency_coding(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
    
    
class Population:    

    def __init__(self,  time_steps=False, pop_neuron=1):
        assert isinstance(time_steps, int) 
        self.time_steps = time_steps
        self.pop_neuron = pop_neuron
        sigma = 1/1.5/(self.pop_neuron-2)
        self.norm_list = []
        for i in range(self.pop_neuron):
            u = (2*i-3)/2/(self.pop_neuron-2)           
            
            normal_dist = Normal(u, sigma)
            self.norm_list.append(normal_dist)
       

    def _population_coding(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]      
            if img.ndim==2:
                img = img.unsqueeze(0)
            
            spike_data = [fun.log_prob(img).exp() for fun in self.norm_list]
            spike_data = [1. - data/data.max() for data in spike_data] #归一化到01区间，并大小反序
            #print(spike_data[0].max(), spike_data[1].max())
            spike_data = torch.stack(spike_data).squeeze(1)#.permute(1,0,2,3,4).squeeze(2)
            
            spike_data =(spike_data*self.time_steps).to(torch.int64)  #10拍
            spike_data[spike_data >(self.time_steps-1)] = 0

            spike_data = FUN.one_hot(spike_data,num_classes=self.time_steps).permute(3,0,1,2).to(torch.float32)
            
            
               
            
            results[key] = spike_data
            results['img_shape'] = spike_data.shape
                
    


    def __call__(self, results):
        self._population_coding(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

def _handle_padding_params(self, padding, nested):
    if isinstance(padding, int):
        if nested:
            self.padding = ((padding,) * 2,) * 2
        else:
            self.padding = (padding,) * 4
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        if nested:
            self.padding = ((padding[0],) * 2, (padding[1],) * 2)
        else:
            self.padding = (padding[0],) * 2 + (padding[1],) * 2
    elif isinstance(padding, (list, tuple)) and len(padding) == 4:
        if nested:
            self.padding = tuple(padding[:2]) + tuple(padding[2:])
        else:
            self.padding = padding
    elif padding is None:
        self.padding = None
    else:
        raise NotImplemented