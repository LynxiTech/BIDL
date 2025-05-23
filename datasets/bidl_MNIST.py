# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

import codecs
import os
import os.path as osp
import numpy as np
import torch
import torch.distributed as dist


from .base_dataset import BaseDataset

from .utils import download_and_extract_archive, rm_suffix
from PIL import Image
import sys
SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}

def get_dist_info():
    return 1, 0  


def master_only(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class MNIST(BaseDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
    """  # noqa: E501

    resource_prefix = 'http://yann.lecun.com/exdb/mnist/'
    resources = {
        'train_image_file':
        ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        'train_label_file':
        ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        'test_image_file':
        ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
        'test_label_file':
        ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
    }

    CLASSES = [
        '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five',
        '6 - six', '7 - seven', '8 - eight', '9 - nine'
    ]

    def load_annotations(self):
        train_image_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['train_image_file'][0]))
        train_label_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['train_label_file'][0]))
        test_image_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['test_image_file'][0]))
        test_label_file = osp.join(
            self.data_prefix, rm_suffix(self.resources['test_label_file'][0]))

        if not osp.exists(train_image_file) or not osp.exists(
                train_label_file) or not osp.exists(
                    test_image_file) or not osp.exists(test_label_file):
            self.download()

        _, world_size = get_dist_info()
        if world_size > 1:
            dist.barrier()
            assert osp.exists(train_image_file) and osp.exists(
                train_label_file) and osp.exists(
                    test_image_file) and osp.exists(test_label_file), \
                'Shared storage seems unavailable. Please download dataset ' \
                f'manually through {self.resource_prefix}.'

        train_set = (read_image_file(train_image_file),
                     read_label_file(train_label_file))
        test_set = (read_image_file(test_image_file),
                    read_label_file(test_label_file))

        if not self.test_mode:
            imgs, gt_labels = train_set
        else:
            imgs, gt_labels = test_set
            
        

        data_infos = []
        for img, gt_label in zip(imgs, gt_labels):
            
            gt_label = np.array(gt_label, dtype=np.int64)
            
            info = {'img': Image.fromarray(img.numpy(), mode="L"), 'gt_label': gt_label}
            data_infos.append(info)
        return data_infos

    @master_only
    def download(self):
        os.makedirs(self.data_prefix, exist_ok=True)

        # download files
        for url, md5 in self.resources.values():
            url = osp.join(self.resource_prefix, url)
            filename = url.rpartition('/')[2]
            download_and_extract_archive(
                url,
                download_root=self.data_prefix,
                filename=filename,
                md5=md5)



class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_
    Dataset."""

    resource_prefix = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'  # noqa: E501
    resources = {
        'train_image_file':
        ('train-images-idx3-ubyte.gz', '8d4fb7e6c68d591d4c3dfef9ec88bf0d'),
        'train_label_file':
        ('train-labels-idx1-ubyte.gz', '25c81989df183df01b3e8a0aad5dffbe'),
        'test_image_file':
        ('t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79'),
        'test_label_file':
        ('t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310')
    }
    CLASSES = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
        'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.

    Decompression occurs when argument `path` is a string and ends with '.gz'
    or '.xz'.
    """
    if not isinstance(path, str):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = torch.iinfo(torch_type).bits // 8
    # The MNIST format uses the big endian byte order. If the system uses little endian byte order by default,
    # we need to reverse the bytes before we can read them with torch.frombuffer().
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_label_file(path):
    #with open(path, 'rb') as f:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    
    #with open(path, 'rb') as f:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert (x.dtype == torch.uint8)
    assert (x.ndimension() == 3)
    return x

