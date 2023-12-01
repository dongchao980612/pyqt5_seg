#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/29 12:34
# Author  : dongchao
# File    : LDN_transforms.py
# Software: PyCharm

import numpy as np
import torch
from skimage.transform import resize
from torchvision import transforms

image_w = 640
image_h = 480


class LDN_ScaleNorm(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        # Bi-linear
        image = resize(image, (image_h, image_w), order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = resize(depth, (image_h, image_w), order=0, mode='reflect', preserve_range=True)


        return {'image': image, 'depth': depth}


# Transforms on torch.Tensor
class LDN_Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(image)
        depth = transforms.Normalize(mean=[19050],
                                     std=[9650])(depth)
        sample['image'] = image
        sample['depth'] = depth

        return sample


class LDN_ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth= sample['image'], sample['depth']

        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float32)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),}
