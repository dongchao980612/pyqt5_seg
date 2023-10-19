#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/10/19 17:13
# Author  : dongchao
# File    : mini_dataset.py
# Software: PyCharm
import imageio
import torch
import numpy as np
import torch.nn as nn
from imageio.v2 import imread
from torch.nn import BatchNorm2d
from torchvision.transforms import Compose

from LDN_transforms import LDN_ScaleNorm, LDN_ToTensor, LDN_Normalize
from model.DF_config import config
from model.EncoderDecoder import EncoderDecoder
from utils import utils


class Mini_Dataset(object):
    def __init__(self, rgb, depth, transform):
        self.rgb = rgb
        self.depth = depth
        self.transform = transform
        print(self.rgb, self.depth)

    def getitem(self):
        sample = {'image': imread(self.rgb), 'depth': imread(self.depth)}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    L_transform = Compose([
        LDN_ScaleNorm(),  # resize
        LDN_ToTensor(),  # ndarrays to Tensors
        LDN_Normalize()  # Normalize
    ])

    data = Mini_Dataset("test_image/rgb/img-000012.jpg", "test_image/depth/00000012.png", L_transform)

    # 定义模型
    model = EncoderDecoder(cfg=config, norm_layer=BatchNorm2d, single_GPU=True)
    # model.init_weights(cfg=config, pretrained="./DFormer/SUNRGBD_DFormer_Tiny.pth")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model_file = "save_model/ckpt_epoch_50.pth"
    print("=> loading checkpoint '{}'".format(model_file))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        checkpoint = torch.load(model_file)
    else:
        checkpoint = torch.load(model_file,
                                map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("load over.............")

    sample = data.getitem()
    image = sample['image'].unsqueeze(0)
    depth = sample['depth'].unsqueeze(0)

    pred = model.forward(image, depth)

    output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]
    print(output.shape)

    imageio.imsave("result.jpg", np.uint8(output.cpu().numpy()).transpose((1, 2, 0)))
