#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/12/25 9:07
# Author  : dongchao
# File    : create_seg_map.py
# Software: PyCharm


import os
import sys

import imageio
import numpy as np
import torch
import torch.optim
from PIL.Image import Image
from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from torch import nn
from torch.nn import BatchNorm2d
from torchvision.transforms import Compose

from LDN_transforms import LDN_ScaleNorm, LDN_ToTensor, LDN_Normalize
from Qt_seg import Ui_MainWindow
from mini_dataset import Mini_Dataset
from model.DF_config import config
from model.EncoderDecoder import EncoderDecoder
from utils import utils

if __name__ == '__main__':
    # 定义模型
    model = EncoderDecoder(cfg=config, norm_layer=BatchNorm2d, single_GPU=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model_file = "save_model/ckpt_epoch_150.pth"
    print("=> loading checkpoint '{}'".format(model_file))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device  = ", device)
    if device.type == 'cuda':
        checkpoint = torch.load(model_file)
    else:
        checkpoint = torch.load(model_file,
                                map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    # 加载数据
    L_transform = Compose([
        LDN_ScaleNorm(),  # resize
        LDN_ToTensor(),  # ndarrays to Tensors
        LDN_Normalize()  # Normalize
    ])
    root = "test_image"
    selected_rgb_file_name_list = os.listdir(root + "/rgb")
    selected_depth_file_name_list = os.listdir(root + "/depth")
    for selected_rgb_file_name, selected_depth_file_name in zip(selected_rgb_file_name_list,
                                                                selected_depth_file_name_list):
        print(selected_rgb_file_name[:10])
        data = Mini_Dataset(root + "/rgb/" + selected_rgb_file_name, root + "/depth/" + selected_depth_file_name, L_transform)
        sample = data.__getitem__()
        image = sample['image'].unsqueeze(0)
        depth = sample['depth'].unsqueeze(0)
        # print("image.shape:", image.shape, "depth.shape:", depth.shape)

        # 预测
        with torch.no_grad():
            pred = model.forward(image, depth)
        output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]

        imageio.imsave("{}/result/{}.jpg".format(root,selected_rgb_file_name[:10]), np.uint8(output.cpu().numpy()).transpose((1, 2, 0)))
        # break
