#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/10/11 10:09
# Author  : dongchao
# File    : DF_config.py
# Software: PyCharm

import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

"""Dataset Path"""
C.num_classes = 37


""" Settings for network, this would be different for each kind of LDN_model"""
C.backbone = 'DFormer-Tiny' # Remember change the path below.
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8
C.nepochs = 500
C.num_workers = 0
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.drop_path_rate=0.1
C.aux_rate=0.0