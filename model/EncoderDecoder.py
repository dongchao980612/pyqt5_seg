#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/10/11 10:02
# Author  : dongchao
# File    : EncoderDecoder.py
# Software: PyCharm
from logging import info

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BatchNorm2d

from utils.utils import init_weight


class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d, single_GPU=False):
        global backbone
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer

        ## backbone
        if cfg.backbone == 'DFormer-Large':
            from model.DFormer import DFormer_Large as backbone
            self.channels = [96, 192, 288, 576]
        elif cfg.backbone == 'DFormer-Base':
            from model.DFormer import DFormer_Base as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Small':
            from model.DFormer import DFormer_Small as backbone
            self.channels = [64, 128, 256, 512]
        elif cfg.backbone == 'DFormer-Tiny':
            from model.DFormer import DFormer_Tiny as backbone
            self.channels = [32, 64, 128, 256]

        if single_GPU:
            info('single GPU')
            norm_cfg = dict(type='BN', requires_grad=True)
        else:
            norm_cfg = dict(type='SyncBN', requires_grad=True)

        if cfg.drop_path_rate is not None:
            self.backbone = backbone(drop_path_rate=cfg.drop_path_rate, norm_cfg=norm_cfg)
        else:
            self.backbone = backbone(drop_path_rate=0.1, norm_cfg=norm_cfg)

        self.aux_head = None

        ## decode_head
        if cfg.decoder == 'MLPDecoder':
            info('Using MLP Decoder')
            from model.decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes,
                                           norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        elif cfg.decoder == 'ham':
            info('Using MLP Decoder')
            print(cfg.num_classes)

            from model.decoders.ham_head import LightHamHead as DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels[1:], num_classes=cfg.num_classes,
                                           in_index=[1, 2, 3], norm_cfg=norm_cfg, channels=cfg.decoder_embed_dim)
            from model.decoders.fcnhead import FCNHead
            if cfg.aux_rate != 0:
                self.aux_index = 2
                self.aux_rate = cfg.aux_rate
                info('aux rate is set to', str(self.aux_rate))
                self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'UPernet':
            info('Using Upernet Decoder')
            from model.decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer,
                                        channels=512)
            from model.decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'deeplabv3+':
            info('Using Decoder: DeepLabV3+')
            from model.decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from model.decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        elif cfg.decoder == 'nl':
            info('Using Decoder: nl+')
            from model.decoders.nl_head import NLHead as Head
            self.decode_head = Head(in_channels=self.channels[1:], in_index=[1, 2, 3], num_classes=cfg.num_classes,
                                    norm_cfg=dict(type='SyncBN', requires_grad=True), channels=512)
            from model.decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        else:
            info('No decoder(FCN-32s)')
            from model.decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes,
                                       norm_layer=norm_layer)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            print('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        print('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[-2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x=None, label=None):
        # print('builder',rgb.shape,modal_x.shape)
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out


if __name__ == '__main__':
    from model.DF_config import config

    model = EncoderDecoder(cfg=config, norm_layer=BatchNorm2d, single_GPU=True)
    model.init_weights(cfg=config, pretrained=None)
    # print(LDN_model)

    image_w = 640
    image_h = 480
    batch_size = 2
    rgb_channels = 3
    depth_channels = 1
    rgb = torch.zeros((batch_size, rgb_channels, image_h, image_w))
    depth = torch.zeros((batch_size, depth_channels, image_h, image_w))
    # for i in LDN_model.forward(rgb,depth):
    #     print(" i ",i.shape)
    print(model.forward(rgb, depth).shape)
    from utils.utils import compute_speed_two

    # print(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    compute_speed_two(model, (batch_size, rgb_channels, image_h, image_w),
                      (batch_size, depth_channels, image_h, image_w), device, 10)

    '''
    =========Speed Testing=========
    Elapsed Time: [0.47 s / 10 iter]
    Speed Time: 47.45 ms / iter   FPS: 21.08
    NonBottleneck1D Number of parameter: 6.28M
    '''
