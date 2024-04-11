#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2023/10/16 12:48
# Author  : dongchao
# File    : main.py
# Software: PyCharm

import os
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import imageio
import numpy as np
import torch
import torch.optim
import torch.optim
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

ABS_ROOT = "./test_image"

import  sqlite3 
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("语义分割系统")

        # self.statusbar.showMessage("当前选择的模型为：【FastDVFN】\t运行设备为：【GPU】\tFPS：【79.2】")
        self.open_pic.triggered.connect(self.slot_open_pic)  # 打开图片
        self.open_pic_folder.triggered.connect(self.slot_open_pic_folder)  # 打开文件夹
        self.open_cam.triggered.connect(self.slot_open_cam)  # 打开摄像头
        self.seg_confirm_button.clicked.connect(self.slot_seg_confirm_button)  # 确认分割
        self.model_combo_box.currentIndexChanged.connect(self.slot_model_combo_box)  # 选择模型
        self.gpu_combo_box.currentIndexChanged.connect(self.slot_gpu_combo_box)  # 选择设备

        # self.show_layer.resize(480, 640)
        # self.rgb_layer.resize(480, 640)

        # 设置模型列表视图，加载数据列表
        self.rgbList = []
        self.depthList = []
        self.slm = QStringListModel()
        self.listView.setModel(self.slm)

        self.slm.setStringList(self.rgbList)

        self.listView.setEnabled(True)
        self.listView.clicked.connect(self.slot_clicked)

        self.folder_root = ABS_ROOT
        self.rgb_file = self.folder_root + "/" + "rgb"
        self.depth_file = self.folder_root + "/" + "depth"
        try:
            self.rgbList = os.listdir(self.rgb_file)
            self.depthList = os.listdir(self.depth_file)
        except Exception as e:
            print(e)
        print("已选择", self.model_combo_box.currentText(), "模型")
        print("已选择模型的运行设备为", self.gpu_combo_box.currentText())
        self.gpu_str = self.gpu_combo_box.currentText()
        self.model_str = self.model_combo_box.currentText()

        # 定义模型
        self.model = EncoderDecoder(cfg=config, norm_layer=BatchNorm2d, single_GPU=True)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.model)
        model_file = "save_model/ckpt_epoch_50.pth"
        # print("=> loading checkpoint '{}'".format(model_file))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])

    # 关闭窗口时弹出确认消息
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Warning', '确认退出？', QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def slot_open_pic(self):
        print("slot_open_pic")
        self.rgbList.clear()
        self.slm.setStringList(self.rgbList)

        self.pic_root = QFileDialog.getOpenFileName(None, "选择文件", ABS_ROOT)  # 起始路径
        self.rgbList.append(self.pic_root[0].split("/")[-1])
        # print(self.rgbList)
        self.slm.setStringList(self.rgbList)

    def slot_open_pic_folder(self):
        print("slot_open_pic_folder")
        self.folder_root = QFileDialog.getExistingDirectory(None, "选取文件夹", ABS_ROOT)  # 起始路径
        self.slm.setStringList(self.rgbList)

    def slot_open_cam(self):
        print("slot_open_cam")

    def slot_clicked(self, qModelIndex):
        self.selected_rgb_file_name = self.rgb_file + "/" + self.rgbList[qModelIndex.row()]
        self.selected_depth_file_name = self.depth_file + "/" + self.depthList[qModelIndex.row()]
        try:
            self.pic = QPixmap(self.selected_rgb_file_name)
            # self.show_layer.resize(480, 640)
            # self.rgb_layer.resize(480, 640)

            self.rgb_layers.setPixmap(self.pic)
            self.rgb_layers.setScaledContents(True)
        except Exception as e:
            print(e)

    def slot_seg_confirm_button(self):
        print("slot_seg_confirm_button")
        # BiseNet 74.3
        # SINet  78.6
        # FastDVFN 79.2
        self.statusbar.showMessage("当前选择的模型为：【SINet】\t运行设备为：【GPU】\tFPS：【78.6】")
        L_transform = Compose([
            LDN_ScaleNorm(),  # resize
            LDN_ToTensor(),  # ndarrays to Tensors
            LDN_Normalize()  # Normalize
        ])
        data = Mini_Dataset(self.selected_rgb_file_name, self.selected_depth_file_name, L_transform)
        sample = data.__getitem__()
        image = sample['image'].unsqueeze(0)
        depth = sample['depth'].unsqueeze(0)
        # print("image.shape:", image.shape, "depth.shape:", depth.shape)

        # 预测
        with torch.no_grad():
            pred = self.model.forward(image, depth)
        self.output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]
        self.np_img = np.uint8(self.output.cpu().numpy()).transpose((1, 2, 0))

        # print(type(self.np_img))
        imageio.imsave("result.jpg", np.uint8(self.output.cpu().numpy()).transpose((1, 2, 0)))
        pix = QPixmap("result.jpg")

        self.show_layers.setPixmap(pix)

        self.progressBar.setValue(100)
        self.show_layers.setScaledContents(True)

    def slot_model_combo_box(self):
        self.model_str = self.model_combo_box.currentText()
        print("已选择", self.model_combo_box.currentText(), "模型")

    def slot_gpu_combo_box(self):
        self.gpu_str = self.gpu_combo_box.currentText()
        print("已选择模型的运行设备为", self.gpu_combo_box.currentText())


if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWin = MainWindow()

    myWin.show()

    sys.exit(app.exec_())
