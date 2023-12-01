#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2023/10/16 12:48
# Author  : dongchao
# File    : main.py
# Software: PyCharm

import os
import sys

import imageio
import numpy as np
import torch
import torch.optim
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


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # self.index =qModelIndex()
        self.rgbList = []
        self.depthList = []
        self.directory = ""
        self.imgW = 0
        self.imgH = 0

        # 设置模型列表视图，加载数据列表
        self.slm = QStringListModel()
        self.listView.setModel(self.slm)

        self.slm.setStringList(self.rgbList)

        self.load_model_Button.clicked.connect(self.show_message_load_model_Button)
        self.start_Button.clicked.connect(self.show_message_start_Button)
        self.open_file_Button.clicked.connect(self.show_message_open_file_Button)
        self.save_result_Button.clicked.connect(self.show_message_save_result_Button)

        self.listView.setEnabled(True)
        self.listView.clicked.connect(self.clicked)
        # Todo listview 添加滚动条

        self.selected_rgb_file_name = ""
        self.selected_depth_file_name = ""

    # 关闭窗口时弹出确认消息
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Warning', '确认退出？', QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def clicked(self, qModelIndex):
        self.selected_rgb_file_name = self.rgb_file + "/" + self.rgbList[qModelIndex.row()]
        self.selected_depth_file_name = self.depth_file + "/" + self.depthList[qModelIndex.row()]
        # print("selected_rgb_file_name", self.selected_rgb_file_name)
        # print("selected_depth_file_name", self.selected_depth_file_name)
        self.pic = QPixmap(self.selected_rgb_file_name)
        self.label_show.setPixmap(self.pic)
        self.label_show.setScaledContents(True)
        # TODO resize pic

    def show_message_load_model_Button(self):
        print("click load_image_Button")

        # 定义模型
        self.model = EncoderDecoder(cfg=config, norm_layer=BatchNorm2d, single_GPU=True)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(self.model)
        model_file = "save_model/ckpt_epoch_50.pth"
        print("=> loading checkpoint '{}'".format(model_file))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file,
                                    map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("load over.............")

    def show_message_start_Button(self):
        print("click startButton")

        # 加载数据
        L_transform = Compose([
            LDN_ScaleNorm(),  # resize
            LDN_ToTensor(),  # ndarrays to Tensors
            LDN_Normalize()  # Normalize
        ])
        data = Mini_Dataset(self.selected_rgb_file_name, self.selected_depth_file_name, L_transform)
        sample = data.__getitem__()
        image = sample['image'].unsqueeze(0)
        depth = sample['depth'].unsqueeze(0)
        print("image.shape:", image.shape, "depth.shape:", depth.shape)

        # 预测
        with torch.no_grad():
            pred = self.model.forward(image, depth)
        self.output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]

        imageio.imsave("result.jpg", np.uint8(self.output.cpu().numpy()).transpose((1, 2, 0)))
        pix = QPixmap("result.jpg")

        self.label_seg.setPixmap(pix)
        self.label_seg.setScaledContents(True)

    def show_message_open_file_Button(self):
        self.root = QFileDialog.getExistingDirectory(None, "选取文件夹", "./test_image")  # 起始路径
        print("root = ", self.root)
        self.rgb_file = self.root + "/" + "rgb"
        self.depth_file = self.root + "/" + "depth"
        print(self.rgb_file)
        print(self.depth_file)
        try:
            self.rgbList = os.listdir(self.rgb_file)
            self.depthList = os.listdir(self.depth_file)
        except Exception as e:
            print(e)
        print("len of filelist = ", len(self.rgbList))

        self.slm.setStringList(self.rgbList)

    def show_message_save_result_Button(self):
        self.save_root = QFileDialog.getSaveFileName(None, "另存为", "./","普通图像(*.jpg *.png *.bmp)")
        if self.save_root is not  None:
            imageio.imsave("result.jpg", np.uint8(self.output.cpu().numpy()).transpose((1, 2, 0)))
            QMessageBox.about(self, "成功", "保存成功！")
        else:
            QMessageBox.critical(self, "错误！", "未保存任何文件！", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWin = MainWindow()

    myWin.show()

    sys.exit(app.exec_())
