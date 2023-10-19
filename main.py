#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2023/10/16 12:48
# Author  : dongchao
# File    : main.py
# Software: PyCharm


import os
import sys

from PyQt5.QtCore import QStringListModel
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog

from Qt_seg import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # self.index =qModelIndex()
        self.qList = []
        self.directory = ""
        self.imgW = 0
        self.imgH = 0

        # 设置模型列表视图，加载数据列表
        self.slm = QStringListModel()
        self.listView.setModel(self.slm)

        self.slm.setStringList(self.qList)

        self.load_image_Button.clicked.connect(self.show_message_load_image_Button)
        self.start_Button.clicked.connect(self.show_message_start_Button)
        self.open_file_Button.clicked.connect(self.show_message_open_file_Button)

        self.listView.setEnabled(True)
        self.listView.clicked.connect(self.clicked)
        # Todo listview 添加滚动条

        self.selected_file_name = ""

    # 关闭窗口时弹出确认消息
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Warning', '确认退出？', QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def clicked(self, qModelIndex):
        self.index = qModelIndex

        print(self.qList[qModelIndex.row()])
        selected_file_name = self.directory + "/" + self.qList[qModelIndex.row()]
        print(selected_file_name)
        self.pic = QPixmap(selected_file_name)
        print(self.pic.size())
        self.imgW = self.pic.width()
        self.imgH = self.pic.height()
        self.fScale = self.imgW / self.imgH

        self.label_show.setPixmap(self.pic)
        self.label_show.setScaledContents(True)
        # TODO resize pic

    def show_message_load_image_Button(self):
        print("click load_image_Button")

    def show_message_start_Button(self):
        print("click startButton")

        # 定义模型

        # 加载数据

        # 预测

        ## 显示

    def show_message_open_file_Button(self):

        self.directory = QFileDialog.getExistingDirectory(None, "选取文件夹",
                                                          "E:/githubpro/dataset/sun_rgbd/image/test")  # 起始路径
        print("directory = ", self.directory)
        try:
            self.qList = os.listdir(self.directory)
        except Exception as e:
            print(e)
        print("len of filelist = ", len(self.qList))

        self.slm.setStringList(self.qList)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWin = MainWindow()

    myWin.show()

    sys.exit(app.exec_())
