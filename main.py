#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Time    : 2023/10/16 12:48
# Author  : dongchao
# File    : main.py
# Software: PyCharm


import sys
from datetime import date
import os

from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, QTextStream, QStringListModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog

from Qt_seg import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.load_image_Button.clicked.connect(self.show_message_load_image_Button)
        self.start_Button.clicked.connect(self.show_message_start_Button)
        self.open_file_Button.clicked.connect(self.show_message_open_file_Button)

        # Todo listview 添加滚动条

        self.fileList = []
        self.directory = ""

    def setmodel(self, qList):
        # 设置模型列表视图，加载数据列表
        slm = QStringListModel()
        slm.setStringList(qList)
        # 设置列表视图的模型
        self.listView.setModel(slm)

    # 关闭窗口时弹出确认消息
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Warning', '确认退出？', QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def show_message_load_image_Button(self):
        print("click load_image_Button")

    def show_message_start_Button(self):
        print("click startButton")

    def show_message_open_file_Button(self):

        self.directory = QFileDialog.getExistingDirectory(None, "选取文件夹", "./")  # 起始路径
        print("directory = ", self.directory)
        try:
            self.fileList = os.listdir(self.directory)
        except Exception as e:
            print(e)
        print("len of filelist = ", len(self.fileList))
        self.setmodel(self.fileList)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWin = MainWindow()

    myWin.show()

    sys.exit(app.exec_())
