#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/10/16 12:48
# Author  : dongchao
# File    : main.py
# Software: PyCharm


import sys
from datetime import datetime, date
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import pyqtSlot
from Qt_seg import Ui_MainWindow


def shwomessage_load_image_Button():
    print("click load_image_Button")


def shwomessage_startButton():
    print("click startButton")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.load_image_Button.clicked.connect(shwomessage_load_image_Button)
        self.startButton.clicked.connect(shwomessage_startButton)

        self.status_bar_init()

    def status_bar_init(self):
        today = date.today().__str__()
        self.statusbar.showMessage('当前时间 : 【' + today + "】")

    # 添加中文的确认退出提示框
    def closeEvent(self, event):
        quit_msg_box = QMessageBox()

        quit_msg_box.setWindowTitle('确认提示')
        quit_msg_box.setText('你确认退出吗？')

        quit_msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        buttonY = quit_msg_box.button(QMessageBox.Yes)
        buttonY.setText('确定')

        buttonN = quit_msg_box.button(QMessageBox.No)
        buttonN.setText('取消')

        quit_msg_box.exec_()

        if quit_msg_box.clickedButton() == buttonY:
            event.accept()
        else:
            event.ignore()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
