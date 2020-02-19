# -*- coding:utf-8 -*-
import sys

from PySide2 import QtWidgets
from PySide2.QtCore import Slot


class MainWin(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1000, 800)
        self._if_video_ready = False

        # 窗口部件
        self.video_win = QtWidgets.QLabel('Hello World')
        self.start_pause_button = QtWidgets.QPushButton('载入视频')

        # 布局
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.video_win)
        self.layout.addWidget(self.start_pause_button)
        self.setLayout(self.layout)

        # 信号 槽
        self.start_pause_button.clicked.connect(self.load_video)

    @Slot()
    def load_video(self):
        dialog = QtWidgets.QFileDialog()
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dialog.exec_():
            # todo 导入视频文件后的准备动作
            print(f'导入文件{dialog.selectedFiles()}')

        self.start_pause_button.setText('选定跟踪对象')
        self.start_pause_button.clicked.connect(self.set_tracking_object)

    @Slot()
    def set_tracking_object(self):
        self.start_pause_button.setText('开始')
        self.start_pause_button.clicked.connect(self.set_tracking_object)

    @Slot()
    def start_tracking(self):
        self.start_pause_button.setText('暂停')
        self.start_pause_button.clicked.connect(self.pause_tracking)

    @Slot()
    def pause_tracking(self):
        self.start_pause_button.setText('开始')
        self.start_pause_button.clicked.connect(self.continue_tracking)

    @Slot()
    def continue_tracking(self):
        self.start_pause_button.setText('暂停')
        self.start_pause_button.clicked.connect(self.pause_tracking)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MainWin()
    widget.show()

    sys.exit(app.exec_())
