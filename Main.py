"""
主文件，负责连接个部分和启动
"""
import sys

from PySide2 import QtWidgets
from PySide2.QtCore import QThreadPool

from Interface import MainWin
from InterfaceController import InterfaceController
from ModelController import ModelController

SUPPORTED_FORMATS = ('jpg', 'png')


class Start:
    def __init__(self):
        self.settings = {'supported_formats': SUPPORTED_FORMATS, 'end_sign': False}
        self.frame_list = list()
        self.controller = InterfaceController(self.settings, self.frame_list)
        self.model_controller = ModelController(self.settings, self.frame_list)

    def run(self):
        app = QtWidgets.QApplication([])

        widget = MainWin(self.settings, self.controller.signal_connection)
        QThreadPool.globalInstance().start(self.controller)
        QThreadPool.globalInstance().start(self.model_controller)
        widget.show()

        if app.exec_():
            self.settings['end_sign'] = True
            # 退出标志应该用海象符号
            sys.exit(0)


if __name__ == '__main__':
    Start().run()
