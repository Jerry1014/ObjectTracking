"""
主文件，负责连接个部分和启动
"""
import sys
from queue import Queue

from PySide2 import QtWidgets
from PySide2.QtCore import QThreadPool

from Interface import MainWin
from InterfaceController import InterfaceController
from Model import TestModel
from ModelController import ModelController

SUPPORTED_FORMATS = ('mp4','mkv')


class Start:
    def __init__(self, model_class):
        self.settings = {'supported_formats': SUPPORTED_FORMATS, 'end_sign': False, 'pause_sign': False}
        self.frame_queue = Queue(24)
        self.controller = InterfaceController(self.settings, self.frame_queue)
        self.model_controller = ModelController(self.settings, self.frame_queue, model_class)

    def run(self):
        app = QtWidgets.QApplication([])

        widget = MainWin(self.settings, self.controller.signal_connection)
        QThreadPool.globalInstance().start(self.controller)
        QThreadPool.globalInstance().start(self.model_controller)
        widget.show()

        app.exec_()
        self.settings['end_sign'] = True
        sys.exit(0)


if __name__ == '__main__':
    Start(TestModel).run()
