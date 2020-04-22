"""
主文件，负责连接个部分和启动
"""
import sys

from PySide2 import QtWidgets
from PySide2.QtCore import QThreadPool

from Interface.MonitoringInterface import MonitoringInterface
from Settings import settings
from ModelController import TestModelController


class Start:
    def __init__(self):
        self.settings = settings
        self.model_controller = TestModelController(self.settings)
        self.model_init_slot = self.model_controller.init_object_tracking_model

    def run(self):
        app = QtWidgets.QApplication([])

        QThreadPool.globalInstance().start(self.model_controller)
        widget = MonitoringInterface(self.settings, self.model_init_slot)
        widget.show()

        app.exec_()
        self.settings.if_end = True
        sys.exit(0)


if __name__ == '__main__':
    Start().run()
