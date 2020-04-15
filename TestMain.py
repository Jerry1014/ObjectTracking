"""
主文件，负责连接个部分和启动
"""
import sys

from PySide2 import QtWidgets
from PySide2.QtCore import QThreadPool

from Interface.Interface import MainWin
from Interface.MonitoringInterface import MonitoringInterface
from Settings import settings
from TestModelController import TestModelController


class Start:
    def __init__(self):
        self.settings = settings
        self.model_controller = TestModelController(self.settings)

    def run(self):
        app = QtWidgets.QApplication([])

        QThreadPool.globalInstance().start(self.model_controller)
        widget = MonitoringInterface(self.settings)
        widget.show()

        app.exec_()
        self.settings.if_end = True
        sys.exit(0)


if __name__ == '__main__':
    Start().run()
