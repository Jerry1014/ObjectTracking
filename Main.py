"""
主文件，负责连接个部分和启动
"""
import sys

from PySide2 import QtWidgets
from PySide2.QtCore import QThreadPool, QProcess

from Interface import MainWin
from InterfaceController import InterfaceController
from ModelController import ModelController
from Settings import settings


class Start:
    def __init__(self):
        self.settings = settings
        self.interface_controller = InterfaceController(self.settings)
        self.model_controller = ModelController(self.settings)

    def run(self):
        app = QtWidgets.QApplication([])

        widget = MainWin(self.settings, self.interface_controller.signal_connection)
        QThreadPool.globalInstance().start(self.interface_controller)
        QThreadPool.globalInstance().start(self.model_controller)
        widget.show()

        app.exec_()
        self.settings.if_end = True
        # sys.exit(0)


if __name__ == '__main__':
    Start().run()
