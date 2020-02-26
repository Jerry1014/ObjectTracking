"""
主文件，负责连接个部分和启动
"""
import sys

from PySide2 import QtWidgets
from PySide2.QtCore import QThreadPool

from Interface import MainWin
from InterfaceController import InterfaceController

SUPPORTED_FORMATS = ('jpg', 'png')


class Start:
    def __init__(self):
        self.settings = {'supported_formats': SUPPORTED_FORMATS}
        self.controller = InterfaceController(self.settings)

    def run(self):
        app = QtWidgets.QApplication([])

        widget = MainWin(self.settings, self.controller.signal_connection)
        widget.show()
        QThreadPool.globalInstance().start(self.controller)

        sys.exit(app.exec_())


if __name__ == '__main__':
    Start().run()
