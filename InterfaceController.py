"""
界面与模型控制之间的中间件
"""
import sys
from time import sleep

from PySide2 import QtWidgets
from PySide2.QtCore import QObject, Signal, QRunnable, QThreadPool, Slot

from Interface import MainWin

SUPPORTED_FORMATS = ('jpg', 'png')


class InterfaceSignalConnection(QObject):
    msg_signal = Signal(str)
    selected_filename = Slot(str)
    pic_signal = Signal(str)


class InterfaceController(QRunnable):
    def __init__(self):
        super().__init__()
        self.signal_connection = InterfaceSignalConnection()
        self.signal_connection.selected_filename = self.after_selected_file
        self.if_selected_file = False
        self.selected_filename = None

    def run(self):
        while not self.if_selected_file:
            pass
        print('000')

    @Slot()
    def after_selected_file(self, filename: str):
        print(f'选择了文件{filename}')
        self.if_selected_file = True
        self.selected_filename = filename

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)


class Start:
    def __init__(self):
        self.settings = {'supported_formats': SUPPORTED_FORMATS}
        self.controller = InterfaceController()

    def run(self):
        app = QtWidgets.QApplication([])

        widget = MainWin(self.settings, self.controller.signal_connection)
        widget.show()
        QThreadPool.globalInstance().start(self.controller)

        sys.exit(app.exec_())


if __name__ == '__main__':
    Start().run()
