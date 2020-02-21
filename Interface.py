# -*- coding:utf-8 -*-
import sys
from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import Slot, Signal


class MainWin(QtWidgets.QWidget):
    selected_file = Signal(str)

    def __init__(self, settings: dict, signal_connection):
        super().__init__()
        self.settings = settings
        fixed_size = self.settings.get('fixed_size', (1000, 800))
        self.setFixedSize(*fixed_size[:2])
        self.settings['pause_sign'] = None
        signal_connection.pic_signal.connect(self.set_pic)
        signal_connection.msg_signal.connect(self.show_msg)
        self.selected_file.connect(signal_connection.selected_filename)

        # 窗口部件
        self.image_win = QtWidgets.QLabel()
        self.start_pause_button = QtWidgets.QPushButton('载入视频')

        # 布局
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_win)
        self.layout.addWidget(self.start_pause_button)
        self.setLayout(self.layout)

        # 信号 槽
        self.start_pause_button.clicked.connect(self.load_video)

    @Slot()
    def load_video(self):
        while True:
            dialog = QtWidgets.QFileDialog()
            dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if dialog.exec_():
                selected_filename: str = dialog.selectedFiles()[0]
                if selected_filename.split('.')[-1] not in self.settings['supported_formats']:
                    self.show_msg('不支持的文件格式')
                    continue
                self.selected_file.emit(selected_filename)
                break

        self.start_pause_button.setText('选定跟踪对象')
        self.start_pause_button.clicked.disconnect(self.load_video)
        self.start_pause_button.clicked.connect(self.set_tracking_object)

    @Slot()
    def set_tracking_object(self):
        self.start_pause_button.setText('开始')
        self.start_pause_button.clicked.disconnect(self.set_tracking_object)
        self.start_pause_button.clicked.connect(self.start_tracking)
        self.settings['pause_sign'] = True

    @Slot()
    def start_tracking(self):
        self.start_pause_button.setText('暂停')
        self.start_pause_button.clicked.disconnect(self.start_tracking)
        self.start_pause_button.clicked.connect(self.pause_tracking)
        self.settings['pause_sign'] = False

    @Slot()
    def pause_tracking(self):
        self.start_pause_button.setText('开始')
        self.start_pause_button.clicked.disconnect(self.pause_tracking)
        self.start_pause_button.clicked.connect(self.continue_tracking)
        self.settings['pause_sign'] = True

    @Slot()
    def continue_tracking(self):
        self.start_pause_button.setText('暂停')
        self.start_pause_button.clicked.disconnect(self.continue_tracking)
        self.start_pause_button.clicked.connect(self.pause_tracking)
        self.settings['pause_sign'] = False

    @Slot(str)
    def show_msg(self, msg: str):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(msg)
        msg_box.exec_()

    @Slot()
    def set_pic(self, pic):
        print(pic)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    test_settings = {'supported_formats': ('jpg')}

    widget = MainWin(test_settings)
    widget.show()

    sys.exit(app.exec_())
