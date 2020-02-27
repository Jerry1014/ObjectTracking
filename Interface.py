# -*- coding:utf-8 -*-
import sys

from PySide2 import QtWidgets
from PySide2.QtCore import Slot, Signal, Qt, QRect
from PySide2.QtGui import QPixmap, QImage, QMouseEvent, QPaintEvent, QPainter


class MainWin(QtWidgets.QWidget):
    selected_file = Signal(str)
    after_setting_tracking_object = Signal(bool)

    def __init__(self, settings: dict, signal_connection):
        super().__init__()
        self.settings = settings
        fixed_size = self.settings.get('fixed_size', (1000, 800))
        self.setFixedSize(*fixed_size[:2])
        self.settings['pause_sign'] = None
        self.if_setting_tracking_object_step = None
        self.paint_rect = None
        if signal_connection is not None:
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
        self.after_setting_tracking_object.connect(self.pause_tracking)

    @Slot()
    def load_video(self):
        """
        重要！！！ 对文件的类型等检查在此完成
        """
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

        self.start_pause_button.setText('请用鼠标选定跟踪对象')
        self.start_pause_button.setEnabled(False)
        self.if_setting_tracking_object_step = True
        self.start_pause_button.clicked.disconnect(self.load_video)

    def mousePressEvent(self, event: QMouseEvent):
        if self.if_setting_tracking_object_step:
            self.paint_rect = (*event.localPos().toTuple(), 0, 0)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.if_setting_tracking_object_step:
            end_pos = event.localPos().toTuple()
            self.paint_rect = (*self.paint_rect[:2], end_pos[0] - self.paint_rect[0], end_pos[1] - self.paint_rect[1])
            self.repaint()
            if self.show_msg('确认选择？') == QtWidgets.QMessageBox.Ok:
                self.if_setting_tracking_object_step = False
                self.start_pause_button.setEnabled(True)
                self.after_setting_tracking_object.emit(True)

    def paintEvent(self, event: QPaintEvent):
        if self.paint_rect:
            painter = QPainter(self)
            painter.setPen(Qt.blue)
            painter.drawRect(QRect(*self.paint_rect))

    @Slot(bool)
    def pause_tracking(self, if_just_start: bool):
        self.start_pause_button.setText('开始')
        if not if_just_start:
            self.start_pause_button.clicked.disconnect(self.pause_tracking)
        self.start_pause_button.clicked.connect(self.start_tracking)
        self.settings['pause_sign'] = True

    @Slot()
    def start_tracking(self):
        self.start_pause_button.setText('暂停')
        self.start_pause_button.clicked.disconnect(self.start_tracking)
        self.start_pause_button.clicked.connect(self.pause_tracking)
        self.settings['pause_sign'] = False

    @Slot(str)
    def show_msg(self, msg: str):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(msg)
        msg_box.addButton(QtWidgets.QMessageBox.Ok)
        msg_box.addButton(QtWidgets.QMessageBox.Cancel)
        return msg_box.exec_()

    @Slot()
    def set_pic(self, image):
        """
        显示图片
        :param image: 一定要是ndarray，RGB888格式
        """
        h, w, ch = image.shape
        self.setFixedSize(w, h)
        self.image_win.setPixmap(QPixmap.fromImage(QImage(image, w, h, ch * w, QImage.Format_RGB888)))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    test_settings = {'supported_formats': ('jpg')}

    widget = MainWin(test_settings, None)
    widget.show()

    sys.exit(app.exec_())
