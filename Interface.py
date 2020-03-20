# -*- coding:utf-8 -*-
import sys
from queue import Queue

from PySide2 import QtWidgets
from PySide2.QtCore import Slot, Signal, Qt, QRect
from PySide2.QtGui import QPixmap, QImage, QMouseEvent, QPaintEvent, QPainter, QColor


class MainWin(QtWidgets.QWidget):
    signal_selected_file = Signal(str)
    signal_after_setting_tracking_object = Signal()
    signal_for_switch_record_mouse_pos = Signal()
    signal_for_switch_paint = Signal()
    signal_for_rect = Signal(list)

    def __init__(self, settings, signal_connection):
        """
        :param settings: 设置类
        :param signal_connection: 用来连接的外部信号，未做信号存在及未来升级的设计优化
        """
        super().__init__()
        # 设置
        self.settings = settings
        self.setFixedSize(*self.settings.init_fix_rect)
        self.settings.if_pause = False

        # 窗口部件
        self.image_win = MyImageLabel(self.signal_for_switch_record_mouse_pos, self.signal_for_switch_paint,
                                      self.signal_for_rect, self.signal_after_setting_tracking_object)
        self.start_pause_button = QtWidgets.QPushButton('载入视频')

        # 布局
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_win)
        self.layout.addWidget(self.start_pause_button)
        self.setLayout(self.layout)

        # 信号/槽相关
        self.signal_after_setting_tracking_object.connect(self.setting_tracking_object)
        if signal_connection is not None:
            signal_connection.pic_signal.connect(self.set_pic)
            signal_connection.msg_signal.connect(self.show_msg)
            self.signal_selected_file.connect(signal_connection.selected_filename)
        self.set_filename()

    def set_filename(self):
        """
        用户选择视频文件，并对选择的文件做验证
        """
        # 重要！！！ 对文件的类型等检查在此完成
        while True:
            dialog = QtWidgets.QFileDialog()
            dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if dialog.exec_():
                selected_filename: str = dialog.selectedFiles()[0]
                if selected_filename.split('.')[-1] not in self.settings.supported_formats:
                    self._show_msg('不支持的文件格式')
                    continue
                self.signal_selected_file.emit(selected_filename)
                break
            else:
                sys.exit(0)

        self.start_pause_button.setText('请用鼠标选定跟踪对象')
        self.start_pause_button.setEnabled(False)
        self.signal_for_switch_paint.emit()
        self.signal_for_switch_record_mouse_pos.emit()

    @Slot()
    def setting_tracking_object(self):
        """
        用户选择追踪对象后的处理
        """
        tracking_object_image = self.settings.get_image_from_first_frame_by_rect(self.image_win.mouse_press_rect)
        h, w, ch = tracking_object_image.shape
        tracking_object_image_pixmap = QPixmap.fromImage(
            QImage(tracking_object_image, w, h, ch * w, QImage.Format_RGB888))
        if self._show_msg('确认跟踪对象？', if_cancel=True, if_image=True,
                          image=tracking_object_image_pixmap) == QtWidgets.QMessageBox.Ok:
            self.settings.tracking_object = tracking_object_image
            self.signal_for_switch_record_mouse_pos.emit()
            self.start_pause_button.setEnabled(True)
            self.start_pause_button.clicked.connect(self.pause_tracking)
            self.start_pause_button.click()

    @Slot()
    def pause_tracking(self):
        self.start_pause_button.setText('开始')
        self.start_pause_button.clicked.disconnect(self.pause_tracking)
        self.start_pause_button.clicked.connect(self.start_tracking)
        self.settings.if_pause = True

    @Slot()
    def start_tracking(self):
        """
        用户按下开始的处理
        """
        self.start_pause_button.setText('暂停')
        self.start_pause_button.clicked.disconnect(self.start_tracking)
        self.start_pause_button.clicked.connect(self.pause_tracking)
        self.settings.if_pause = False

    @Slot(str)
    def show_msg(self, msg: str):
        """
        展示提示框，未来将优化按键及消息类型
        :param msg: str 提示的信息
        """
        self._show_msg(msg)

    def _show_msg(self, msg, if_cancel=None, if_image=None, image=None):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setText(msg)
        msg_box.addButton(QtWidgets.QMessageBox.Ok)
        if if_cancel:
            msg_box.addButton(QtWidgets.QMessageBox.Cancel)
        if if_image:
            msg_box.setIconPixmap(image)
        return msg_box.exec_()

    @Slot()
    def set_pic(self, frame):
        """
        显示图片
        :param image: 一定要是ndarray，RGB888格式
        """
        image, rect_list = frame
        h, w, ch = image.shape
        self.setFixedSize(w, h)
        self.image_win.setPixmap(QPixmap.fromImage(QImage(image, w, h, ch * w, QImage.Format_RGB888)))
        self.signal_for_rect.emit(rect_list)


class MyImageLabel(QtWidgets.QLabel):
    def __init__(self, signal_for_switch_record_mouse_pos: Signal, signal_for_switch_paint: Signal,
                 signal_for_rect: Signal, signal_after_setting_tracking_object: Signal):
        """
        特别定义的Label，可以用鼠标单击画矩形
        :param signal_for_switch_record_mouse_pos: 改变是否记录鼠标按下释放
        :param signal_for_switch_paint: 改变是否绘制矩形框
        :param signal_for_rect: signal(list rect(x,y,w,h)) 修改绘制的矩形的形状和大小
        :param signal_after_setting_tracking_object: 当鼠标松开后且处在记录鼠标按下释放时，通过其发出信号
        """
        super().__init__()
        self.mouse_press_rect = None
        self.if_record_mouse_pos = False
        self.if_paint = False
        signal_for_switch_record_mouse_pos.connect(self.switch_mouse_pos_record)
        signal_for_switch_paint.connect(self.switch_paint)
        signal_for_rect.connect(self.add_needed_paint_rect)
        self.signal_after_setting_tracking_object = signal_after_setting_tracking_object
        self.needed_paint_rect_list = list()

    @Slot()
    def switch_mouse_pos_record(self):
        self.if_record_mouse_pos = self.if_record_mouse_pos ^ True

    @Slot()
    def switch_paint(self):
        self.if_paint = self.if_paint ^ True

    @Slot()
    def add_needed_paint_rect(self, rect):
        """
        用于改变绘制的矩形的位置和大小
        :param rect: tuple 矩形的x,y,w,h
        """
        self.needed_paint_rect_list = rect

    @Slot()
    def mousePressEvent(self, event: QMouseEvent):
        if self.if_record_mouse_pos:
            self.mouse_press_rect = event.localPos().toTuple()

    def mouseReleaseEvent(self, event: QMouseEvent):
        # fixme 鼠标只能由左上画到右下，且没有判断
        if self.if_record_mouse_pos:
            end_pos = event.localPos().toTuple()
            self.mouse_press_rect = (
            *self.mouse_press_rect[:2], end_pos[0] - self.mouse_press_rect[0], end_pos[1] - self.mouse_press_rect[1])
            self.needed_paint_rect_list.append((self.mouse_press_rect, 'blue'))
            self.repaint()
            self.signal_after_setting_tracking_object.emit()

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        if self.if_paint and self.needed_paint_rect_list:
            painter = QPainter(self)
            for rect, color in self.needed_paint_rect_list:
                painter.setPen(color)
                painter.drawRect(QRect(*rect))
