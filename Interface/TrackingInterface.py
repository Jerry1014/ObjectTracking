from configparser import ConfigParser

from PySide2 import QtWidgets
from PySide2.QtCore import Slot, Signal, QRect, Qt
from PySide2.QtGui import QMouseEvent, QPaintEvent, QPainter, QPixmap, QImage, QCloseEvent


class TrackingWin(QtWidgets.QWidget):
    after_tracking_signal = Signal()
    signal_for_close_new_win = Signal()

    def __init__(self, settings, model_init_signal, frame_pixmap=None, slider_value=None):
        super().__init__()
        self.setWindowTitle('目标跟踪')
        self.settings = settings

        # 部件
        self.image_win = MyImageLabel(self.after_tracking_signal)
        if frame_pixmap:
            self.image_win.setPixmap(frame_pixmap)
        self.button = QtWidgets.QPushButton('请用鼠标选择跟踪对象')
        self.button.setEnabled(False)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        if slider_value:
            self.slider.setValue(slider_value)

        # 布局
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_win)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)

        # 信号槽
        self.after_tracking_signal.connect(self.after_tracking)
        self.model_init_signal = model_init_signal

        # 其他
        self.sub_win = None

    @Slot()
    def after_tracking(self):
        """
        用户选择追踪对象后的处理
        """
        if self._show_msg('确认跟踪对象？') == QtWidgets.QMessageBox.Ok:
            self.button.setText('选择模型')
            cf = ConfigParser()
            cf.read('./Model/ModelConfig.ini')
            model_choose_win = ModelSelectWin(cf.sections(), self.signal_for_close_new_win)
            self.signal_for_close_new_win.connect(self.after_choose_model)
            model_choose_win.show()
            model_choose_win.activateWindow()
            self.sub_win = model_choose_win

            self.settings.tracking_object_rect = self.image_win.mouse_press_rect
            self.image_win.if_paint_mouse = False

    @Slot()
    def after_choose_model(self):
        self.button.setText('模型载入中')
        self.model_init_signal.emit(self.sub_win.get_all_data())
        self.sub_win = None

    def _show_msg(self, msg, if_cancel=None, if_image=None, image=None):
        msg_box = QtWidgets.QMessageBox()
        msg_box.setWindowTitle('提示')
        msg_box.setText(msg)
        msg_box.addButton(QtWidgets.QMessageBox.Ok)
        if if_cancel:
            msg_box.addButton(QtWidgets.QMessageBox.Cancel)
        if if_image:
            msg_box.setIconPixmap(image)
        return msg_box.exec_()

    def set_frame(self, frame):
        frame, cur_frame_num = frame
        if type(frame) == str:
            self.image_win.setText(frame)
        else:
            h, w, ch = frame.shape
            tem_pixmap = QPixmap.fromImage(QImage(frame, w, h, ch * w, QImage.Format_RGB888))
            tem_pixmap.scaled(self.image_win.size())
            self.image_win.setPixmap(tem_pixmap)
            self.slider.blockSignals(True)
            self.slider.setValue(cur_frame_num)
            self.slider.blockSignals(False)


class MyImageLabel(QtWidgets.QLabel):
    def __init__(self, after_tracking_signal):
        """
        特别定义的Label，可以用鼠标单击画矩形
        """
        super().__init__()
        self.mouse_press_rect = None
        self.if_paint_mouse = True
        self.needed_paint_rect_list = list()
        self.after_tracking_signal = after_tracking_signal

    def mousePressEvent(self, event: QMouseEvent):
        if self.if_paint_mouse:
            self.mouse_press_rect = event.localPos().toTuple()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.if_paint_mouse:
            # fixme 鼠标只能由左上画到右下，且没有判断
            end_pos = event.localPos().toTuple()
            self.mouse_press_rect = (
                *self.mouse_press_rect[:2], end_pos[0] - self.mouse_press_rect[0],
                end_pos[1] - self.mouse_press_rect[1])
            self.needed_paint_rect_list = [(self.mouse_press_rect, 'blue')]
            self.after_tracking_signal.emit()
            self.repaint()

    def paintEvent(self, event: QPaintEvent):
        super().paintEvent(event)
        if self.needed_paint_rect_list:
            painter = QPainter(self)
            for rect, color in self.needed_paint_rect_list:
                painter.setPen(color)
                painter.drawRect(QRect(*rect))


class AModelElection(QtWidgets.QWidget):
    color = ('red', 'orange', 'yellow', 'blue')

    def __init__(self, model_name, if_selected=False):
        super().__init__()
        self.check_box = QtWidgets.QCheckBox(model_name)
        self.check_box.isChecked()
        self.check_box.setChecked(if_selected)
        self.color_select = QtWidgets.QComboBox()
        self.color_select.addItems(self.color)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.check_box)
        self.layout.addWidget(self.color_select)
        self.setLayout(self.layout)

    def get_data(self):
        return (self.check_box.text(), self.color_select.currentText()) if self.check_box.isChecked() else (None, None)


class ModelSelectWin(QtWidgets.QWidget):
    def __init__(self, model_name_list, close_signal):
        super().__init__()
        self.setWindowTitle('选择模型')
        self.close_signal = close_signal
        self.layout = QtWidgets.QVBoxLayout()
        self.model_election_list = list()

        first_model_election = AModelElection(model_name_list[0], True)
        self.layout.addWidget(first_model_election)
        self.model_election_list.append(first_model_election)
        for name in model_name_list[1:]:
            model_election = AModelElection(name)
            self.model_election_list.append(model_election)
            self.layout.addWidget(model_election)
        self.setLayout(self.layout)

    def get_all_data(self):
        data_list = dict()
        for i in self.model_election_list:
            model, color = i.get_data()
            if model:
                data_list[model] = color
        return data_list

    def closeEvent(self, event: QCloseEvent):
        self.close_signal.emit()