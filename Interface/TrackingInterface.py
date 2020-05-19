from configparser import ConfigParser

import numpy as np

from PySide2 import QtWidgets
from PySide2.QtCharts import QtCharts
from PySide2.QtCore import Slot, Signal, QRect, Qt
from PySide2.QtGui import QMouseEvent, QPaintEvent, QPainter, QPixmap, QImage, QCloseEvent, QPen


class TrackingWin(QtWidgets.QWidget):
    after_tracking_signal = Signal()
    signal_for_close_new_win = Signal()
    change_play_process_signal = Signal(int, int)
    after_close_tracking_signal = Signal()
    change_play_state_signal = Signal(int)
    restart_tracking_signal = Signal(tuple)

    def __init__(self, index, settings, model_init_signal, change_play_process_slot, after_close_tracking_slot,
                 change_play_state_slot, slider_max_num, start_tracking_slot):
        super().__init__()
        self.setWindowTitle('目标跟踪')
        self.index = index
        self.settings = settings

        # 部件
        self.image_win = MyImageLabel(self.after_tracking_signal)
        self.button = QtWidgets.QPushButton('请用鼠标选择跟踪对象，点击按钮直接使用上次的跟踪对象')
        self.re_init_button = QtWidgets.QPushButton('重新初始化')
        self.re_init_button.setVisible(False)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_max_num = slider_max_num
        self.slider.setRange(0, slider_max_num)

        # 布局
        self.img_layout = QtWidgets.QHBoxLayout()
        self.img_layout.addWidget(self.image_win)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.img_layout)
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.button)
        self.button_layout.addWidget(self.re_init_button)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)

        # 信号槽
        self.after_tracking_signal.connect(self.after_tracking)
        self.model_init_signal = model_init_signal
        self.slider.valueChanged.connect(self.change_play_process_event)
        self.change_play_process_signal.connect(change_play_process_slot)
        self.after_close_tracking_signal.connect(after_close_tracking_slot)
        self.change_play_state_signal.connect(change_play_state_slot)
        self.restart_tracking_signal.connect(start_tracking_slot)
        self.re_init_button.clicked.connect(self.re_init_model)
        self.button.clicked.connect(self.tracking_last_object)

        # 其他
        self.sub_win = None
        self.model_state = 0
        self.benckmart_list = None
        self.benckmart_color_series_set = dict()
        self.score_map_win = None
        self.score_map_win_label = None
        self.settings.if_tracking = True
        self.play_state = False
        self.last_frame = None

    @Slot()
    def tracking_last_object(self):
        if self.settings.tracking_object_rect is not None and self.settings.last_frame is not None:
            self.image_win.if_paint_mouse = False
            self.image_win.mouse_press_rect = self.settings.tracking_object_rect
            self.settings.first_frame = self.settings.last_frame
            self.after_tracking_signal.emit()
        else:
            self._show_msg('无上次跟踪记录，请重新选取跟踪对象')

    @Slot()
    def after_tracking(self):
        """
        用户选择追踪对象后的处理
        """
        tracking_object_image = self.settings.get_image_from_first_frame_by_rect(self.image_win.mouse_press_rect)
        h, w, ch = tracking_object_image.shape
        tracking_object_image_pixmap = QPixmap.fromImage(
            QImage(tracking_object_image, w, h, ch * w, QImage.Format_RGB888))
        if self._show_msg('确认跟踪对象？', if_cancel=True, if_image=True,
                          image=tracking_object_image_pixmap) == QtWidgets.QMessageBox.Ok:
            self.button.setText('选择模型')
            cf = ConfigParser()
            cf.read('./Model/ModelConfig.ini')
            model_choose_win = ModelSelectWin(cf.sections(), self.signal_for_close_new_win)
            self.signal_for_close_new_win.connect(self.after_choose_model)
            model_choose_win.show()
            model_choose_win.activateWindow()
            self.sub_win = model_choose_win

            self.slider.setEnabled(False)
            self.settings.tracking_object_rect = self.image_win.mouse_press_rect
            self.settings.last_frame = self.settings.first_frame
            self.image_win.if_paint_mouse = False

    @Slot()
    def after_choose_model(self):
        self.button.setEnabled(False)
        self.button.clicked.disconnect(self.tracking_last_object)
        self.button.setText('模型载入中')
        all_data = self.sub_win.get_all_data()
        self.sub_win = None
        self.model_init_signal.emit(all_data)
        self.model_state = 1
        self.change_play_state_signal.emit(self.index)

    @Slot()
    def re_init_model(self):
        self.close()
        self.restart_tracking_signal.emit((self.index, self.last_frame, self.slider.value(), self.slider_max_num))

    @Slot()
    def change_play_process_event(self, value):
        self.change_play_process_signal.emit(self.index, value)

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

    def set_frame(self, frame_data):
        frame, cur_frame_num = frame_data.frame
        benckmark = frame_data.benckmark
        score_map_list = frame_data.score_map_list
        self.last_frame = frame
        if type(frame) == str:
            self.image_win.setText(frame)
        else:
            h, w, ch = frame.shape
            tem_pixmap = QPixmap.fromImage(QImage(frame, w, h, ch * w, QImage.Format_RGB888))
            if tem_pixmap.size().toTuple() != self.image_win.size().toTuple():
                if self.model_state != 0:
                    tem_pixmap = tem_pixmap.scaled(*self.image_win.size().toTuple())
                else:
                    self.image_win.setFixedSize(tem_pixmap.size())
            self.image_win.setPixmap(tem_pixmap)
            self.slider.blockSignals(True)
            self.slider.setValue(cur_frame_num)
            self.slider.blockSignals(False)
            if self.model_state == 2:
                self.set_score_map(score_map_list)
                self.set_benckmark(cur_frame_num, benckmark)
        if self.model_state == 0:
            # 未选择跟踪目标
            self.settings.first_frame = frame
        elif self.model_state == 1:
            # 选择完模型后
            self.model_state = 2
            self.button.setEnabled(True)
            self.button.clicked.connect(self.pause_tracking)
            self.button.click()
            self.re_init_button.setVisible(True)
        self.image_win.needed_paint_rect_list = frame_data.model_result

    def set_score_map(self, score_map_list):
        @Slot()
        def change_combox(index):
            self.score_map_win_label = index

        if score_map_list:
            if self.score_map_win is None:
                self.score_map_win_label = 0
                new_combobox = QtWidgets.QComboBox()
                new_combobox.currentIndexChanged.connect(change_combox)
                self.score_map_win = QtWidgets.QLabel()
                self.score_map_win.setFixedSize(self.image_win.size())
                for model_name, _ in score_map_list:
                    new_combobox.addItem(model_name)
                new_layout = QtWidgets.QVBoxLayout()
                new_layout.addWidget(self.score_map_win)
                new_layout.addWidget(new_combobox)
                self.img_layout.addLayout(new_layout)

            model_name, score_map = score_map_list[self.score_map_win_label]
            if score_map is not None:
                h, w, ch = score_map.shape
                tem_pixmap = QPixmap.fromImage(QImage(score_map, w, h, ch * w, QImage.Format_RGB888))
                if tem_pixmap.size().toTuple() != self.score_map_win.size().toTuple():
                    tem_pixmap = tem_pixmap.scaled(*self.score_map_win.size().toTuple())
                self.score_map_win.setPixmap(tem_pixmap)
            else:
                self.score_map_win.setText('当前模型无score map')

    def set_benckmark(self, x, benckmark_list):
        if benckmark_list:
            # 初始化
            if self.benckmart_list is None:
                self.benckmart_list = list()
                for benckmart in benckmark_list:
                    new_widget = QtCharts.QChartView()
                    height = self.size().toTuple()[1]
                    new_widget.setFixedHeight(height/2)
                    new_widget.chart().setTitle(benckmart[0])
                    self.layout.addWidget(new_widget)
                    color_data_series = dict()
                    self.benckmart_list.append((new_widget, color_data_series))
                    for model_result in benckmart[1:]:
                        new_data_series = QtCharts.QSplineSeries()
                        new_data_series.setPen(QPen(model_result[1]))
                        new_widget.chart().addSeries(new_data_series)
                        color_data_series[model_result[1]] = [new_data_series, model_result[0]]

            # 添加点
            for benckmart, chart_view_and_data_series_set in zip(benckmark_list, self.benckmart_list):
                chart_view, data_series_set = chart_view_and_data_series_set
                for model_result in benckmart[1:]:
                    data_series, avg = data_series_set[model_result[1]]
                    data_series.append(x, model_result[0])
                    # 计算平均值
                    data_series_count = data_series.count()
                    if data_series_count == 0:
                        avg = model_result[0]
                    else:
                        avg = (avg + model_result[0] / data_series_count) * data_series_count / (data_series_count + 1)
                    data_series_set[model_result[1]][1] = avg
                    data_series.setName(str(avg))
                    chart_view.chart().removeSeries(data_series)
                    chart_view.chart().addSeries(data_series)
                chart_view.chart().createDefaultAxes()

    def closeEvent(self, event):
        self.after_close_tracking_signal.emit()

    @Slot()
    def pause_tracking(self):
        self.button.setText('开始')
        self.button.clicked.disconnect(self.pause_tracking)
        self.button.clicked.connect(self.start_tracking)
        self.change_play_state_signal.emit(self.index)
        self.play_state = False

    @Slot()
    def start_tracking(self):
        """
        用户按下开始的处理
        """
        self.button.setText('暂停')
        self.button.clicked.disconnect(self.start_tracking)
        self.button.clicked.connect(self.pause_tracking)
        self.change_play_state_signal.emit(self.index)
        self.play_state = True


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
        self.button = QtWidgets.QPushButton('确认')
        self.button.clicked.connect(self.button_event)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def get_all_data(self):
        data_list = dict()
        for i in self.model_election_list:
            model, color = i.get_data()
            if model:
                data_list[model] = color
        return data_list

    def button_event(self):
        self.button.setEnabled(False)
        self.button.setText('模型载入中')
        self.close()

    def closeEvent(self, event: QCloseEvent):
        self.close_signal.emit()
