"""
第一个界面，展示监控界面
"""
from time import sleep

from PySide2 import QtWidgets
from PySide2.QtCore import Slot, Qt, Signal
from PySide2.QtGui import QPixmap, QImage
from DataStructure import FrameData
from Interface.TrackingInterface import TrackingWin


class MonitoringInterface(QtWidgets.QWidget):
    frame_update_signal = Signal(FrameData)
    model_init_signal = Signal(dict)

    def __init__(self, settings, model_init_slot):
        super().__init__()
        # 设置
        self.settings = settings
        self.settings.frame_update_signal = self.frame_update_signal
        self.setWindowTitle('监控界面')
        self.play_state = self.settings.monitor_play_state

        # 等待控制模块读取监控配置文件
        while self.settings.monitor_config_list is None:
            sleep(0.5)
        for _ in range(len(self.settings.monitor_config_list)):
            self.play_state.append(True)

        # 布局
        column_number = 1
        while column_number ** 2 < len(self.settings.monitor_config_list):
            column_number += 1
        self.monitor_list = list()
        self.layout = QtWidgets.QGridLayout()
        for i, config in enumerate(self.settings.monitor_config_list):
            # 是否需要传入更多的信息
            monitor = MonitoringSubInterface(i, config, self.settings.each_monitor_rect, self.change_play_state,
                                             self.change_play_process, self.start_tracking)
            self.monitor_list.append(monitor)
            self.layout.addWidget(monitor, i // column_number + 1, i % column_number + 1)
        self.setLayout(self.layout)

        # 其他
        self.sub_win = None
        self.tem_index_monitor = None
        self.frame_update_signal.connect(self.set_frame)
        self.model_init_signal.connect(model_init_slot)

    @Slot(FrameData)
    def set_frame(self, frame_data):
        self.monitor_list[frame_data.index].set_frame(frame_data)

    @Slot(int)
    def change_play_state(self, index):
        self.play_state[index] = not self.play_state[index]

    @Slot(int, int)
    def change_play_process(self, index, frame_num):
        self.play_state[index] = frame_num

    @Slot(tuple)
    def start_tracking(self, frame):
        index, frame_pixmap, slider_value, slider_max_num = frame
        for index, state in enumerate(self.play_state):
            if state:
                self.monitor_list[index].button_event()
        self.sub_win = TrackingWin(index, self.settings, self.model_init_signal, self.change_play_process,
                                   self.after_close_tracking, self.change_play_state, frame_pixmap, slider_value,
                                   slider_max_num)
        self.sub_win.show()
        self.sub_win.activateWindow()
        self.tem_index_monitor = (index, self.monitor_list[index])
        self.monitor_list[index] = self.sub_win

    @Slot()
    def after_close_tracking(self):
        index, monitor = self.tem_index_monitor
        for index, state in enumerate(self.play_state):
            if not state:
                self.monitor_list[index].button_event()
        self.change_play_state(index)
        self.sub_win = None


class MonitoringSubInterface(QtWidgets.QWidget):
    play_state_change_signal = Signal(int)
    play_process_change_signal = Signal(int, int)
    start_tracking_signal = Signal(tuple)

    def __init__(self, index, monitor_config, monitor_rect, play_state_slot, play_process_slot, start_tracking_slot):
        super().__init__()
        self.index = index
        # 部件
        self.monitor_name = QtWidgets.QLabel(monitor_config.name)
        self.monitor_win = QtWidgets.QLabel()
        self.monitor_win.setFixedSize(*monitor_rect)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_max_num = monitor_config.total_frame_num
        self.slider.setRange(0, self.slider_max_num)
        self.play_button = QtWidgets.QPushButton('暂停')
        self.track_button = QtWidgets.QPushButton('跟踪')
        self.play_state = True

        # 布局
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.monitor_name)
        self.main_layout.addWidget(self.monitor_win)
        self.sub_layout = QtWidgets.QHBoxLayout()
        self.sub_layout.addWidget(self.play_button)
        self.sub_layout.addWidget(self.track_button)
        self.sub_layout.addWidget(self.slider)
        self.main_layout.addLayout(self.sub_layout)
        self.setLayout(self.main_layout)

        self.play_button.clicked.connect(self.button_event)
        self.play_state_change_signal.connect(play_state_slot)
        self.slider.valueChanged.connect(self.slider_event)
        self.play_process_change_signal.connect(play_process_slot)
        self.track_button.clicked.connect(self.track_button_event)
        self.start_tracking_signal.connect(start_tracking_slot)

    def set_frame(self, frame_data):
        frame_data, cur_frame_num = frame_data.frame
        if type(frame_data) == str:
            self.monitor_win.setText(frame_data)
        else:
            h, w, ch = frame_data.shape
            tem_pixmap = QPixmap.fromImage(QImage(frame_data, w, h, ch * w, QImage.Format_RGB888))
            tem_pixmap.scaled(self.monitor_win.size())
            self.monitor_win.setPixmap(tem_pixmap)
            self.slider.blockSignals(True)
            self.slider.setValue(cur_frame_num)
            self.slider.blockSignals(False)

    @Slot()
    def button_event(self):
        if self.play_state:
            self.play_button.setText('播放')
        else:
            self.play_button.setText('暂停')
        self.play_state = not self.play_state
        self.play_state_change_signal.emit(self.index)

    @Slot()
    def slider_event(self, value):
        if self.play_state:
            self.play_button.setText('播放')
            self.play_state = not self.play_state
        self.play_process_change_signal.emit(self.index, value)

    @Slot()
    def track_button_event(self):
        self.start_tracking_signal.emit(
            (self.index, self.monitor_win.pixmap(), self.slider.value(), self.slider_max_num))
