"""
第一个界面，展示监控界面
"""
from time import sleep

from PySide2 import QtWidgets
from PySide2.QtCore import Slot, Qt, Signal
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QApplication

from DataStructure import MonitorConfig


class MonitoringInterface(QtWidgets.QWidget):
    frame_update_signal = Signal(int, list, list, list)

    def __init__(self, settings):
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
                                             self.change_play_process)
            self.monitor_list.append(monitor)
            self.layout.addWidget(monitor, i // column_number + 1, i % column_number + 1)
        self.setLayout(self.layout)

        # 其他
        self.sub_win = None
        self.frame_update_signal.connect(self.set_frame)

    @Slot(int, list, list, list)
    def set_frame(self, monitor_num: int, frame, model_result_list, gt):
        self.monitor_list[monitor_num].set_frame(frame[0])

    @Slot(int)
    def change_play_state(self, index):
        self.play_state[index] = not self.play_state[index]

    @Slot(int, int)
    def change_play_process(self, index, frame_num):
        self.play_state[index] = frame_num


class MonitoringSubInterface(QtWidgets.QWidget):
    play_state_change_signal = Signal(int)
    play_process_change_signal = Signal(int, int)

    def __init__(self, index, monitor_config, monitor_rect, play_state_slot, play_process_slot):
        super().__init__()
        self.index = index
        # 部件
        self.monitor_name = QtWidgets.QLabel(monitor_config.name)
        self.monitor_win = QtWidgets.QLabel()
        self.monitor_win.setFixedSize(*monitor_rect)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, monitor_config.total_frame_num)
        self.play_button = QtWidgets.QPushButton('暂停')
        self.play_state = True

        # 布局
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.monitor_name)
        self.main_layout.addWidget(self.monitor_win)
        self.sub_layout = QtWidgets.QHBoxLayout()
        self.sub_layout.addWidget(self.play_button)
        self.sub_layout.addWidget(self.slider)
        self.main_layout.addLayout(self.sub_layout)
        self.setLayout(self.main_layout)

        self.play_button.clicked.connect(self.button_event)
        self.play_state_change_signal.connect(play_state_slot)
        self.slider.valueChanged.connect(self.slider_event)
        self.play_process_change_signal.connect(play_process_slot)

    def set_frame(self, frame):
        if type(frame) == str:
            self.monitor_win.setText(frame)
        else:
            h, w, ch = frame.shape
            tem_pixmap = QPixmap.fromImage(QImage(frame, w, h, ch * w, QImage.Format_RGB888))
            tem_pixmap.scaled(self.monitor_win.size())
            self.monitor_win.setPixmap(tem_pixmap)

    @Slot()
    def button_event(self):
        if self.play_state:
            self.play_button.setText('暂停')
        else:
            self.play_button.setText('播放')
        self.play_state = not self.play_state
        self.play_state_change_signal.emit(self.index)

    @Slot()
    def slider_event(self, value):
        self.play_process_change_signal.emit(self.index, value)


if __name__ == '__main__':
    app = QApplication()
    from Settings import Settings

    test = Settings()
    test.monitor_config_list = [MonitorConfig('test', 10) for _ in range(5)]
    test.each_monitor_rect = (100, 100)

    test = MonitoringInterface(test)
    test.show()
    app.exec_()