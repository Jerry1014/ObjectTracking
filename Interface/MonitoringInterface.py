"""
第一个界面，展示监控界面
"""
from time import sleep

from PySide2 import QtWidgets
from PySide2.QtCore import Slot
from PySide2.QtGui import QPixmap, QImage
from PySide2.QtWidgets import QApplication


class MonitoringInterface(QtWidgets.QWidget):
    def __init__(self, settings, external_signal=None):
        super().__init__()
        # 设置
        self.settings = settings
        self.setWindowTitle('监控界面')

        # 等待控制模块读取监控配置文件
        while self.settings.monitor_config_list is None:
            sleep(0.5)

        # 布局
        column_number = 1
        while column_number ** 2 < len(self.settings.monitor_config_list):
            column_number += 1
        self.monitor_list = list()
        self.layout = QtWidgets.QGridLayout()
        for i, config in enumerate(self.settings.monitor_config_list):
            # 是否需要传入更多的信息
            monitor = MonitoringSubInterface(config, self.settings.each_monitor_rect)
            self.monitor_list.append(monitor)
            self.layout.addWidget(monitor, i // column_number + 1, i % column_number + 1)
        self.setLayout(self.layout)

        # 信号与槽
        if external_signal:
            # todo
            pass

        # 其他
        self.sub_win = None

    @Slot(list)
    def set_frame(self, monitor_num: int, frame):
        self.monitor_list[monitor_num].set_frame(frame)


class MonitoringSubInterface(QtWidgets.QWidget):
    def __init__(self, monitor_config, monitor_rect):
        super().__init__()
        self.monitor_name = QtWidgets.QLabel(monitor_config['name'])
        self.monitor_win = QtWidgets.QLabel('frame')
        self.monitor_win.setFixedSize(*monitor_rect)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.monitor_name)
        self.layout.addWidget(self.monitor_win)
        self.setLayout(self.layout)

    def set_frame(self, frame):
        h, w, ch = frame.shape
        tem_pixmap = QPixmap.fromImage(QImage(frame, w, h, ch * w, QImage.Format_RGB888))
        tem_pixmap.scaled(self.monitor_win.size())
        self.image_win.setPixmap(tem_pixmap)


if __name__ == '__main__':
    app = QApplication()


    class test_class:
        def __init__(self):
            self.monitor_config_list = [{'name': 'test'} for _ in range(5)]
            self.each_monitor_rect = (100, 100)


    test = MonitoringInterface(test_class())
    test.show()
    app.exec_()
