"""
第一个界面，展示监控界面
"""
from time import sleep

from PySide2 import QtWidgets
from PySide2.QtCore import Slot, Qt, Signal
from PySide2.QtGui import QPixmap, QImage, QWheelEvent, QMouseEvent

from DataStructure import FrameData
from Interface.TrackingInterface import TrackingWin
from User.User import User, UserException


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
        self.user = User()

        # 等待控制模块读取监控配置文件
        while self.settings.monitor_config_list is None:
            sleep(0.5)
        for _ in range(len(self.settings.monitor_config_list)):
            self.play_state.append(True)

        # 布局
        self.layout = QtWidgets.QGridLayout()
        self.user_interface = UserInterface(self.user)
        self.layout.addWidget(self.user_interface, 1, 1)

        column_number = 1
        while column_number ** 2 < len(self.settings.monitor_config_list):
            column_number += 1
        self.monitor_list = list()
        for i, config in enumerate(self.settings.monitor_config_list):
            # 是否需要传入更多的信息
            monitor = MonitoringSubInterface(i, config, self.settings.each_monitor_rect, self.change_play_state,
                                             self.change_play_process, self.start_tracking)
            self.monitor_list.append(monitor)
            self.layout.addWidget(monitor, (i + 1) // column_number + 1, (i + 1) % column_number + 1)
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
    def start_tracking(self, tracking_msg):
        index, last_frame, slider_value, slider_max_num = tracking_msg
        for enu_index, state in enumerate(self.play_state):
            if state:
                self.monitor_list[enu_index].button_event()
        self.sub_win = TrackingWin(index, self.settings, self.model_init_signal, self.change_play_process,
                                   self.after_close_tracking, self.change_play_state, slider_max_num,
                                   self.start_tracking)
        self.change_play_process(index, slider_value)
        self.sub_win.show()
        self.sub_win.activateWindow()
        self.tem_index_monitor = (index, self.monitor_list[index])
        self.monitor_list[index] = self.sub_win

    @Slot()
    def after_close_tracking(self):
        self.settings.if_tracking = False
        index, monitor = self.tem_index_monitor
        self.monitor_list[index] = monitor
        for index, state in enumerate(self.play_state):
            if not state:
                self.monitor_list[index].button_event()
        self.sub_win = None


class MonitoringSubInterface(QtWidgets.QWidget):
    play_state_change_signal = Signal(int)
    play_process_change_signal = Signal(int, int)
    start_tracking_signal = Signal(tuple)

    def __init__(self, index, monitor_config, monitor_rect, play_state_slot, play_process_slot, start_tracking_slot):
        super().__init__()
        self.index = index
        self.last_frame = None
        # 部件
        self.monitor_name = QtWidgets.QLabel(monitor_config.name)
        self.monitor_win = MonitoringSubInterfaceLabel(monitor_rect)
        # self.monitor_win = QtWidgets.QLabel()
        # self.monitor_win.setFixedSize(*monitor_rect)
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
        frame, cur_frame_num = frame_data.frame
        if type(frame) == str:
            self.monitor_win.setText(frame)
        else:
            self.last_frame = frame
            self.monitor_win.set_image(frame)
            # h, w, ch = frame.shape
            # tem_pixmap = QPixmap.fromImage(QImage(frame, w, h, ch * w, QImage.Format_RGB888))
            # tem_pixmap.scaled(self.monitor_win.size())
            # self.monitor_win.setPixmap(tem_pixmap)
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
        self.start_tracking_signal.emit((self.index, self.last_frame, self.slider.value(), self.slider_max_num))


class MonitoringSubInterfaceLabel(QtWidgets.QLabel):
    # scale_value越大缩放越慢 mouse_value越大，鼠标拖动越快
    scale_value = 5000
    mouse_value = 0.05
    step_each_angle = None
    min_rect = (100, 100)

    def __init__(self, max_rect):
        super().__init__()
        self.step_each_angle = max(max_rect) / self.scale_value
        self.max_rect = max_rect
        self.setFixedSize(*max_rect)
        self.cur_pos_rect = [0, 0, *max_rect]
        self.last_frame = None
        self.start_pos = None

    def wheelEvent(self, event: QWheelEvent):
        step_w = -event.angleDelta().y() * self.step_each_angle
        mouse_pos = event.position().toTuple()
        step_x = step_w * mouse_pos[0] / self.max_rect[0]
        step_h = step_w / self.max_rect[0] * self.max_rect[1]
        step_y = step_h * mouse_pos[1] / self.max_rect[1]
        tem_x = self.cur_pos_rect[0] - step_x
        tem_y = self.cur_pos_rect[1] - step_y
        tem_w = self.cur_pos_rect[2] + step_w
        tem_h = self.cur_pos_rect[3] + step_h
        if step_w > 0:
            tem_x = max(tem_x, 0)
            tem_y = max(tem_y, 0)
            tem_w = min(tem_w, self.max_rect[0])
            tem_h = min(tem_h, self.max_rect[1])
        else:
            tem_x = min(tem_x, self.max_rect[0] - self.min_rect[0])
            tem_y = min(tem_y, self.max_rect[1] - self.min_rect[1])
            tem_w = max(tem_w, self.min_rect[0])
            tem_h = max(tem_h, self.min_rect[1])

        self.cur_pos_rect = [tem_x, tem_y, tem_w, tem_h]
        if self.last_frame is not None:
            self.set_image(self.last_frame)

    def mousePressEvent(self, ev: QMouseEvent):
        self.start_pos = ev.localPos().toTuple()

    def mouseMoveEvent(self, ev: QMouseEvent):
        end_pos = ev.localPos().toTuple()
        step_x, step_y = [end - start for end, start in zip(end_pos, self.start_pos)]
        scale_num = self.mouse_value * self.cur_pos_rect[2] / self.max_rect[0]
        new_x = self.cur_pos_rect[0] - step_x * scale_num
        new_y = self.cur_pos_rect[1] - step_y * scale_num
        if new_x > 0:
            self.cur_pos_rect[0] = min(new_x, self.max_rect[0] - self.cur_pos_rect[2])
        else:
            self.cur_pos_rect[0] = max(new_x, 0)
        if new_y > 0:
            self.cur_pos_rect[1] = min(new_y, self.max_rect[1] - self.cur_pos_rect[3])
        else:
            self.cur_pos_rect[1] = max(new_y, 0)
        if self.last_frame is not None:
            self.set_image(self.last_frame)

    def set_image(self, image):
        self.last_frame = image
        rect = [int(i) for i in self.cur_pos_rect]
        # rect = [0, 0, 500, 500]
        new_image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].copy(order='C')
        h, w, ch = new_image.shape
        tem_pixmap = QPixmap.fromImage(QImage(new_image, w, h, ch * w, QImage.Format_RGB888))
        if tem_pixmap.size().toTuple() != self.size().toTuple():
            tem_pixmap = tem_pixmap.scaled(self.size())
        self.setPixmap(tem_pixmap)


class UserInterface(QtWidgets.QWidget):
    def __init__(self, user):
        super().__init__()
        self.user = user
        if self.user.user_psw:
            self.user_dialog = UserDialog(self.user)
        else:
            self.user_dialog = UserDialog(self.user, '注册')
        self.user_dialog.show()
        self.user_dialog.exec_()

        # 部件
        self.user_name_label = QtWidgets.QLabel('用户: '+self.user.user_name)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.user_name_label)
        self.setLayout(self.layout)


class UserDialog(QtWidgets.QDialog):
    def __init__(self, user, method='登录', user_name=None):
        super().__init__(None)
        self.setWindowTitle(method)
        self.user = user
        self.method = method

        self.user_name_edit = QtWidgets.QLineEdit()
        if method == '修改密码':
            self.user_name_edit.setText(user_name)
            self.user_name_edit.setEnabled(False)
        else:
            self.user_name_edit.setPlaceholderText('请输入用户名')
        self.psw_edit = QtWidgets.QLineEdit()
        self.psw_edit.setPlaceholderText('请输入密码')
        self.psw_edit.setEchoMode(QtWidgets.QLineEdit.PasswordEchoOnEdit)

        self.button = QtWidgets.QPushButton(method)
        self.button.clicked.connect(self.button_event)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.user_name_edit)
        self.layout.addWidget(self.psw_edit)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def button_event(self):
        if self.method == '注册':
            self.user.sign_in(self.user_name_edit.text(), self.psw_edit.text())
        elif self.method == '修改密码':
            self.user.change_psw(self.psw_edit.text())
        else:
            try:
                self.user.login(self.user_name_edit.text(), self.psw_edit.text())
            except UserException as e:
                msg_box = QtWidgets.QMessageBox()
                msg_box.setWindowTitle('错误')
                msg_box.setText(str(e))
                msg_box.exec_()
                return
        self.close()
