"""
界面与模型控制之间的中间件
"""
from queue import Empty
from time import sleep

from PySide2.QtCore import QObject, Signal, QRunnable, Slot


class InterfaceSignalConnection(QObject):
    msg_signal = Signal(str)
    frame_num_changed = Slot(int)
    pic_signal = Signal(list)
    finish_one_frame_signal = Slot()
    model_ready_signal = Signal()
    total_frame_num = Signal(int)


class InterfaceController(QRunnable):
    def __init__(self, settings):
        """
        :param settings: 设置类
        """
        super().__init__()
        self.signal_connection = InterfaceSignalConnection()
        self.signal_connection.frame_num_changed = self.after_change_frame_num
        self.signal_connection.finish_one_frame_signal = self.after_finish_one_frame
        self.settings = settings
        self.frame_queue = self.settings.frame_queue
        self.interface_cur_frame_num = 0
        self.controller_cur_frame_num = 0

    def run(self):
        while not self.settings.filename:
            sleep(0.5)
        while self.settings.total_frame_num is None:
            sleep(0.5)
        self.signal_connection.total_frame_num.emit(self.settings.total_frame_num)

        # 第一帧
        self.settings.if_pause = True
        while self.settings.cur_tracking_object_frame_num >= 0:
            try:
                frame_and_result_rect = self.frame_queue.get(timeout=0.5)
                self.settings.first_frame = frame_and_result_rect[0][0]
                self.emit_pic(frame_and_result_rect)
            except Empty:
                pass

        # 当模型准备好后发送信号
        while self.frame_queue.qsize() == 0:
            sleep(0.5)
        self.signal_connection.model_ready_signal.emit()

        # 后续帧发送
        while not (self.settings.if_end and self.frame_queue.qsize() == 0):
            try:
                frame_and_result_rect = self.frame_queue.get(timeout=1)
            except Empty:
                continue
            while self.settings.if_pause:
                sleep(0.5)
            self.emit_pic(frame_and_result_rect)
            self.controller_cur_frame_num += 1
            while self.controller_cur_frame_num - self.interface_cur_frame_num > 1:
                sleep(0.05)

        try:
            self.emit_msg('视频结束')
        except RuntimeError:
            pass

    @Slot(int)
    def after_change_frame_num(self, frame_num: int):
        self.settings.cur_tracking_object_frame_num = frame_num

    @Slot()
    def after_finish_one_frame(self):
        self.interface_cur_frame_num += 1

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)
