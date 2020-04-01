"""
界面与模型控制之间的中间件
"""
from queue import Empty
from time import time, sleep

from PySide2.QtCore import QObject, Signal, QRunnable, Slot


class InterfaceSignalConnection(QObject):
    msg_signal = Signal(str)
    selected_filename = Slot(str)
    pic_signal = Signal(list)
    finish_one_frame_signal = Slot()


class InterfaceController(QRunnable):
    def __init__(self, settings):
        """
        :param settings: 设置类
        """
        super().__init__()
        self.signal_connection = InterfaceSignalConnection()
        self.signal_connection.selected_filename = self.after_selected_file
        self.signal_connection.finish_one_frame_signal = self.after_finish_one_frame
        self.settings = settings
        self.frame_queue = self.settings.frame_queue
        self.interface_cur_frame_num = 0
        self.controller_cur_frame_num = 0

    def run(self):
        while not self.settings.filename:
            pass
        frame = self.frame_queue.get()
        self.emit_pic(frame)
        self.settings.first_frame = frame[0]
        self.settings.if_pause = True
        while not (self.settings.if_end and self.frame_queue.qsize() == 0):
            """轮询管道内的视频帧"""
            if not self.settings.if_pause:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue
                self.emit_pic(frame)
                self.controller_cur_frame_num += 1
                while self.controller_cur_frame_num - self.interface_cur_frame_num > 1:
                    sleep(0.05)
            else:
                sleep(0.5)

        self.emit_msg('视频结束')

    @Slot()
    def after_selected_file(self, filename: str):
        self.settings.filename = filename

    @Slot()
    def after_finish_one_frame(self):
        self.interface_cur_frame_num += 1

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)
