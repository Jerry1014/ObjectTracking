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


class InterfaceController(QRunnable):
    def __init__(self, settings):
        """
        :param settings: 设置类
        """
        super().__init__()
        self.signal_connection = InterfaceSignalConnection()
        self.signal_connection.selected_filename = self.after_selected_file
        self.settings = settings
        self.frame_queue = self.settings.frame_queue

    def run(self):
        while not self.settings.filename:
            pass

        last_time = time()
        frame = self.frame_queue.get()
        self.emit_pic(frame)
        self.settings.first_frame = frame[0]
        self.settings.if_pause = True
        while not (self.settings.if_end and self.frame_queue.qsize() == 0):
            if not self.settings.if_pause:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue
                # 防止帧输出过快
                while time() - last_time < 0.04:
                    sleep(0.1)
                last_time = time()
                self.emit_pic(frame)

        self.emit_msg('视频结束')

    @Slot()
    def after_selected_file(self, filename: str):
        self.settings.filename = filename

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)
