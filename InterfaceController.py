"""
界面与模型控制之间的中间件
"""
from queue import Empty

from PySide2.QtCore import QObject, Signal, QRunnable, Slot


class InterfaceSignalConnection(QObject):
    msg_signal = Signal(str)
    selected_filename = Slot(str)
    pic_signal = Signal(list)
    signal_for_rect = Signal(list)


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

        frame = self.frame_queue.get()
        self.emit_pic(frame[0])
        self.settings.first_frame = frame[0]
        self.settings.if_pause = True
        while not self.settings.if_end:
            if not self.settings.if_pause:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue
                self.emit_pic(frame[0])
                self.emit_rect(frame[1])

    @Slot()
    def after_selected_file(self, filename: str):
        self.settings.filename = filename

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)

    def emit_rect(self, rect):
        self.signal_connection.signal_for_rect.emit(tuple(rect))
