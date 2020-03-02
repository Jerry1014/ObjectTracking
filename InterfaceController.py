"""
界面与模型控制之间的中间件
"""
from queue import Queue, Empty

from PySide2.QtCore import QObject, Signal, QRunnable, Slot
from cv2.cv2 import cvtColor, COLOR_BGR2RGB

from ReadVideo import ReadVideoFromFile, EndOfVideo


class InterfaceSignalConnection(QObject):
    msg_signal = Signal(str)
    selected_filename = Slot(str)
    pic_signal = Signal(list)
    signal_for_rect = Signal(list)


class InterfaceController(QRunnable):
    def __init__(self, settings, frame_queue: Queue):
        """
        :param settings: 设置字典
        :param frame_queue: 帧队列
        """
        super().__init__()
        self.signal_connection = InterfaceSignalConnection()
        self.signal_connection.selected_filename = self.after_selected_file
        self.settings = settings
        self.frame_queue = frame_queue
        self.first_frame_sign = True

    def run(self):
        while not self.settings.get('if_selected_file', None):
            pass

        while not self.settings['end_sign']:
            if not self.settings['pause_sign']:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue
                self.emit_pic(frame[0])
                if self.first_frame_sign:
                    self.settings['first_frame'] = frame[0]
                    self.first_frame_sign = False
                    self.settings['pause_sign'] = True
                else:
                    self.emit_rect(frame[1])

    @Slot()
    def after_selected_file(self, filename: str):
        self.settings['selected_filename'] = filename
        self.settings['if_selected_file'] = True

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)

    def emit_rect(self, rect):
        self.signal_connection.signal_for_rect.emit(tuple(rect))
