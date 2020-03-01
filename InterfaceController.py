"""
界面与模型控制之间的中间件
"""

from PySide2.QtCore import QObject, Signal, QRunnable, Slot
from cv2.cv2 import cvtColor, COLOR_BGR2RGB

from ReadVideo import ReadVideoFromFile, EndOfVideo


class InterfaceSignalConnection(QObject):
    msg_signal = Signal(str)
    selected_filename = Slot(str)
    pic_signal = Signal(list)
    signal_for_rect = Signal(list)


class InterfaceController(QRunnable):
    def __init__(self, settings, frame_list):
        super().__init__()
        self.signal_connection = InterfaceSignalConnection()
        self.signal_connection.selected_filename = self.after_selected_file
        self.settings = settings
        self.frame_list = frame_list
        self.first_frame_sign = True

    def run(self):
        while not self.settings.get('if_selected_file', None):
            pass

        while not self.settings['end_sign']:
            print(self.settings['pause_sign'])
            if not self.settings['pause_sign'] and self.frame_list:
                # 此处存在简略，忽略帧的顺序问题，同时没有错误提示
                frame = self.frame_list[0]
                self.frame_list.remove(frame)
                self.emit_pic(frame)
                if self.first_frame_sign:
                    self.settings['first_frame'] = frame
                    self.first_frame_sign = False
                    self.settings['pause_sign'] = True

    @Slot()
    def after_selected_file(self, filename: str):
        self.settings['selected_filename'] = filename
        self.settings['if_selected_file'] = True

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)
