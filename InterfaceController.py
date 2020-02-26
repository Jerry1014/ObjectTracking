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


class InterfaceController(QRunnable):
    def __init__(self, settings):
        super().__init__()
        self.signal_connection = InterfaceSignalConnection()
        self.signal_connection.selected_filename = self.after_selected_file
        self.settings = settings

    def run(self):
        while not self.settings.get('if_selected_file', None):
            pass

        # test
        cap = ReadVideoFromFile()
        cap.open_video('./resources/video/因为我穷.mp4')
        while cap.is_open():
            try:
                from numpy.core.multiarray import ndarray
                frame: ndarray = cap.get_one_frame()
                frame = cvtColor(frame, COLOR_BGR2RGB)
                self.emit_pic(frame)
            except EndOfVideo:
                break
        cap.release_init()

    @Slot()
    def after_selected_file(self, filename: str):
        self.settings['selected_filename'] = filename
        self.settings['if_selected_file'] = True

    def emit_msg(self, msg: str):
        self.signal_connection.msg_signal.emit(msg)

    def emit_pic(self, pic):
        self.signal_connection.pic_signal.emit(pic)
