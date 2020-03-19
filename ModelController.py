"""
承接界面和模型的中间模块，负责总体调度
"""

from PySide2.QtCore import QRunnable
from cv2.cv2 import cvtColor, COLOR_BGR2RGB

from ReadVideo import ReadVideoFromFile, EndOfVideoError


class ModelController(QRunnable):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.video_reader = ReadVideoFromFile()
        self.model_class = self.settings.model_class
        self.model = self.model_class(None)
        self.frame_queue = self.settings.frame_queue

    def run(self):
        while not self.settings.filename:
            pass

        self.video_reader.open_video(self.settings.filename)
        frame = self.video_reader.get_one_frame()
        image = cvtColor(frame, COLOR_BGR2RGB)
        self.frame_queue.put((image, None))
        # todo 未做将需要追踪的模板传入模型类
        while self.video_reader.is_open():
            try:
                frame = self.video_reader.get_one_frame()
                image = cvtColor(frame, COLOR_BGR2RGB)
                rect = self.model.get_tracking_result(image)
                self.frame_queue.put((image, rect))
            except EndOfVideoError:
                break
        self.video_reader.release_init()
