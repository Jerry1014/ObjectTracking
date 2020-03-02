"""
承接界面和模型的中间模块，负责总体调度
"""
from queue import Queue

from PySide2.QtCore import QRunnable

from Model import BaseModel
from ReadVideo import ReadVideoFromFile, EndOfVideo
from cv2.cv2 import cvtColor, COLOR_BGR2RGB


class ModelController(QRunnable):
    def __init__(self, settings, frame_queue: Queue, model_class):
        super().__init__()
        self.video_reader = ReadVideoFromFile()
        self.model_class = model_class
        self.model = model_class(None)
        self.settings = settings
        self.frame_queue = frame_queue

    def run(self):
        while not self.settings.get('if_selected_file', None):
            pass

        self.video_reader.open_video(self.settings['selected_filename'])
        # todo 未做将需要追踪的模板传入模型类
        while self.video_reader.is_open():
            try:
                frame = self.video_reader.get_one_frame()
                image = cvtColor(frame, COLOR_BGR2RGB)
                rect = self.model.get_tracking_result(image)
                self.frame_queue.put((image, rect))
            except EndOfVideo:
                break
        self.video_reader.release_init()
