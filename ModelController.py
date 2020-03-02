"""
承接界面和模型的中间模块，负责总体调度
"""
from queue import Queue

from PySide2.QtCore import QRunnable

from ReadVideo import ReadVideoFromFile, EndOfVideo
from cv2.cv2 import cvtColor, COLOR_BGR2RGB


class ModelController(QRunnable):
    def __init__(self, settings, frame_queue: Queue):
        super().__init__()
        self.video_reader = ReadVideoFromFile()
        self.model = TestModel()
        self.settings = settings
        self.frame_queue = frame_queue

    def run(self):
        while not self.settings.get('if_selected_file', None):
            pass

        self.video_reader.open_video(self.settings['selected_filename'])
        while self.video_reader.is_open():
            try:
                frame = self.video_reader.get_one_frame()
                image = cvtColor(frame, COLOR_BGR2RGB)
                rect = self.model.get_rect()
                self.frame_queue.put((image, rect))
            except EndOfVideo:
                assert True
                break
        self.video_reader.release_init()


class TestModel:
    def __init__(self):
        self.x = 0
        self.y = 0

    def get_rect(self):
        self.x += 1
        self.y += 1
        return self.x % 300, self.y % 300, 100, 100
