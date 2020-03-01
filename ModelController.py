"""
承接界面和模型的中间模块，负责总体调度
"""
from PySide2.QtCore import QRunnable

from ReadVideo import ReadVideo


class ModelController(QRunnable):
    def __init__(self, settings, frame_list):
        super().__init__()
        self.video_reader = ReadVideo()
        self.model = None
        self.settings = settings
        self.frame_list = frame_list

    def run(self):
        while not self.settings.get('if_selected_file', None):
            pass

        from MonkeyTest import TestCaseForReadVideo
        cap = TestCaseForReadVideo.ReadVideoFromFile()
        cap.open_video(self.settings['selected_filename'])
        while cap.is_open():
            try:
                frame = cap.get_one_frame()
                from cv2.cv2 import cvtColor
                from cv2.cv2 import COLOR_BGR2RGB
                image = cvtColor(frame, COLOR_BGR2RGB)
                self.frame_list.append(image)
                while len(self.frame_list) > 10:
                    pass
            except TestCaseForReadVideo.EndOfVideo:
                assert True
                break
        cap.release_init()
