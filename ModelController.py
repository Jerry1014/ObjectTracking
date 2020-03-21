"""
承接界面和模型的中间模块，负责总体调度
"""
import sys
from configparser import ConfigParser
from os import getcwd
from os.path import sep, isdir
from time import sleep

from PySide2.QtCore import QRunnable
from cv2.cv2 import cvtColor, COLOR_BGR2RGB

from ReadVideo import ReadVideoFromFile, EndOfVideoError, ReadPicFromDir


class ModelController(QRunnable):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.video_reader = None
        self.model_list = list()
        self.frame_queue = self.settings.frame_queue

    def run(self):
        while not self.settings.filename:
            pass

        # 读取第一帧
        # 判断过于简单，以后修正
        if isdir(self.settings.filename):
            self.video_reader = ReadPicFromDir()
        else:
            self.video_reader = ReadVideoFromFile()
        self.video_reader.init(self.settings.filename)
        frame = self.video_reader.get_one_frame()
        self.frame_queue.put((frame, list()))

        # 等待用户选择模型
        while self.settings.model_color_dict is None:
            pass

        cf = ConfigParser()
        cf.read('./Model/config.ini')
        for i in self.settings.model_color_dict.keys():
            path = cf[i]['path']
            if path not in sys.path:
                sys.path.append(getcwd() + sep + 'Model' + sep + path)
            try:
                m = __import__(i)
                model = getattr(m, i)()
                model.set_tracking_object(self.settings.tracking_object)
                self.model_list.append(model)
            except (ModuleNotFoundError, AttributeError):
                print(f'反射失败 反射模块{i} 模块路径{path} 反射类{i}')

        # 等待用户选择开始
        while self.settings.if_pause:
            pass

        while self.video_reader.is_open():
            try:
                frame = self.video_reader.get_one_frame()
                rect_list = list()
                for i in self.model_list:
                    rect_list.append(
                        (i.get_tracking_result(frame), self.settings.get_model_color(i.__class__.__name__)))
                self.frame_queue.put((frame, rect_list))
                while self.frame_queue.qsize() > 10:
                    sleep(0.1)
            except EndOfVideoError:
                break
        self.settings.if_end = True
        self.video_reader.release_init()
