"""
承接界面和模型的中间模块，负责总体调度
"""
import sys
from configparser import ConfigParser
from multiprocessing import Queue, Event
from os import getcwd
from os.path import sep
from time import sleep

from PySide2.QtCore import QRunnable

from ReadVideo import EndOfVideoError


class ModelController(QRunnable):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.video_reader = self.settings.file_reader
        self.frame_queue = self.settings.frame_queue
        self.model_input_queue_list = list()
        self.model_output_queue_list = list()
        self.exit_event = Event()
        self.exit_event.clear()

    def run(self):
        while not self.settings.filename:
            pass

        # 读取第一帧
        self.video_reader.init(self.settings.filename)
        frame = self.video_reader.get_one_frame()
        self.frame_queue.put((frame, list(), 0))

        # 等待用户选择模型
        while self.settings.model_color_dict is None:
            sleep(0.5)

        cf = ConfigParser()
        cf.read('./Model/config.ini')
        for i in self.settings.model_color_dict.keys():
            path = cf[i]['path']
            if path not in sys.path:
                path = sep.join([getcwd(), 'Model'] + path.split(' '))
                sys.path.append(path)
            try:
                m = __import__(i)
                tem_input_queue = Queue()
                tem_output_queue = Queue()
                self.model_input_queue_list.append(tem_input_queue)
                self.model_output_queue_list.append(tem_output_queue)
                tem_input_queue.put((self.settings.first_frame, self.settings.tracking_object_rect))

                model_class = getattr(m, i)
                model_color = self.settings.get_model_color(model_class.__name__)
                model = model_class(tem_input_queue, tem_output_queue, model_color, self.exit_event)
                model.start()
            except (ModuleNotFoundError, AttributeError) as e:
                print(f'反射失败 反射模块{i} 模块路径{path} 反射类{i} 失败原因{e}')

        # 等待用户选择开始
        while self.settings.if_pause:
            sleep(0.5)

        test = 0
        while True:
            try:
                frame = self.video_reader.get_one_frame()
                result_rect_list = list()
                # 将当前帧输入到模型输入队列
                for i in self.model_input_queue_list:
                    i.put(frame)
                # 取回模型结果
                for i in self.model_output_queue_list:
                    result_rect_list.append(i.get())
                test += 1
                self.frame_queue.put((frame, result_rect_list, test))
            except EndOfVideoError:
                self.settings.if_end = True
                self.exit_event.set()
                break
