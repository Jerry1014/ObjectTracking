"""
承接界面和模型的中间模块，负责总体调度
"""
import sys
from configparser import ConfigParser
from multiprocessing import Queue, Event
from os import getcwd
from os.path import sep, dirname
from time import sleep

from PySide2.QtCore import QRunnable

from GroundTrue.GroundTrueParser1 import GroundTrueParser1
from ReadVideo import EndOfVideoError, ReadVideoBase


class ModelController(QRunnable):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.video_reader: ReadVideoBase = self.settings.file_reader
        self.frame_queue = self.settings.frame_queue
        self.model_input_queue_list = list()
        self.model_output_queue_list = list()
        self.exit_event = Event()
        self.exit_event.clear()
        self.gt_list = None
        self.benckmart_list = None

    def run(self):
        while not self.settings.filename:
            sleep(0.5)

        # 准备ground true
        cf = ConfigParser()
        dataset_dir = dirname(self.settings.filename) + sep
        cf.read(dataset_dir + 'config.ini')
        try:
            gt_parser_section = cf[cf.sections()[0]]
            gt_filename = gt_parser_section['ground_true_filename']
            self.gt_list = GroundTrueParser1().get_result_list(dataset_dir + gt_filename) \
                if gt_parser_section['parser'] == '0' else None
            self.settings.if_have_gt = True
            # 初始化评价类
            self.benckmart_list = self.settings.benckmart_list
            for i in self.benckmart_list:
                next(i)
        except IndexError:
            pass

        # 读取第一帧
        self.video_reader.init(self.settings.filename)
        self.settings.total_frame_num = int(self.video_reader.get_frame_total_num())
        last_tracking_object_frame_num = -1
        cur_frame_num = 0
        while self.settings.cur_tracking_object_frame_num >= 0:
            cur_frame_num = self.settings.cur_tracking_object_frame_num
            if last_tracking_object_frame_num != self.settings.cur_tracking_object_frame_num:
                frame = self.video_reader.get_one_frame(self.settings.cur_tracking_object_frame_num)
                gt_rect = ((self.gt_list[cur_frame_num], 'green'),) if self.settings.if_have_gt else ()
                self.frame_queue.put((frame, gt_rect))
                last_tracking_object_frame_num = self.settings.cur_tracking_object_frame_num
            else:
                sleep(0.5)

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

        while True:
            try:
                frame = self.video_reader.get_one_frame()
                result_rect_list = list()
                # 将当前帧输入到模型输入队列
                for i in self.model_input_queue_list:
                    i.put(frame[0])
                # 取回模型结果
                for i in self.model_output_queue_list:
                    result_rect_list.append(i.get())
                # 取gt
                benckmart_list = None
                if self.settings.if_have_gt:
                    gt = self.gt_list[cur_frame_num]
                    cur_frame_num += 1
                    benckmart_list = tuple(i.send((j[0], gt)) for j in result_rect_list for i in self.benckmart_list)
                    result_rect_list.append((gt, 'green'))

                self.frame_queue.put((frame, result_rect_list, benckmart_list))
            except EndOfVideoError:
                self.settings.if_end = True
                self.exit_event.set()
                break
