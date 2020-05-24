"""
承接界面和模型的中间模块，负责总体调度
"""
import sys
from configparser import ConfigParser
from multiprocessing import Event, Queue
from os import getcwd
from os.path import sep
from time import sleep, time

from PySide2.QtCore import Slot, QRunnable

from DataStructure import MonitorConfig, FrameData
from GroundTrue.GroundTrueParser1 import GroundTrueParser1
from ReadVideo import ReadVideoFromFile, EndOfVideoError


class ModelController(QRunnable):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.video_reader_list = list()
        self.emit_frame_signal = None
        self.if_model = False
        self.model_input_queue_list = list()
        self.model_output_queue_list = list()
        self.video_gt_list = list()
        self.exit_event = Event()

        self.benckmark_list = self.settings.benckmart_list
        for i in self.benckmark_list:
            next(i[1])

    def run(self):
        # 载入视频与gt
        video_config = ConfigParser()
        video_config.read('./Resources/MonitoringConfig.ini')
        monitor_config_list = list()

        min_video_width = 1000
        min_video_height = 1000
        if video_config:
            for i in video_config.sections():
                tem_config = video_config[i]
                tem_video_reader = ReadVideoFromFile()
                tem_path_list = tem_config['path'].split()
                tem_path_dir = ['.', 'Resources', 'video'] + tem_path_list[:-1]
                tem_video_reader.init(sep.join(tem_path_dir + [tem_path_list[-1]]))
                tem_w, tem_h = tem_video_reader.get_frame_shape()
                if tem_w < min_video_width:
                    min_video_width = tem_w
                if tem_h < min_video_height:
                    min_video_height = tem_h
                tem_monitor_config = MonitorConfig(tem_config['name'], tem_video_reader.get_frame_total_num())
                self.video_reader_list.append(tem_video_reader)
                monitor_config_list.append(tem_monitor_config)

                video_gt_config = ConfigParser()
                video_gt_config.read(sep.join(tem_path_dir + ['GTConfig.ini']))
                tem_gt = None
                try:
                    gt_parser_section = video_gt_config[video_gt_config.sections()[0]]
                    gt_filename = gt_parser_section['ground_true_filename']
                    tem_gt = GroundTrueParser1().get_result_list(sep.join(tem_path_dir + [gt_filename])) \
                        if gt_parser_section['parser'] == '0' else None
                except IndexError:
                    pass
                self.video_gt_list.append(tem_gt)
        else:
            print('监控初始化失败，请检查配置文件')
            exit()
        self.settings.monitor_config_list = monitor_config_list
        self.settings.each_monitor_rect = (min_video_width, min_video_height)

        # 新线程启动界面
        while self.settings.frame_update_signal is None:
            sleep(0.5)
            if self.settings.if_end:
                return
        self.emit_frame_signal = self.settings.frame_update_signal

        # 帧发送
        monitor_num = len(self.settings.monitor_config_list)
        last_emit_frame_time = [time() for _ in range(monitor_num + 1)]
        while not self.settings.if_end:
            while time() - last_emit_frame_time[-1] < 0.03 * monitor_num:
                sleep(0.01)
            for index, sign in enumerate(self.settings.monitor_play_state):
                if sign:
                    try:
                        if type(sign) == bool:
                            frame = self.video_reader_list[index].get_one_frame()
                        else:
                            frame = self.video_reader_list[index].get_one_frame(sign)
                            self.settings.monitor_play_state[index] = False
                    except EndOfVideoError:
                        frame = ('视频已结束', None)
                    result_rect_list = list()
                    score_map_list = list()
                    benckmark_list = None
                    if self.settings.if_tracking:
                        # 取模型结果
                        if self.if_model:
                            # 将当前帧输入到模型输入队列
                            for i in self.model_input_queue_list:
                                i.put(frame[0])
                            # 取回模型结果
                            for i in self.model_output_queue_list:
                                tem_model_result = i.get()
                                result_rect_list.append(tem_model_result[0])
                                score_map_list.append((tem_model_result[1]))

                        if self.video_gt_list[index] and frame[1]:
                            gt = self.video_gt_list[index][frame[1]]
                            if self.if_model:
                                benckmark_list = tuple(
                                    (i[0],) + tuple((i[1].send((gt, j[0])), j[1]) for j in result_rect_list) for i in
                                    self.benckmark_list)
                            # 最后加入gt，防止在模型评价中计算gt自身
                            result_rect_list.append((gt, 'green'))

                    else:
                        self.exit_event.set()
                        self.if_model = False

                    while time() - last_emit_frame_time[index] < 0.03:
                        sleep(0.01)
                    new_frame_config = FrameData(index, frame, result_rect_list, benckmark_list, score_map_list)
                    try:
                        self.emit_frame_signal.emit(new_frame_config)
                    except RuntimeError:
                        # 一般是由于界面退出导致
                        print('模型控制类线程已退出')
                    last_emit_frame_time[index] = time()
            last_emit_frame_time[-1] = time()
        self.exit_event.set()

    @Slot(dict)
    def init_object_tracking_model(self, model_name_list):
        self.exit_event.clear()
        self.model_output_queue_list = list()
        self.model_input_queue_list = list()

        cf = ConfigParser()
        cf.read('./Model/ModelConfig.ini')
        for i in model_name_list.keys():
            # 当前模型控制类使用多线程，为了给界面进程一点时间，避免无响应
            sleep(0.1)

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
                model_color = model_name_list[model_class.__name__]
                model = model_class(tem_input_queue, tem_output_queue, model_color, self.exit_event)
                model.start()
            except (ModuleNotFoundError, AttributeError) as e:
                print(f'反射失败 反射模块{i} 模块路径{path} 反射类{i} 失败原因{e}')
        self.if_model = True

