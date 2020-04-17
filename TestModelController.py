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
from ReadVideo import ReadVideoFromFile, EndOfVideoError


class TestModelController(QRunnable):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.video_reader_list = list()
        self.emit_frame_signal = None
        self.if_model = False
        self.model_input_queue_list = list()
        self.model_output_queue_list = list()
        self.exit_event = Event()

    def run(self):
        # 载入视频与gt
        video_config = ConfigParser()
        video_config.read('./Resources/MonitoringConfig.ini')
        monitor_config_list = list()

        max_video_width = 0
        max_video_height = 0
        if video_config:
            for i in video_config.sections():
                tem_config = video_config[i]
                tem_video_reader = ReadVideoFromFile()
                tem_video_reader.init(sep.join(['.', 'Resources', 'video'] + tem_config['path'].split()))
                tem_w, tem_h = tem_video_reader.get_frame_shape()
                if tem_w > max_video_width:
                    max_video_width = tem_w
                if tem_h > max_video_height:
                    max_video_height = tem_h
                tem_monitor_config = MonitorConfig(tem_config['name'], tem_video_reader.get_frame_total_num())
                self.video_reader_list.append(tem_video_reader)
                monitor_config_list.append(tem_monitor_config)

                # todo gt
        else:
            print('监控初始化失败，请检查配置文件')
            exit()
        self.settings.monitor_config_list = monitor_config_list
        self.settings.each_monitor_rect = (max_video_width, max_video_height)

        # 新线程启动界面
        while self.settings.frame_update_signal is None:
            sleep(0.5)
        self.emit_frame_signal = self.settings.frame_update_signal

        # 帧发送
        monitor_num = len(self.settings.monitor_config_list)
        last_emit_frame_time = [time() for _ in range(monitor_num + 1)]
        while not self.settings.if_end:
            while time() - last_emit_frame_time[-1] < 0.02 * monitor_num:
                sleep(0.02)
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
                    if self.if_model:
                        # 将当前帧输入到模型输入队列
                        for i in self.model_input_queue_list:
                            i.put(frame[0])
                        # 取回模型结果
                        for i in self.model_output_queue_list:
                            result_rect_list.append(i.get())
                    # todo model_rect

                    while time() - last_emit_frame_time[index] < 0.03:
                        sleep(0.01)
                    new_frame_config = FrameData(index, frame, result_rect_list, None)
                    self.emit_frame_signal.emit(new_frame_config)
                    last_emit_frame_time[index] = time()
            last_emit_frame_time[-1] = time()
        self.exit_event.set()

    @Slot(dict)
    def init_object_tracking_model(self, model_name_list):
        cf = ConfigParser()
        cf.read('./Model/ModelConfig.ini')
        for i in model_name_list.keys():
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
