"""
承接界面和模型的中间模块，负责总体调度
"""
from configparser import ConfigParser
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
        self.model_list = list()

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
        last_emit_frame_time = time()
        while not self.settings.if_end:
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
                    # todo model_rect gt

                    while time() - last_emit_frame_time < 0.01:
                        sleep(0.01)
                    new_frame_config = FrameData(index, frame, None, None)
                    self.emit_frame_signal.emit(new_frame_config)
                    last_emit_frame_time = time()

    @Slot(list)
    def init_object_tracking_model(self, model_name_list):
        pass
