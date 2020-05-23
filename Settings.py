"""
在组件中传递的设置，也可是说是消息类，单例模式
未来将修改为自动处理管道中的信息，以实现管道两边同步
"""
from queue import Queue

from numpy.core.multiarray import ndarray

from Benckmark.AOR import AOR
from Benckmark.APE import APE
from ReadVideo import ReadVideoFromFile, ReadPicFromDir


class Settings:
    def __init__(self):
        # 设置项
        self.benckmart_list = [('APE', APE().get_iterator()), ('AOR', AOR().get_iterator())]
        self.frame_queue_max_num = 10
        self.model_color_dict = None

        # 在组件中传递的标记
        self.total_frame_num = None
        self.if_end = False
        self.first_frame: ndarray = None
        self.last_frame: ndarray = None
        self.tracking_object_rect = None
        self.monitor_config_list = None
        self.monitor_play_state = list()
        self.each_monitor_rect = (500, 500)
        self.frame_update_signal = None
        self.if_tracking = False

    def get_image_from_first_frame_by_rect(self, rect):
        rect = [int(i) for i in rect]
        return self.first_frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].copy(order='C')

    def get_image_from_last_frame_by_rect(self):
        return self.last_frame[self.tracking_object_rect[1]:self.tracking_object_rect[1] + self.tracking_object_rect[3],
               self.tracking_object_rect[0]:self.tracking_object_rect[0] + self.tracking_object_rect[2]].copy(order='C')

    def get_model_color(self, model_class):
        return self.model_color_dict.get(model_class, 'black')


settings = Settings()
