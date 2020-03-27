"""
在组件中传递的设置，也可是说是消息类，单例模式
"""
from queue import Queue

from numpy.core.multiarray import ndarray

from ReadVideo import ReadVideoFromFile


class Settings:
    def __init__(self):
        # 设置项
        self.init_fix_rect = (1000, 800)
        # fixme 后缀名大小写
        self.supported_formats = ('mp4', 'mkv','MP4')
        self.frame_queue_max_num = 24
        self.model_color_dict = None
        self.file_reader = ReadVideoFromFile()

        # 在组件中传递的标记
        self.if_end = None
        self.if_pause = True
        self.filename = None
        self.frame_queue = Queue(self.frame_queue_max_num)
        self.first_frame: ndarray = None
        self.tracking_object_rect = None

    def get_image_from_first_frame_by_rect(self, rect):
        rect = [int(i) for i in rect]
        # fixme 图片y轴存在偏移，暂通过临时加解决
        rect[1] = rect[1] + 38
        return self.first_frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].copy(order='C')

    def get_model_color(self, model_class):
        return self.model_color_dict.get(model_class, 'black')


settings = Settings()
