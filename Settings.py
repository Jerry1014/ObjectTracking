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
        self.init_fix_rect = None
        self.supported_formats = ('mp4', 'mkv', 'avi', 'mpg')
        self.benckmart_list = [APE().get_iterator(), AOR().get_iterator()]
        self.frame_queue_max_num = 10
        self.model_color_dict = None
        self.file_reader = ReadVideoFromFile()

        # 在组件中传递的标记
        self.if_end = None
        self.if_pause = True
        self.if_have_gt = False
        self.filename = None
        self.total_frame_num = None
        self.cur_tracking_object_frame_num = 0
        self.frame_queue = Queue(maxsize=self.frame_queue_max_num)
        self.first_frame: ndarray = None
        self.tracking_object_rect = None

    def get_image_from_first_frame_by_rect(self, rect):
        rect = [int(i) for i in rect]
        return self.first_frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]].copy(order='C')

    def get_model_color(self, model_class):
        return self.model_color_dict.get(model_class, 'black')


settings = Settings()
"json:{}"
