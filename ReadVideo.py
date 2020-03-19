# -*- coding:utf-8 -*-
"""
读取视频模块
"""
from os import walk
from os.path import exists, sep
from queue import Queue

from cv2.cv2 import imread


class OpenVideoError(Exception):
    pass


class EndOfVideoError(Exception):
    pass


class ReadVideoBase:
    """
    指定视频资源，通过调用get_one_frame方法逐一获取每一帧
    """

    # CV2中的VideoCapture的初始化参数中的filename，既可以是视频文件名，也可以是图片序列，也可以是url
    # 故这个基类是否有存在的必要？我还没有想好，暂时这么写着
    def __init__(self, file_or_dir):
        self.file_or_dir = file_or_dir

    def get_one_frame(self):
        """
        获取一帧的图像
        """
        raise NotImplementedError()


class ReadVideoFromFile(ReadVideoBase):
    """
    从文件中逐帧读取
    """

    def __init__(self, file_or_dir=None):
        super().__init__(file_or_dir)
        self.video_capture = None
        self.init(file_or_dir)

    def init(self, video_filename=None):
        """
        视频容器初始化
        :param video_filename: 视频文件名称。可为空，后续通过open_video方法打开视频文件
        """
        from cv2.cv2 import VideoCapture
        self.video_capture = VideoCapture(video_filename) if video_filename else VideoCapture()

    def open_video(self, video_filename):
        """
        打开视频文件
        :param video_filename: 视频文件名称
        :raise: OpenVideoError(Exception)文件打开失败
        """
        if video_filename and not self.video_capture.open(video_filename):
            raise OpenVideoError()

    def is_open(self):
        """
        判断视频容器是否已经打开了视频文件
        :return: True 已打开视频文件
        """
        return self.video_capture.isOpened()

    def get_one_frame(self):
        """
        获取视频的一帧
        :return:视频帧
        :raise: EndOfVideo(Exception)视频结束
        """
        ret, frame = self.video_capture.read()
        if ret:
            return frame
        else:
            # 此处释放逻辑对于视频可行，对于摄像头则有错
            self.release_init()
            raise EndOfVideoError()

    def release_init(self):
        """
        释放当前视频容器，并重新初始化一个新的视频容器
        """
        self.video_capture.release()
        self.init()


class ReadPicFromDir(ReadVideoBase):
    support_format = ('jpg',)

    def __init__(self, file_or_dir):
        super().__init__(file_or_dir)

        if not exists(self.file_or_dir):
            raise OpenVideoError()

        self.pic_queue = Queue()
        _, _, file_list = next(walk(self.file_or_dir))
        for filename in file_list:
            if filename.split('.')[-1] in self.support_format:
                self.pic_queue.put(imread(self.file_or_dir + sep + filename))

    def get_one_frame(self):
        if self.pic_queue.qsize() > 0:
            return self.pic_queue.get()
        else:
            raise EndOfVideoError()
