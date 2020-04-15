# -*- coding:utf-8 -*-
"""
读取视频模块
"""
from os import walk
from os.path import exists, sep
from queue import Queue

from cv2.cv2 import imread, cvtColor, COLOR_BGR2RGB, VideoCapture, CAP_PROP_POS_FRAMES, CAP_PROP_FRAME_COUNT, \
    CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH


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
    def init(self, file_or_dir):
        """"
        初始化
        """
        raise NotImplementedError()

    def get_one_frame(self, frame_num=-1):
        """
        获取一帧的图像
        :param frame_num: -1 读取下一帧 >=0 读取frame_num帧
        """
        raise NotImplementedError()

    def get_frame_total_num(self):
        """
        获取总帧数
        """
        raise NotImplementedError()

    def get_frame_shape(self):
        """
        获取帧的h w
        """
        raise NotImplementedError()


class ReadVideoFromFile(ReadVideoBase):
    """
    从文件中逐帧读取
    """

    def get_frame_shape(self):
        if self.video_capture and self.video_capture.isOpened():
            return self.video_capture.get(CAP_PROP_FRAME_WIDTH), self.video_capture.get(CAP_PROP_FRAME_HEIGHT)

    def __init__(self):
        self.video_capture: VideoCapture = None

    def init(self, video_filename):
        """
        视频容器初始化
        :param video_filename: 视频文件名称。可为空，后续通过open_video方法打开视频文件
        """
        self.video_capture = VideoCapture(video_filename)

    def is_open(self):
        """
        判断视频容器是否已经打开了视频文件
        :return: True 已打开视频文件
        """
        return self.video_capture.isOpened()

    def get_one_frame(self, frame_num=-1):
        """
        获取视频的一帧
        :return:视频帧
        :raise: EndOfVideo(Exception)视频结束
        """
        if self.is_open():
            if frame_num > 0:
                self.video_capture.set(CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.video_capture.read()
            if ret:
                return cvtColor(frame, COLOR_BGR2RGB), self.video_capture.get(CAP_PROP_POS_FRAMES) - 1
            else:
                # 此处释放逻辑对于视频可行，对于摄像头则有错
                self.release_init()
                raise EndOfVideoError()
        else:
            raise EndOfVideoError()

    def release_init(self):
        """
        释放当前视频容器
        """
        self.video_capture.release()

    def get_frame_total_num(self):
        if self.video_capture:
            return self.video_capture.get(CAP_PROP_FRAME_COUNT)


class ReadPicFromDir(ReadVideoBase):
    support_format = ('jpg',)

    def get_frame_shape(self):
        if self.pic_list:
            h, w, ch = self.pic_list[0].shape
            return w, h

    def __init__(self):
        self.pic_list = list()
        self.cur_index = -1

    def init(self, file_or_dir):
        if not exists(file_or_dir):
            raise OpenVideoError()

        _, _, file_list = next(walk(file_or_dir))
        for filename in file_list:
            if filename.split('.')[-1] in self.support_format:
                self.pic_list.append(imread(file_or_dir + sep + filename))

    def get_one_frame(self, frame_num=-1):
        if frame_num < 0:
            if self.cur_index < len(self.pic_list):
                self.cur_index += 1
                return self.pic_list[self.cur_index]
            else:
                raise EndOfVideoError()
        else:
            return self.pic_list[frame_num] if frame_num < len(self.pic_list) else None

    def get_frame_total_num(self):
        return len(self.pic_list)
