# -*- coding:utf-8 -*-
"""
读取视频模块
"""


class OpenVideoError(Exception):
    pass


class EndOfVideo(Exception):
    pass


class ReadVideo:
    """
    指定视频资源，通过调用get_one_frame方法逐一获取每一帧
    """

    # CV2中的VideoCapture的初始化参数中的filename，既可以是视频文件名，也可以是图片序列，也可以是url
    # 故这个基类是否有存在的必要？我还没有想好，暂时这么写着
    def init(self, video_filename):
        """
        初始化视频读取
        :param video_filename: 视频文件名称
        """
        raise NotImplementedError()

    def get_one_frame(self):
        """
        获取一帧的图像
        """
        raise NotImplementedError()


class ReadVideoFromFile(ReadVideo):
    """
    从文件中逐帧读取
    """

    def __init__(self, video_filename=None):
        self.video_capture = None
        self.init(video_filename)

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
            raise EndOfVideo()

    def release_init(self):
        """
        释放当前视频容器，并重新初始化一个新的视频容器
        """
        self.video_capture.release()
        self.init()
