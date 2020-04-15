"""
定义了使用到的各种各样的数据结构
"""


class MonitorConfig:
    def __init__(self, name, total_frame_num):
        """
        监控的设置
        :param name: 监控、视频名称
        :param total_frame_num: 总帧数
        """
        self.name = name
        self.total_frame_num = total_frame_num
