"""
模型类，给定跟踪的对象和当前帧，输出跟踪结果
"""


class BaseModel:
    def __init__(self, tracking_object):
        self.tracking_object = tracking_object

    def get_tracking_result(self, cur_frame):
        raise NotImplementedError()
