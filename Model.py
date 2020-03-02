"""
模型类，给定跟踪的对象和当前帧，输出跟踪结果
"""


class BaseModel:
    def __init__(self, tracking_object):
        self.tracking_object = tracking_object

    def get_tracking_result(self, cur_frame):
        raise NotImplementedError()


class TestModel(BaseModel):
    def __init__(self, tracking_object):
        super().__init__(tracking_object)
        self.x = 0
        self.y = 0

    def get_tracking_result(self, cur_frame):
        self.x += 1
        self.y += 1
        return self.x % 300, self.y % 300, 100, 100
