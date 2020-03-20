from Model.Model import BaseModel


class TestModel2(BaseModel):
    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0

    def get_tracking_result(self, cur_frame):
        self.x += 10
        self.y += 10
        return self.x % 300, self.y % 300, 100, 100
