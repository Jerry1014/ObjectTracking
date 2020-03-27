from Model.Model import BaseModel
from Model.Gradnet.Import.track import just_show
from numpy import array


class TestForImport(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = just_show()

    def set_tracking_object(self, first_frame, tracking_object):
        # yx hw
        self.model.send((first_frame, (array(tracking_object[1::-1]), array(tracking_object[-1:-3:-1]))))

    def get_tracking_result(self, cur_frame):
        return self.model.send(cur_frame)
