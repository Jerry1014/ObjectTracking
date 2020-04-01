from Model.Model import BaseModel
from Model.Gradnet.Import.track import just_show
from numpy import array


class TestForImport(BaseModel):
    def __init__(self, input_queue, output_queue, rect_color, exit_event):
        self.model = just_show()
        super().__init__(input_queue, output_queue, rect_color, exit_event)

    def _set_tracking_object(self):
        # yx hw
        super()._set_tracking_object()
        self.model.send(
            (self.first_frame, (array(self.tracking_object_rect[1::-1]), array(self.tracking_object_rect[-1:-3:-1]))))

    def get_tracking_result(self, cur_frame):
        return self.model.send(cur_frame)
