from Model.Model import ModelBaseWithMultiProcess


class TestModel(ModelBaseWithMultiProcess):
    def _set_tracking_object(self):
        tracking_object = self.input_queue.get()
        self.first_frame = tracking_object[0]
        self.tracking_object_rect = tracking_object[1]
        self.model_name = 'Test'

    def __init__(self, input_queue, output_queue, rect_color, exit_event):
        super().__init__(input_queue, output_queue, rect_color, exit_event)
        self.x = 0
        self.y = 0

    def get_tracking_result(self, cur_frame):
        self.x += 1
        self.y += 1
        return ((self.x % 300, self.y % 300, 100, 100), self.color), (self.model_name, None)
