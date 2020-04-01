from queue import Empty

from Model.Gradnet.Import.track import just_show
from numpy import array


class TestForImport():
    def __init__(self, input_queue, output_queue, rect_color, exit_event):
        self.model = just_show()
        self.first_frame = None
        self.tracking_object_rect = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.color = rect_color
        self.exit_event = exit_event
        self._set_tracking_object()

    def _set_tracking_object(self):
        # yx hw
        tracking_object = self.input_queue.get()
        self.first_frame = tracking_object[0]
        self.tracking_object_rect = tracking_object[1]
        self.model.send(
            (self.first_frame, (array(self.tracking_object_rect[1::-1]), array(self.tracking_object_rect[-1:-3:-1]))))

    def get_tracking_result(self, cur_frame):
        return self.model.send(cur_frame)

    def start(self) -> None:
        while True:
            try:
                self.output_queue.put((self.get_tracking_result(self.input_queue.get(timeout=1)), self.color))
            except Empty:
                pass
            if self.exit_event.is_set():
                break
