"""
模型类，给定跟踪的对象和当前帧，输出跟踪结果
"""
from multiprocessing import Process, Event
from queue import Queue, Empty


class ModelBaseWithMultiProcess(Process):
    def __init__(self, input_queue: Queue, output_queue: Queue, rect_color, exit_event: Event):
        super().__init__()
        self.first_frame = None
        self.tracking_object_rect = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.color = rect_color
        self.exit_event = exit_event
        self._set_tracking_object()

    def _set_tracking_object(self):
        """
        设置跟踪对象
        """
        raise NotImplementedError()

    def get_tracking_result(self, cur_frame):
        """
        获取跟踪对象
        :param cur_frame: 当前帧
        :return: tuple 对象的x y w h
        """
        raise NotImplementedError()

    def run(self) -> None:
        while True:
            try:
                self.output_queue.put((self.get_tracking_result(self.input_queue.get(timeout=1)), self.color))
            except Empty:
                pass
            except Exception as e:
                print(e)
                return
            if self.exit_event.is_set():
                break
