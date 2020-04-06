"""
模型来源
@InProceedings{GradNet_ICCV2019,
author = {Peixia Li, Boyu Chen, Wanli Ouyang, Dong Wang, Xiaoyun Yang, Huchuan Lu},
title = {GradNet: Gradient-Guided Network for Visual Object Tracking},
booktitle = {ICCV},
month = {October},
year = {2019}
}
"""

from queue import Empty

from PySide2.QtCore import QRunnable, QThreadPool
from numpy import array

from Model.Gradnet.Import.track import just_show


class ImportThread(QRunnable):
    def __init__(self, input_queue, output_queue, rect_color, exit_event):
        super().__init__()
        self.model = just_show()
        self.first_frame = None
        self.tracking_object_rect = None
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.color = rect_color
        self.exit_event = exit_event

    def _set_tracking_object(self):
        # yx hw
        tracking_object = self.input_queue.get()
        self.first_frame = tracking_object[0]
        self.tracking_object_rect = tracking_object[1]
        x, y, w, h = self.tracking_object_rect
        self.model.send(
            (self.first_frame, (array((y + h / 2, x + w / 2)), array((h, w)))))

    def get_tracking_result(self, cur_frame):
        return self.model.send(cur_frame)

    def run(self):
        self._set_tracking_object()
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


class GradnetImport:
    def __init__(self, input_queue, output_queue, rect_color, exit_event):
        self.my_class = ImportThread(input_queue, output_queue, rect_color, exit_event)

    def start(self):
        QThreadPool.globalInstance().start(self.my_class)
