"""
模型及代码来自
@inproceedings{ iccv19_SPLT,
    title={`Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-term Tracking},
    author={Yan, Bin and Zhao, Haojie and Wang, Dong and Lu, Huchuan and Yang, Xiaoyun},
    booktitle={IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
"""
from queue import Empty

from PySide2.QtCore import QRunnable, QThreadPool
from cv2.cv2 import cvtColor, COLOR_RGB2BGR

from Model.SPLT.Import.track import MobileTracker


class SPLTImport(QRunnable):
    def __init__(self, input_queue, output_queue, rect_color, exit_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.rect_color = rect_color
        self.exit_event = exit_event

        self.tracker = MobileTracker()

    def run(self) -> None:
        tracking_object = self.input_queue.get()
        frame = cvtColor(tracking_object[0], COLOR_RGB2BGR)
        self.tracker.init_first(frame, tracking_object[1])
        # import os
        # import cv2
        # path = r'D:\TEM\PycharmProjects\ObjectTracking\Resources\video\Walking2\img' + os.path.sep
        # frame_list = os.listdir(path)
        # image = cv2.imread(path + frame_list[0])
        # selection = [130, 132, 31, 115]
        # self.tracker.init_first(image, selection)

        while True:
            try:
                frame = self.input_queue.get(timeout=0.5)
                frame = cvtColor(frame, COLOR_RGB2BGR)
                track_result = self.tracker.track(frame)[0]
                self.output_queue.put(((track_result, self.rect_color),None))
            except Empty:
                if self.exit_event.is_set():
                    break


class SPLT:
    def __init__(self, input_queue, output_queue, rect_color, exit_event):
        super().__init__()
        self.model = SPLTImport(input_queue, output_queue, rect_color, exit_event)

    def start(self):
        QThreadPool.globalInstance().start(self.model)
