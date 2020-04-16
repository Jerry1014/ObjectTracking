from PySide2 import QtWidgets
from PySide2.QtCore import Slot, Signal, QRect, Qt


class TrackingWin(QtWidgets.QWidget):
    def __init__(self, settings):
        super().__init__()
        self.setWindowTitle('目标跟踪')
        self.settings = settings

        # 部件
        self.image_win = QtWidgets.QLabel('test')
        self.button = QtWidgets.QPushButton('请用鼠标选择跟踪对象')
        self.button.setEnabled(False)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)

        # 布局
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_win)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.slider)
        self.setLayout(self.layout)

        # 信号槽

        # 其他
        self.sub_win = None
