import threading

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtQml

gray_color_table = [QtGui.qRgb(i, i, i) for i in range(256)]


class CVAbstractFilter(QtCore.QObject):
    def process_image(self, src):
        dst = src
        return dst


class CVCapture(QtCore.QObject):
    started = QtCore.pyqtSignal()
    imageReady = QtCore.pyqtSignal()
    indexChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(CVCapture, self).__init__(parent)
        self._image = QtGui.QImage()
        self._index = 0

        self.m_videoCapture = cv2.VideoCapture()
        self.m_timer = QtCore.QBasicTimer()
        self.m_filters = []
        self.m_busy = False

    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(int)
    def start(self, *args):
        if args:
            self.setIndex(args[0])
        self.m_videoCapture.release()
        self.m_videoCapture = cv2.VideoCapture(self._index)
        if self.m_videoCapture.isOpened():
            self.m_timer.start(0, self)
            self.started.emit()

    @QtCore.pyqtSlot()
    def stop(self):
        self.m_timer.stop()

    def timerEvent(self, e):
        if e.timerId() != self.m_timer.timerId():
            return
        ret, frame = self.m_videoCapture.read()
        if not ret:
            self.m_timer.stop()
            return
        if not self.m_busy:
            threading.Thread(target=self.process_image, args=(np.copy(frame),)).start()

    @QtCore.pyqtSlot(np.ndarray)
    def process_image(self, frame):
        self.m_busy = True
        for f in self.m_filters:
            frame = f.process_image(frame)
        image = CVCapture.ToQImage(frame)
        self.m_busy = False
        QtCore.QMetaObject.invokeMethod(
            self,
            "setImage",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(QtGui.QImage, image),
        )

    @staticmethod
    def ToQImage(im):
        if im is None:
            return QtGui.QImage()
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QtGui.QImage(
                    im.data,
                    im.shape[1],
                    im.shape[0],
                    im.strides[0],
                    QtGui.QImage.Format_Indexed8,
                )
                qim.setColorTable(gray_color_table)
                return qim.copy()

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    w, h, _ = im.shape
                    rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    flip_image = cv2.flip(rgb_image, 1)
                    qim = QtGui.QImage(
                        flip_image.data, h, w, QtGui.QImage.Format_RGB888
                    )
                    return qim.copy()
        return QtGui.QImage()

    def image(self):
        return self._image

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if self._image == image:
            return
        self._image = image
        self.imageReady.emit()

    def index(self):
        return self._index

    def setIndex(self, index):
        if self._index == index:
            return
        self._index = index
        self.indexChanged.emit()

    @QtCore.pyqtProperty(QtQml.QQmlListProperty)
    def filters(self):
        return QtQml.QQmlListProperty(CVAbstractFilter, self, self.m_filters)

    image = QtCore.pyqtProperty(QtGui.QImage, fget=image, notify=imageReady)
    index = QtCore.pyqtProperty(int, fget=index, fset=setIndex, notify=indexChanged)
