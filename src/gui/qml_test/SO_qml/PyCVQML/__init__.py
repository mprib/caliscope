from PyQt5 import QtQml

from .cvcapture import CVAbstractFilter, CVCapture
from .cvitem import CVItem


def registerTypes(uri="PyCVQML"):
    QtQml.qmlRegisterType(CVCapture, uri, 1, 0, "CVCapture")
    QtQml.qmlRegisterType(CVItem, uri, 1, 0, "CVItem")
