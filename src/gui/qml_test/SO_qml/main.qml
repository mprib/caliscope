import QtQuick 2.0
import PyCVQML 1.0
import Filters 1.0

Item {
    width: 800
    height: 600

    CVItem  {
        id: imageWriter
        anchors.fill: parent
        image: capture.image
    }

    MaxRGBFilter{
        id: max_rgb_filter
    }
    GrayFilter{
        id: gray_filter
    }

    CVCapture{
        id: capture
        index: 0
        filters: [max_rgb_filter, gray_filter]
        Component.onCompleted: capture.start()
    }
}