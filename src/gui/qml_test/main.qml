import QtQuick 2.15
import QtQuick.Window 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15


ApplicationWindow{
    id: window
    width: 400
    height: 500
    visible: true
    title: qsTr("Login")

    Rectangle{
        width: 100
        height:100
        color: "black"
    }

    Rectangle{
        width: 100
        height:100
        color: "red"
    }
}