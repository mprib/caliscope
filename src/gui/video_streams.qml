import QtQuick 2.5
import QtQuick.Controls 2.5
import QtMultimedia


ApplicationWindow {
    visible:true
    width:600
    height:400
    title:"Camera Feeds"

    MediaPlayer {
        id:player
        source:0
        videoOutput:videoOutput
    }

    VideoOutput {

        id:videoOutput
        anchors.fill:parent     
    }
}