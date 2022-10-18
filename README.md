# Overview

Trying to slowly build up a stable system for calibrating cameras for use with markerless motion capture. 

Within almost all the primary code modules there should be an `if __name__ == "__main__":` showcase at the bottom. No proper tests yet, but this is where someone might look to get a quick and dirty idea of what is going on.

Basic overview of the current functionality is this:

The `Camera` object is provided with a port (a.k.a. numerical source) and during initialization it will figure out basic resolution options/exposure settings. This camera object provides an interface to a camera.capture that can be read from. The `stop_rolling()` method uses a `self.stop_rolling_trigger` to inititate a shutdown of the capture. This is important when using threads to read from a capture device and then trying to make changes to the configuration of that device.

I am working with 4 webcams here. 3 of them can establish connections in about 2 seconds, but I also have a pricey logitech model that takes ~30 seconds to establish a connection. OBS is able to connect to this camera almost instantly, so who knows what is going on (there is also a highly notable lag in the real-time OBS video stream that is not present when connecting more slowly through OpenCV).

The `RealTimeDevice` is a central feature of the workflow. It takes a `Camera` and starts a thread of the method `roll_camera()`. Within `roll_camera()` a `_working_frame` is read from the camera and actions are performed on it before being copied over to `self.frame`. The RTD can be assigned a `Charuco` for use in the calibration.

The `FrameEmitter` reads the `real_time_device.frame` that is being updated within the `RealTimeDevice.roll_camera()` thread, and broadcasts it to a pyqt signal that can be picked up by the GUI. **IMPORTANTLY**: pixmap scaling takes place within this `FrameEmmitter` Qthread. There is only one scale shown within the GUI proper, and a `cv2.imshow` window is launched to show actual resolution. Attempts to scale images within the main GUI thread lead to errorless crashes when attempting to move the window after changing to a higer resolution from the default. I think the thread was just overwhelmed with that task on top of whatever the GUI has to manage.

The `Charuco` object provides lots of methods and properties to aid with calibration. It extends the basic OpenCV charuco to provided some properties and methods useful to the calibrator, as well as a means to override the square edge length.

The `IntrinsicCalibrator` is initialized by the `RealTimeDevice` with the relevant `Camera` and a `Charuco`. The camera is really just there to provide needed resolution settings that may be changing in real time via user interaction with the `RealTimeDevice`. It tracks charuco corners that the RTD feeds it and maintains an overlay of the grid capture history for providing visual feedback to the user about what is going into the calibration. This object will ultimately be called upon to run the calibration. 
