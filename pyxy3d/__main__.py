import sys

from caliscope.gui.main_widget import launch_main
import caliscope.logger

logger = caliscope.logger.get(__name__)


def CLI_parser():
    if len(sys.argv) == 1:
        launch_main()

    if len(sys.argv) == 2:
        launch_widget = sys.argv[1]

        # if launch_widget in ["calibrate", "cal", "-c"]:
        #     launch_extrinsic_calibration_widget(session_path)

        if launch_widget in ["record", "rec", "-r"]:
            pass
            # launch_recording_widget(session_path)
