import logging
import sys

from caliscope.gui.main_widget import launch_main
from caliscope.logger import setup_logging
from caliscope.startup import initialize_app

setup_logging()
initialize_app()
logger = logging.getLogger(__name__)


def CLI_parser():
    if len(sys.argv) == 1:
        launch_main()

    if len(sys.argv) == 2:
        sys.argv[1]
