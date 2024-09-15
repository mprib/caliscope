import caliscope.logger

logger = caliscope.logger.get(__name__)


def test_no_video_calibrate():
    """
    Saving out of video may be maxing system resources when high res/many cameras.
    Allow user to toggle this off to free up resources
    """

    # copy over project to test
    pass


if __name__ == "__main__":
    pass
