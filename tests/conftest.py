import pytest

from caliscope.logger import setup_logging


@pytest.fixture(scope="session", autouse=True)
def setup_app_logging():
    """
    Fixture to configure the application's logging for the entire test session.
    This runs automatically once before any tests are executed.
    """
    setup_logging()
