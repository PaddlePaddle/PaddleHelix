import os
import pytest
import logging
import tempfile
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for all tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    yield

@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary directory for test output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def testcases_dir():
    """Return the path to the testcases directory."""
    return os.path.join(os.path.dirname(__file__), "testcases") 