import pytest
from tqdm import TMonitor

@pytest.fixture(scope="session", autouse=True)
def prevent_tqdm_monitor():
    """
    Globally prevent tqdm monitor thread from starting.
    This fixes segmentation faults when using FAISS + tqdm on macOS.
    """
    # Simply monkey-patch the start method of TMonitor to do nothing
    original_start = TMonitor.start
    TMonitor.start = lambda self: None

    yield

    # Restore (though usually not needed for session scope fixture)
    TMonitor.start = original_start


