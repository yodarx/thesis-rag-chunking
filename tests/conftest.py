import pytest
import faiss
from tqdm import tqdm

@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """
    Sets up the test environment to prevent common threading issues.
    Runs once per test session.
    """
    tqdm.monitor_interval = 0
    faiss.omp_set_num_threads(1)

    yield