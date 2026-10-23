import pytest
import torch

from sfno.config import load_config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def paper():
    return load_config()


@pytest.fixture
def smoke():
    return load_config(profile="smoke")
