import torch
from ..models import modeling_gpt
import pytest
import os

from pathlib import Path

@pytest.fixture(scope="session")
def gpt() -> modeling_gpt.GPT:
    return modeling_gpt.GPT(modeling_gpt.GPTConfig)

def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="auto", help="Specify device: 'cuda', 'cpu', 'mps'")
    parser.addoption("--data-path", action="store", help="Custom path for tiny_shakespeare dataset")

@pytest.fixture(scope="module")
def device(request) -> str:
    option_device = request.config.getoption("--device")
    if option_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return option_device

@pytest.fixture(scope="module")
def tiny_shakespeare(request) -> str:
    custom_path = request.config.getoption("--data-path")
    if custom_path:
        path = Path(custom_path)
    else:
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "data" / "tiny_shakespeare" / "input.txt"

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    return path.read_text()
