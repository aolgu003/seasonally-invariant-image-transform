"""Shared fixtures for the seasonally-invariant-image-transform test suite."""

import os
import random
import sys

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure the project root is on sys.path so that 'model', 'dataset', 'utils'
# packages can be imported without installing the project.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.unet import UNet


# ---------------------------------------------------------------------------
# Autouse fixture: deterministic seeds
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed_everything():
    """Fix all random seeds before every test for reproducibility."""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    yield


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """Return a CPU device for testing."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Image directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paired_image_dir(tmp_path):
    """Create a temporary directory with on/ and off/ subdirs containing
    matching 64x64 grayscale PNG files.  Returns the parent path that contains
    both subdirectories.
    """
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()

    rng = np.random.RandomState(42)
    for i in range(5):
        name = f"img_{i:04d}.png"
        # Grayscale images (mode 'L')
        arr_on = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        arr_off = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(arr_on, mode="L").save(on_dir / name)
        Image.fromarray(arr_off, mode="L").save(off_dir / name)

    return str(tmp_path)


@pytest.fixture
def paired_rgb_image_dir(tmp_path):
    """Same as paired_image_dir but with 3-channel RGB PNGs."""
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()

    rng = np.random.RandomState(42)
    for i in range(5):
        name = f"img_{i:04d}.png"
        arr_on = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        arr_off = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr_on, mode="RGB").save(on_dir / name)
        Image.fromarray(arr_off, mode="RGB").save(off_dir / name)

    return str(tmp_path)


@pytest.fixture
def single_gray_png(tmp_path):
    """Create a single 64x64 grayscale PNG.  Returns the file path."""
    arr = np.random.RandomState(42).randint(0, 256, (64, 64), dtype=np.uint8)
    path = tmp_path / "gray_test.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


@pytest.fixture
def single_rgb_png(tmp_path):
    """Create a single 64x64 RGB PNG.  Returns the file path."""
    arr = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "rgb_test.png"
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path)


@pytest.fixture
def large_png(tmp_path):
    """Create a PNG larger than 2000x2000 pixels.  Returns the file path."""
    arr = np.random.RandomState(42).randint(0, 256, (2200, 2400), dtype=np.uint8)
    path = tmp_path / "large_test.png"
    Image.fromarray(arr, mode="L").save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unet():
    """Return a default UNet on CPU (1-ch input, 1-ch output, bilinear)."""
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.eval()
    return model
