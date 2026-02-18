"""Shared fixtures for the test suite.

Provides reusable test infrastructure: temporary directories with paired
on/off season images, pre-built model instances, and device selection.
"""

import os
import random

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Reproducibility — mirror the seed strategy used in the training scripts
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def seed_everything():
    """Fix all random seeds before every test for deterministic behavior."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
@pytest.fixture
def device():
    """Return a CPU device (CI-safe default)."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Temporary paired image directories
# ---------------------------------------------------------------------------
def _create_png(path, size=(64, 64), mode="L", value=128):
    """Write a small PNG to *path*."""
    img = Image.new(mode, size, color=value)
    img.save(path)


@pytest.fixture
def paired_image_dir(tmp_path):
    """Create a minimal on/off paired dataset and return *tmp_path*.

    Layout::

        tmp_path/on/img_000.png
        tmp_path/on/img_001.png
        tmp_path/off/img_000.png
        tmp_path/off/img_001.png
    """
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()

    for i in range(4):
        name = f"img_{i:03d}.png"
        _create_png(str(on_dir / name), value=140 + i)
        _create_png(str(off_dir / name), value=100 + i)

    return tmp_path


@pytest.fixture
def paired_rgb_image_dir(tmp_path):
    """Same as *paired_image_dir* but with 3-channel RGB images."""
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()

    for i in range(4):
        name = f"img_{i:03d}.png"
        _create_png(str(on_dir / name), size=(64, 64), mode="RGB", value=(140, 140, 140))
        _create_png(str(off_dir / name), size=(64, 64), mode="RGB", value=(100, 100, 100))

    return tmp_path


@pytest.fixture
def single_gray_png(tmp_path):
    """Create a single grayscale PNG and return its path."""
    p = tmp_path / "test_img.png"
    _create_png(str(p), size=(128, 128))
    return p


@pytest.fixture
def single_rgb_png(tmp_path):
    """Create a single RGB PNG and return its path."""
    p = tmp_path / "test_img.png"
    _create_png(str(p), size=(128, 128), mode="RGB", value=(128, 128, 128))
    return p


@pytest.fixture
def large_png(tmp_path):
    """Create a PNG larger than 2000x2000 for inference_img cap testing."""
    p = tmp_path / "big.png"
    _create_png(str(p), size=(2500, 3000))
    return p


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def unet():
    """Return a default UNet(1, 1, bilinear=True) in eval mode."""
    from model.unet import UNet
    m = UNet(n_channels=1, n_classes=1, bilinear=True)
    m.eval()
    return m
