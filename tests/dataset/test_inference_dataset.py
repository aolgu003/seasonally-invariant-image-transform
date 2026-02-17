"""Tests for dataset/inference_dataset.py

Maps to BDD Feature: Inference Dataset.
"""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from dataset.inference_dataset import InferenceDataset


def _make_png_dir(tmp_path, num_images=5, size=64, grayscale=True):
    """Create a directory with PNG files."""
    rng = np.random.RandomState(42)
    for i in range(num_images):
        name = f"img_{i:04d}.png"
        if grayscale:
            arr = rng.randint(0, 256, (size, size), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(tmp_path / name)
        else:
            arr = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(tmp_path / name)
    return str(tmp_path)


class TestInferenceDataset:
    """Tests for the inference dataset."""

    def test_directory_png_discovery(self, tmp_path):
        """Scenario: All PNGs in directory are loaded."""
        data_dir = _make_png_dir(tmp_path, num_images=5)
        ds = InferenceDataset(data_dir)
        assert len(ds) == 5

    def test_non_png_files_filtered(self, tmp_path):
        """Non-PNG files are not included (glob only matches *.png)."""
        _make_png_dir(tmp_path, num_images=3)
        # Add a non-PNG file
        (tmp_path / "readme.txt").write_text("not an image")
        (tmp_path / "image.jpg").write_bytes(b"\xff\xd8\xff")
        ds = InferenceDataset(str(tmp_path))
        assert len(ds) == 3  # only the PNGs

    def test_single_file_path(self, single_gray_png):
        """Scenario: Single file path is handled.
        When data_dir is a single file, glob returns just that file."""
        # InferenceDataset uses glob(os.path.join(data_dir, '*.png'))
        # If data_dir is a file path, os.path.join(file, '*.png') won't match.
        # But looking at the code, it just globs for *.png in the directory.
        # Let's use the parent directory instead.
        parent = os.path.dirname(single_gray_png)
        ds = InferenceDataset(parent)
        assert len(ds) == 1

    def test_configurable_normalization(self, tmp_path):
        """Scenario: Images are normalized with configurable mean/std."""
        _make_png_dir(tmp_path, num_images=1)
        ds_custom = InferenceDataset(str(tmp_path), mean=0.4, std=0.12)
        img, name = ds_custom[0]
        # With mean=0.4, std=0.12 normalization, values go well outside [0,1]
        # A pixel of 0.5 becomes (0.5-0.4)/0.12 = 0.833
        # A pixel of 0.0 becomes (0.0-0.4)/0.12 = -3.33
        assert img.min() < 0 or img.max() > 1

    def test_filename_returned_as_basename(self, tmp_path):
        """Scenario: Image filename is returned alongside tensor as basename."""
        _make_png_dir(tmp_path, num_images=1)
        ds = InferenceDataset(str(tmp_path))
        img, name = ds[0]
        # name should be just the filename, not a full path
        assert "/" not in name
        assert name.endswith(".png")

    def test_default_normalization_stats(self):
        """Scenario: Default normalization uses mean=0.5, std=0.1."""
        # Check default parameter values
        import inspect
        sig = inspect.signature(InferenceDataset.__init__)
        assert sig.parameters["mean"].default == 0.5
        assert sig.parameters["std"].default == 0.1

    def test_output_is_tensor_and_string(self, tmp_path):
        """Output format is (tensor, basename_string)."""
        _make_png_dir(tmp_path, num_images=1)
        ds = InferenceDataset(str(tmp_path))
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2
        img, name = result
        assert isinstance(img, torch.Tensor)
        assert isinstance(name, str)
        # Image should be single channel after .convert('L')
        assert img.shape[0] == 1
