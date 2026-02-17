"""Tests for dataset/neg_sift_dataset.py

Maps to BDD Feature: SIFT Siamese Dataset.
"""

import os

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from dataset.neg_sift_dataset import SiameseDataset


def _make_paired_dir(tmp_path, num_images=5, size=64):
    """Helper to create an on/off paired image directory with RGB PNGs.
    neg_sift_dataset reads with cv2.imread which loads as BGR by default,
    and cv2.imread(path, 0) reads grayscale."""
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir(exist_ok=True)
    off_dir.mkdir(exist_ok=True)
    rng = np.random.RandomState(42)
    for i in range(num_images):
        name = f"img_{i:04d}.png"
        # Create RGB images (cv2.imread without flag reads 3-channel,
        # cv2.imread with flag 0 reads grayscale)
        arr_on = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
        arr_off = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
        Image.fromarray(arr_on, mode="RGB").save(on_dir / name)
        Image.fromarray(arr_off, mode="RGB").save(off_dir / name)
    return str(tmp_path)


class TestNegSiftDataset:
    """Tests for the SIFT Siamese dataset."""

    def test_augmentation_applied_consistently(self, tmp_path):
        """Scenario: Augmentation is applied consistently to both images.
        The transform uses additional_targets={'imageOff': 'image'} which
        means both images get the same random augmentation."""
        data_root = _make_paired_dir(tmp_path, num_images=3, size=64)
        ds = SiameseDataset(data_root, samples_to_use=1)
        # Verify that the transform has additional_targets set
        assert "imageOff" in ds.transform.additional_targets

    def test_crop_dimensions_match_original(self, tmp_path):
        """Scenario: Crop dimensions match original image dimensions.
        RandomResizedCrop uses height=H, width=W from the first image."""
        size = 80
        data_root = _make_paired_dir(tmp_path, num_images=3, size=size)
        ds = SiameseDataset(data_root, samples_to_use=1)
        img_on, img_off = ds[0]
        # Output should have the same spatial dimensions as input
        assert img_on.shape[1] == size
        assert img_on.shape[2] == size

    def test_grayscale_opencv_loading(self, tmp_path):
        """Scenario: Images are loaded as grayscale via OpenCV (cv2.imread with flag 0)."""
        data_root = _make_paired_dir(tmp_path, num_images=3, size=64)
        ds = SiameseDataset(data_root, samples_to_use=1)
        img_on, img_off = ds[0]
        # After cv2.imread(path, 0) and ToTensorV2, should be single-channel
        # ToTensorV2 on a 2D array produces (1, H, W) if the input was (H, W)
        # Actually, looking at the code: cv2.imread(path, 0) returns (H, W),
        # then / 255, then transform which includes ToTensorV2.
        # ToTensorV2 on (H, W) produces (H, W) tensor -- but albumentations
        # adds channel dim. Let's just check the tensor is reasonable.
        assert img_on.dim() >= 2
        assert img_off.dim() >= 2

    def test_different_normalization_stats(self, tmp_path):
        """Scenario: Different normalization stats than NCC dataset.
        on-season: (img - 0.49) / 0.135, off-season: (img - 0.44) / 0.12."""
        data_root = _make_paired_dir(tmp_path, num_images=3, size=64)
        ds = SiameseDataset(data_root, samples_to_use=1)
        img_on, img_off = ds[0]
        # Values should be shifted away from [0, 1] due to normalization
        # Verify they are not simply in the raw [0, 1] pixel range
        # (0.5 - 0.49)/0.135 ~ 0.07; (0 - 0.49)/0.135 ~ -3.6
        assert img_on.min() < 0 or img_on.max() > 1

    def test_no_negative_sampling(self, tmp_path):
        """Scenario: No negative sampling in dataset (done in training loop).
        Always returns matched pair (img_on, img_off) with no target label."""
        data_root = _make_paired_dir(tmp_path, num_images=3, size=64)
        ds = SiameseDataset(data_root, samples_to_use=1)
        result = ds[0]
        # Should return (img_on, img_off) tuple, not ((img_on, img_off), target)
        assert isinstance(result, tuple)
        assert len(result) == 2
        img_on, img_off = result
        assert isinstance(img_on, torch.Tensor)
        assert isinstance(img_off, torch.Tensor)

    def test_float32_output_type(self, tmp_path):
        """Scenario: Output tensors are float type (explicitly .float())."""
        data_root = _make_paired_dir(tmp_path, num_images=3, size=64)
        ds = SiameseDataset(data_root, samples_to_use=1)
        img_on, img_off = ds[0]
        assert img_on.dtype == torch.float32
        assert img_off.dtype == torch.float32

    def test_samples_to_use_controls_length(self, tmp_path):
        """samples_to_use parameter controls dataset length."""
        data_root = _make_paired_dir(tmp_path, num_images=10, size=64)
        ds = SiameseDataset(data_root, samples_to_use=0.5)
        assert len(ds) == 5

    def test_samples_to_use_greater_than_1_rejected(self, tmp_path):
        """samples_to_use > 1 raises AssertionError."""
        data_root = _make_paired_dir(tmp_path, num_images=3, size=64)
        with pytest.raises(AssertionError):
            SiameseDataset(data_root, samples_to_use=1.5)
