"""Tests for the SIFT Siamese dataset.

BDD Feature: SIFT Siamese Dataset (dataset/neg_sift_dataset.py)

This dataset applies albumentations augmentation consistently to both
images, loads via OpenCV in grayscale, and uses different normalization
stats than the NCC dataset.  Negative sampling is NOT done here — it
happens in the training loop.
"""

import os

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from dataset.neg_sift_dataset import SiameseDataset


# -- Helpers ---------------------------------------------------------------
def _make_pair_dir(tmp_path, n=4, size=(64, 64)):
    on = tmp_path / "on"
    off = tmp_path / "off"
    on.mkdir()
    off.mkdir()
    for i in range(n):
        name = f"img_{i:03d}.png"
        Image.new("L", size, color=140).save(str(on / name))
        Image.new("L", size, color=100).save(str(off / name))
    return tmp_path


# -- Scenario: Augmentation is applied consistently to both images ---------
class TestConsistentAugmentation:
    """BDD: Given paired images · When augmentation runs · Then the same
    spatial transforms are applied to both via additional_targets.

    We verify indirectly: both tensors have the same spatial shape after
    augmentation, and the dataset does not crash.
    """

    def test_same_shape_after_augment(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=3)
        ds = SiameseDataset(str(root))
        img_on, img_off = ds[0]
        assert img_on.shape == img_off.shape

    def test_augmentation_pipeline_exists(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=2)
        ds = SiameseDataset(str(root))
        assert ds.transform is not None


# -- Scenario: Crop dimensions match original image dimensions -------------
class TestCropMatchesOriginal:
    """BDD: Given images of size (H,W) · When the transform is created ·
    Then RandomResizedCrop uses height=H, width=W.

    The constructor reads the first image to determine H, W.
    """

    def test_crop_size_matches_image(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=2, size=(80, 120))
        ds = SiameseDataset(str(root))
        img_on, img_off = ds[0]
        # Output spatial dims should match the original image size
        # (augmentation crops then resizes back to H,W)
        assert img_on.shape[1] == 80
        assert img_on.shape[2] == 120


# -- Scenario: Images are loaded as grayscale via OpenCV -------------------
class TestGrayscaleOpenCV:
    """BDD: Given an image path · When loaded · Then cv2.imread(path, 0)
    reads as grayscale and the output has 1 channel.
    """

    def test_single_channel_output(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=2)
        ds = SiameseDataset(str(root))
        img_on, img_off = ds[0]
        assert img_on.shape[0] == 1
        assert img_off.shape[0] == 1


# -- Scenario: Different normalization stats than NCC dataset ---------------
class TestNormalizationStats:
    """BDD: Given a loaded pair · When normalisation is applied ·
    Then on-season: (x-0.49)/0.135, off-season: (x-0.44)/0.12.

    We create uniform images to compute the expected normalised value.
    """

    def test_normalization_values(self, tmp_path):
        on = tmp_path / "on"
        off = tmp_path / "off"
        on.mkdir()
        off.mkdir()

        Image.new("L", (64, 64), color=128).save(str(on / "x.png"))
        Image.new("L", (64, 64), color=128).save(str(off / "x.png"))

        ds = SiameseDataset(str(tmp_path))
        img_on, img_off = ds[0]

        raw_val = 128.0 / 255.0
        expected_on = (raw_val - 0.49) / 0.135
        expected_off = (raw_val - 0.44) / 0.12

        # Augmentation may alter values slightly (crop/resize interpolation),
        # so use a generous tolerance.
        assert abs(img_on.float().mean().item() - expected_on) < 0.15
        assert abs(img_off.float().mean().item() - expected_off) < 0.15


# -- Scenario: No negative sampling in dataset ----------------------------
class TestNoNegativeSampling:
    """BDD: When __getitem__ is called · Then it returns a matched pair
    with no target label (just two tensors).
    """

    def test_returns_two_tensors(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=3)
        ds = SiameseDataset(str(root))
        result = ds[0]
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)


# -- Scenario: Output tensors are float type --------------------------------
class TestFloatOutput:
    """BDD: When __getitem__ is called · Then both tensors are .float().
    """

    def test_dtype_is_float32(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=2)
        ds = SiameseDataset(str(root))
        img_on, img_off = ds[0]
        assert img_on.dtype == torch.float32
        assert img_off.dtype == torch.float32
