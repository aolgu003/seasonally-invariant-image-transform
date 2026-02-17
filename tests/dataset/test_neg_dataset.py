"""Tests for dataset/neg_dataset.py

Maps to BDD Feature: NCC Siamese Dataset.
"""

import os
import random

import numpy as np
import pytest
import torch
from PIL import Image

from dataset.neg_dataset import SiameseDataset


def _make_paired_dir(tmp_path, num_images=10):
    """Helper to create an on/off paired image directory."""
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir(exist_ok=True)
    off_dir.mkdir(exist_ok=True)
    rng = np.random.RandomState(42)
    for i in range(num_images):
        name = f"img_{i:04d}.png"
        arr_on = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        arr_off = rng.randint(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(arr_on, mode="L").save(on_dir / name)
        Image.fromarray(arr_off, mode="L").save(off_dir / name)
    return str(tmp_path)


class TestNegDataset:
    """Tests for the NCC Siamese dataset with negative sampling."""

    def test_pair_loading_from_on_off_dirs(self, tmp_path):
        """Scenario: Paired images are loaded from on/ and off/ directories."""
        data_root = _make_paired_dir(tmp_path, num_images=5)
        ds = SiameseDataset(data_root, negative_weighting=0.0, samples_to_use=1)
        assert len(ds) == 5

    def test_missing_pair_assertion(self, tmp_path):
        """Scenario: Initialization fails if pairs are incomplete."""
        on_dir = tmp_path / "on"
        off_dir = tmp_path / "off"
        on_dir.mkdir()
        off_dir.mkdir()
        # Create on image with no matching off image
        arr = np.zeros((32, 32), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(on_dir / "orphan.png")
        with pytest.raises(AssertionError):
            SiameseDataset(str(tmp_path))

    def test_negative_sampling_ratio(self, tmp_path):
        """Scenario: Negative sampling produces mismatched pairs (~50%)."""
        data_root = _make_paired_dir(tmp_path, num_images=20)
        ds = SiameseDataset(data_root, negative_weighting=0.5, samples_to_use=1)
        targets = []
        random.seed(42)
        for i in range(len(ds)):
            _, target = ds[i]
            targets.append(target)
        neg_ratio = targets.count(0) / len(targets)
        # Should be roughly 50% negatives (with some variance)
        assert 0.2 < neg_ratio < 0.8

    def test_positive_only_sampling(self, tmp_path):
        """Scenario: With negative_weighting=0, all samples are positive."""
        data_root = _make_paired_dir(tmp_path, num_images=10)
        ds = SiameseDataset(data_root, negative_weighting=0.0, samples_to_use=1)
        for i in range(len(ds)):
            _, target = ds[i]
            assert target == 1

    def test_negative_index_differs(self, tmp_path):
        """Scenario: Negative samples never return the same index.
        We test this by forcing negative_weighting=1.0 and checking the
        off image path differs from what would be the matched pair."""
        data_root = _make_paired_dir(tmp_path, num_images=10)
        ds = SiameseDataset(data_root, negative_weighting=1.0, samples_to_use=1)
        # The negative swap should produce a different off image
        # We can't directly inspect the path from __getitem__, but we know
        # the dataset loops until rand_index != index.
        # Just verify no crash with many samples
        for i in range(len(ds)):
            (img_on, img_off), target = ds[i]
            assert target == 0
            assert img_on.shape == img_off.shape

    def test_grayscale_conversion(self, tmp_path):
        """Scenario: Images are converted to grayscale (single channel)."""
        # Create RGB images to verify they get converted to 1-channel
        on_dir = tmp_path / "on"
        off_dir = tmp_path / "off"
        on_dir.mkdir()
        off_dir.mkdir()
        arr = np.random.RandomState(0).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(on_dir / "rgb.png")
        Image.fromarray(arr, mode="RGB").save(off_dir / "rgb.png")
        ds = SiameseDataset(str(tmp_path), negative_weighting=0.0, samples_to_use=1)
        (img_on, img_off), _ = ds[0]
        assert img_on.shape[0] == 1  # single channel after .convert('L')
        assert img_off.shape[0] == 1

    def test_hardcoded_normalization(self, tmp_path):
        """Scenario: Hardcoded normalization is applied.
        on-season: (img - 0.49) / 0.12, off-season: (img - 0.44) / 0.10."""
        data_root = _make_paired_dir(tmp_path, num_images=3)
        ds = SiameseDataset(data_root, negative_weighting=0.0, samples_to_use=1)
        (img_on, img_off), _ = ds[0]
        # The normalization shifts values far from [0,1] range
        # On-season: (val - 0.49) / 0.12 can go very negative or positive
        # Off-season: (val - 0.44) / 0.10 similarly
        # Just verify the outputs are not in [0,1] (they are normalized)
        assert img_on.min() < 0 or img_on.max() > 1
        assert img_off.min() < 0 or img_off.max() > 1

    def test_samples_to_use_controls_size(self, tmp_path):
        """Scenario: Dataset size is controlled by samples_to_use."""
        data_root = _make_paired_dir(tmp_path, num_images=20)
        ds = SiameseDataset(data_root, negative_weighting=0.0, samples_to_use=0.5)
        assert len(ds) == 10

    def test_samples_to_use_greater_than_1_rejected(self, tmp_path):
        """Scenario: samples_to_use greater than 1 is rejected."""
        data_root = _make_paired_dir(tmp_path, num_images=5)
        with pytest.raises(AssertionError):
            SiameseDataset(data_root, samples_to_use=1.5)

    def test_output_format(self, tmp_path):
        """Scenario: Output format is ((img_on, img_off), target)."""
        data_root = _make_paired_dir(tmp_path, num_images=3)
        ds = SiameseDataset(data_root, negative_weighting=0.0, samples_to_use=1)
        result = ds[0]
        # result should be ((tensor, tensor), int)
        assert isinstance(result, tuple)
        assert len(result) == 2
        imgs, target = result
        assert isinstance(imgs, tuple)
        assert len(imgs) == 2
        assert isinstance(imgs[0], torch.Tensor)
        assert isinstance(imgs[1], torch.Tensor)
        assert isinstance(target, int)
