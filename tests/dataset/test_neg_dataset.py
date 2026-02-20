"""Tests for the NCC Siamese dataset.

BDD Feature: NCC Siamese Dataset (dataset/neg_dataset.py)

The NCC dataset loads paired on/off season images, applies negative
sampling, converts to grayscale, and normalises with hardcoded stats.
These tests verify every documented behavior using a tiny synthetic
dataset created in a temp directory.
"""

import os
import random

import pytest
import torch
from PIL import Image

from dataset.neg_dataset import SiameseDataset


# -- Helpers ---------------------------------------------------------------
def _make_pair_dir(tmp_path, n=4, size=(32, 32)):
    """Create on/ and off/ dirs with *n* matching grayscale PNGs."""
    on = tmp_path / "on"
    off = tmp_path / "off"
    on.mkdir()
    off.mkdir()
    for i in range(n):
        name = f"img_{i:03d}.png"
        Image.new("L", size, color=140 + i).save(str(on / name))
        Image.new("L", size, color=100 + i).save(str(off / name))
    return tmp_path


# -- Scenario: Paired images are loaded from on/ and off/ directories ------
class TestPairLoading:
    """BDD: Given data_root with on/ and off/ containing matching PNGs ·
    When the dataset is initialised · Then each on/ image has a verified
    off/ counterpart.

    The constructor globs on/*.png and asserts each has a match in off/.
    """

    def test_loads_all_pairs(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=5)
        ds = SiameseDataset(str(root))
        assert len(ds) == 5


# -- Scenario: Initialization fails if pairs are incomplete ----------------
class TestMissingPairFails:
    """BDD: Given an on/ image with no off/ match · When dataset inits ·
    Then AssertionError is raised (line 16).
    """

    def test_missing_off_raises(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=2)
        # Add an extra on/ image with no off/ pair
        Image.new("L", (32, 32)).save(str(tmp_path / "on" / "orphan.png"))
        with pytest.raises(AssertionError):
            SiameseDataset(str(root))


# -- Scenario: Negative sampling produces mismatched pairs -----------------
class TestNegativeSampling:
    """BDD: Given negative_weighting=0.5 · When __getitem__ is called many
    times · Then ~50% of samples have target=0.

    We run 200 draws and check the ratio is in a reasonable window.
    """

    def test_negative_ratio(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=10)
        ds = SiameseDataset(str(root), negative_weighting=0.5)

        negatives = sum(ds[i % len(ds)][1] == 0 for i in range(200))
        ratio = negatives / 200
        assert 0.3 < ratio < 0.7


# -- Scenario: Positive samples return matched pairs -----------------------
class TestPositiveSamples:
    """BDD: Given negative_weighting=0 (no negatives) · When __getitem__
    is called · Then target is always 1.
    """

    def test_all_positive(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=4)
        ds = SiameseDataset(str(root), negative_weighting=0.0)
        for i in range(len(ds)):
            _, target = ds[i]
            assert target == 1


# -- Scenario: Negative samples never return the same index ----------------
class TestNegativeIndexDiffers:
    """BDD: Given a negative sample · When the replacement index is chosen ·
    Then it is guaranteed != index.

    With negative_weighting=1.0 every sample is negative.  We call
    __getitem__ and verify the function doesn't hang (the while-loop exits).
    """

    def test_does_not_hang(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=4)
        ds = SiameseDataset(str(root), negative_weighting=1.0)
        # Just exercise every index — if the loop were broken it'd hang
        for i in range(len(ds)):
            result = ds[i]
            assert result[1] == 0


# -- Scenario: Images are converted to grayscale --------------------------
class TestGrayscaleConversion:
    """BDD: Given any input image · When loaded · Then it is single-channel.

    Even if the source PNG is RGB, .convert('L') reduces it to 1 channel.
    """

    def test_output_single_channel(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=2)
        ds = SiameseDataset(str(root), negative_weighting=0.0)
        (img_on, img_off), _ = ds[0]
        assert img_on.shape[0] == 1
        assert img_off.shape[0] == 1


# -- Scenario: Hardcoded normalization is applied --------------------------
class TestHardcodedNormalization:
    """BDD: Given a loaded pair · When normalisation is applied ·
    Then on-season uses (x-0.49)/0.12 and off-season uses (x-0.44)/0.10.

    We create uniform-value images so we can predict the exact normalised
    value.
    """

    def test_normalization_values(self, tmp_path):
        on = tmp_path / "on"
        off = tmp_path / "off"
        on.mkdir()
        off.mkdir()

        # Value 128/255 ≈ 0.502
        Image.new("L", (32, 32), color=128).save(str(on / "x.png"))
        Image.new("L", (32, 32), color=128).save(str(off / "x.png"))

        ds = SiameseDataset(str(tmp_path), negative_weighting=0.0)
        (img_on, img_off), _ = ds[0]

        # on: (0.502 - 0.49) / 0.12 ≈ 0.1
        raw_val = 128.0 / 255.0
        expected_on = (raw_val - 0.49) / 0.12
        expected_off = (raw_val - 0.44) / 0.10

        assert torch.allclose(img_on, torch.full_like(img_on, expected_on), atol=0.01)
        assert torch.allclose(img_off, torch.full_like(img_off, expected_off), atol=0.01)


# -- Scenario: Dataset size is controlled by samples_to_use ---------------
class TestSamplesToUse:
    """BDD: Given samples_to_use=0.5 and 10 images · When __len__ is
    called · Then it returns 5.
    """

    def test_half_dataset(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=10)
        ds = SiameseDataset(str(root), samples_to_use=0.5)
        assert len(ds) == 5


# -- Scenario: samples_to_use greater than 1 is rejected ------------------
class TestSamplesToUseUpperBound:
    """BDD: Given samples_to_use=1.5 · When dataset inits ·
    Then AssertionError is raised (line 21).
    """

    def test_rejects_gt_one(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=4)
        with pytest.raises(AssertionError):
            SiameseDataset(str(root), samples_to_use=1.5)


# -- Scenario: Output format is ((img_on, img_off), target) ---------------
class TestOutputFormat:
    """BDD: When __getitem__ is called · Then it returns
    ((tensor, tensor), int).
    """

    def test_structure(self, tmp_path):
        root = _make_pair_dir(tmp_path, n=2)
        ds = SiameseDataset(str(root))
        item = ds[0]
        (img_on, img_off), target = item
        assert isinstance(img_on, torch.Tensor)
        assert isinstance(img_off, torch.Tensor)
        assert isinstance(target, int)
        assert target in (0, 1)
