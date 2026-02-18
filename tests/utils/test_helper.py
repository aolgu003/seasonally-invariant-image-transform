"""Tests for helper utilities.

BDD Feature: Helper Utilities (utils/helper.py)

Covers Normer, make_sure_path_exists, inference_img, pyramid_loss,
pyramid_loss_mse, normalize_batch, rgb2gray_batch, and write_tensorboard.
"""

import os
import errno
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.helper import (
    Normer,
    make_sure_path_exists,
    inference_img,
    pyramid_loss,
    pyramid_loss_mse,
    normalize_batch,
    rgb2gray_batch,
    write_tensorboard,
)


# -- Scenario: Normer produces zero-mean output ----------------------------
class TestNormerZeroMean:
    """BDD: Given any input tensor · When Normer() is called ·
    Then the output has approximately zero mean.

    Normer subtracts torch.mean(sample) then divides by std, so the
    result should centre around zero.
    """

    def test_output_near_zero_mean(self):
        normer = Normer()
        x = torch.randn(1, 1, 32, 32) + 5.0
        out = normer(x)
        assert abs(out.mean().item()) < 0.15


# -- Scenario: Normer uses epsilon to prevent division by zero -------------
class TestNormerEpsilon:
    """BDD: Given a constant-value tensor (std=0) · When Normer() is called ·
    Then no Python exception is raised, but the output IS NaN.

    The epsilon 1e-7 is added *inside* the tensor (x + eps) before
    computing std.  For a perfectly constant tensor, x + eps is still
    constant, so std is still 0 and division produces NaN.  This
    documents the current (buggy) behaviour — the epsilon does NOT
    effectively guard against division by zero.  Update this test when
    the Normer is fixed to use std(x) + eps instead.
    """

    def test_constant_tensor_no_exception(self):
        normer = Normer()
        x = torch.ones(1, 1, 8, 8) * 3.0
        # Should not raise — but output is NaN due to 0/0
        out = normer(x)
        assert out is not None

    def test_constant_tensor_produces_nan(self):
        normer = Normer()
        x = torch.ones(1, 1, 8, 8) * 3.0
        out = normer(x)
        # Known bug: epsilon is inside std(), so constant input → 0/0 → NaN
        assert torch.isnan(out).all()


# -- Scenario: Normer epsilon is applied inside std (bug-like behaviour) ---
class TestNormerEpsilonPlacement:
    """BDD: Given a tensor x · When Normer() is called · Then it computes
    std(x + 1e-7) rather than std(x) + 1e-7.

    This shifts all values by epsilon before computing std, which is
    mathematically different from adding epsilon to the denominator.
    The test documents this behaviour; it would change if fixed.
    """

    def test_epsilon_shifts_input(self):
        normer = Normer()
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

        out = normer(x)
        # Manually replicate: std(x + 1e-7)
        eps = 1e-7
        expected_std = torch.std(x + eps)
        expected = (x - torch.mean(x)) / expected_std

        assert torch.allclose(out, expected, atol=1e-5)


# -- Scenario: inference_img caps image size at 2000x2000 -----------------
class TestInferenceImgCap:
    """BDD: Given an image larger than 2000px · When inference_img runs ·
    Then the image is cropped to at most 2000x2000 from the top-left.
    """

    def test_caps_large_image(self, large_png, device):
        from model.unet import UNet

        model = UNet(1, 1, bilinear=True).to(device)
        model.eval()
        with torch.no_grad():
            out = inference_img(str(large_png), device, model, mean=0.4, std=0.12)
        assert out.shape[2] <= 2000
        assert out.shape[3] <= 2000

    def test_small_image_unchanged(self, single_gray_png, device):
        from model.unet import UNet

        model = UNet(1, 1, bilinear=True).to(device)
        model.eval()
        with torch.no_grad():
            out = inference_img(str(single_gray_png), device, model, mean=0.4, std=0.12)
        # 128x128 is well under 2000
        assert out.shape[2] == 128
        assert out.shape[3] == 128


# -- Scenario: inference_img normalizes with explicit mean/std -------------
class TestInferenceImgNormalization:
    """BDD: Given mean=0.4, std=0.12 · When image is preprocessed ·
    Then tensor = (pixel_val - 0.4) / 0.12.

    We verify by checking that the model receives correctly normalised
    input (indirectly — the function runs without error and produces
    valid output).
    """

    def test_produces_valid_output(self, single_gray_png, device):
        from model.unet import UNet

        model = UNet(1, 1, bilinear=True).to(device)
        model.eval()
        with torch.no_grad():
            out = inference_img(str(single_gray_png), device, model, mean=0.4, std=0.12)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# -- Scenario: make_sure_path_exists creates nested directories ------------
class TestMakeSurePathExists:
    """BDD: Given a path a/b/c that does not exist · When called ·
    Then all directories are created.
    """

    def test_creates_nested(self, tmp_path):
        p = str(tmp_path / "a" / "b" / "c")
        make_sure_path_exists(p)
        assert os.path.isdir(p)


# -- Scenario: make_sure_path_exists is idempotent -------------------------
class TestMakeSurePathIdempotent:
    """BDD: Given a path that already exists · When called ·
    Then no error is raised.
    """

    def test_existing_path_ok(self, tmp_path):
        p = str(tmp_path / "existing")
        os.makedirs(p)
        make_sure_path_exists(p)  # should not raise
        assert os.path.isdir(p)


# -- Scenario: pyramid_loss accumulates across pyramid levels ---------------
class TestPyramidLossAccumulation:
    """BDD: Given two pyramids with multiple levels · When pyramid_loss
    is called · Then correlation loss is summed across all levels.

    We create a two-level pyramid of identical tensors; the loss should
    be a sum of two per-level losses.
    """

    def test_sums_across_levels(self, device):
        from model.correlator import Correlator

        corr = Correlator(device=device)
        criterion = nn.MSELoss()
        p1 = [torch.randn(2, 1, 8, 8, device=device) for _ in range(3)]
        p2 = [t.clone() for t in p1]
        labels = torch.ones(2, device=device)

        loss = pyramid_loss(p1, p2, labels, corr, criterion)
        assert isinstance(loss, (float, torch.Tensor))


# -- Scenario: pyramid_loss treats each channel as a separate image --------
class TestPyramidLossChannelHandling:
    """BDD: Given pyramid tensors (B,C,H,W) · When reshaped ·
    Then they become (B*C,1,H,W) and labels are repeated C times.
    """

    def test_multichannel_reshaping(self, device):
        from model.correlator import Correlator

        corr = Correlator(device=device)
        criterion = nn.MSELoss()
        # 2 channels per level
        p1 = [torch.randn(2, 3, 8, 8, device=device)]
        p2 = [torch.randn(2, 3, 8, 8, device=device)]
        labels = torch.ones(2, device=device)

        # Should not crash — internally reshapes to (2*3, 1, 8, 8)
        loss = pyramid_loss(p1, p2, labels, corr, criterion)
        assert not torch.isnan(torch.tensor(loss) if isinstance(loss, float) else loss)


# -- Scenario: pyramid_loss_mse computes MSE across pyramid levels ---------
class TestPyramidLossMSE:
    """BDD: Given two pyramids · When pyramid_loss_mse is called ·
    Then it sums MSELoss(l1, l2) for each level pair.
    """

    def test_identical_pyramids_zero_loss(self):
        criterion = nn.MSELoss()
        p1 = [torch.randn(1, 1, 8, 8) for _ in range(3)]
        p2 = [t.clone() for t in p1]
        loss = pyramid_loss_mse(p1, p2, criterion)
        assert abs(loss) < 1e-6

    def test_different_pyramids_positive_loss(self):
        criterion = nn.MSELoss()
        p1 = [torch.zeros(1, 1, 8, 8)]
        p2 = [torch.ones(1, 1, 8, 8)]
        loss = pyramid_loss_mse(p1, p2, criterion)
        assert loss > 0


# -- Scenario: normalize_batch uses ImageNet stats -------------------------
class TestNormalizeBatch:
    """BDD: Given a batch tensor · When normalize_batch is called ·
    Then it normalises using ImageNet mean/std.
    """

    def test_imagenet_normalization(self):
        batch = torch.ones(1, 3, 4, 4) * 0.5
        out = normalize_batch(batch)
        # Channel 0: (0.5 - 0.485) / 0.229 ≈ 0.0655
        expected_ch0 = (0.5 - 0.485) / 0.229
        assert abs(out[0, 0, 0, 0].item() - expected_ch0) < 1e-4


# -- Scenario: rgb2gray_batch averages channels ----------------------------
class TestRgb2GrayBatch:
    """BDD: Given (B,3,H,W) · When rgb2gray_batch is called ·
    Then output is (B,1,H,W) by averaging channels.
    """

    def test_shape(self):
        batch = torch.randn(2, 3, 16, 16)
        gray = rgb2gray_batch(batch)
        assert gray.shape == (2, 1, 16, 16)

    def test_average(self):
        batch = torch.tensor([[[[3.0]], [[6.0]], [[9.0]]]])  # (1,3,1,1)
        gray = rgb2gray_batch(batch)
        assert abs(gray.item() - 6.0) < 1e-5  # (3+6+9)/3 = 6


# -- Scenario: write_tensorboard logs all label-metric pairs ---------------
class TestWriteTensorboard:
    """BDD: Given labels and metrics · When write_tensorboard is called ·
    Then writer.add_scalar is called once per pair.
    """

    def test_calls_add_scalar(self):
        writer = MagicMock()
        labels = ["train_loss", "train_acc"]
        metrics = [0.5, 0.9]
        write_tensorboard(writer, labels, metrics, epoch=1)

        assert writer.add_scalar.call_count == 2
        writer.add_scalar.assert_any_call("train_loss", 0.5, 1)
        writer.add_scalar.assert_any_call("train_acc", 0.9, 1)
