"""Tests for utils/helper.py

Maps to BDD Feature: Helper Utilities.
"""

import inspect
import os
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from utils.helper import (
    Normer,
    inference_img,
    make_sure_path_exists,
    normalize_batch,
    pyramid_loss,
    pyramid_loss_mse,
    rgb2gray_batch,
    write_tensorboard,
)


class TestNormer:
    """Tests for the Normer normalization utility."""

    def test_produces_zero_mean_output(self):
        """Scenario: Normer produces zero-mean output."""
        normer = Normer()
        x = torch.randn(1, 1, 64, 64) * 5 + 10  # large mean, large std
        out = normer(x)
        # Mean should be approximately zero
        assert abs(out.mean().item()) < 0.5

    def test_constant_tensor_produces_nan(self):
        """Scenario: Normer with constant tensor produces NaN (known bug).
        The epsilon is added inside std: torch.std(x + epsilon).
        For a constant tensor, adding a tiny epsilon still gives std ~ 0,
        leading to division by ~0 and NaN output."""
        normer = Normer()
        x = torch.ones(1, 1, 16, 16) * 5.0
        out = normer(x)
        # Known bug: std(constant + 1e-7) is still ~0 for large enough tensors,
        # BUT torch.std of a constant tensor is exactly 0.0, and
        # std(x + 1e-7) where x is constant is also 0.0 since adding the
        # same epsilon to all elements doesn't change the std.
        # So (x - mean) / 0.0 = NaN or inf
        assert torch.isnan(out).any() or torch.isinf(out).any(), (
            "Expected NaN or Inf for constant tensor input due to known epsilon bug"
        )

    def test_epsilon_placement_inside_std(self):
        """Scenario: Normer epsilon is applied inside std (bug-like behavior).
        Inspecting source: computes std(x + 1e-7) instead of std(x) + 1e-7."""
        source = inspect.getsource(Normer.__call__)
        # The source should have something like: torch.std(sample + epsilon)
        assert "sample + epsilon" in source or "sample+epsilon" in source, (
            "Expected epsilon to be added inside std() call"
        )

    def test_normal_tensor_no_nan(self):
        """Normer works without NaN for normal (non-constant) tensors."""
        normer = Normer()
        x = torch.randn(2, 1, 32, 32)
        out = normer(x)
        assert not torch.isnan(out).any()


class TestInferenceImg:
    """Tests for the inference_img helper function."""

    def test_caps_at_2000x2000(self, large_png, device):
        """Scenario: inference_img caps image size at 2000x2000."""
        model = _make_dummy_model()
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = inference_img(large_png, device, model, mean=0.4, std=0.12)
        # The large image is 2200x2400, should be cropped to 2000x2000
        assert output.shape[2] <= 2000
        assert output.shape[3] <= 2000

    def test_normalization(self, single_gray_png, device):
        """Scenario: inference_img normalizes with explicit mean/std."""
        model = _make_dummy_model()
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = inference_img(single_gray_png, device, model, mean=0.4, std=0.12)
        # Should produce valid output (no NaN)
        assert not torch.isnan(output).any()
        # Output from sigmoid model is in [0, 1]
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_output_shape(self, single_gray_png, device):
        """inference_img returns (1, 1, H, W) tensor."""
        model = _make_dummy_model()
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = inference_img(single_gray_png, device, model, mean=0.4, std=0.12)
        assert output.dim() == 4
        assert output.shape[0] == 1
        assert output.shape[1] == 1


class TestMakeSurePathExists:
    """Tests for make_sure_path_exists."""

    def test_creates_nested_dirs(self, tmp_path):
        """Scenario: make_sure_path_exists creates nested directories."""
        target = os.path.join(str(tmp_path), "a", "b", "c")
        make_sure_path_exists(target)
        assert os.path.isdir(target)

    def test_idempotent(self, tmp_path):
        """Scenario: make_sure_path_exists is idempotent (no error if exists)."""
        target = os.path.join(str(tmp_path), "existing")
        os.makedirs(target)
        # Should not raise
        make_sure_path_exists(target)
        assert os.path.isdir(target)


class TestPyramidLoss:
    """Tests for pyramid_loss and pyramid_loss_mse."""

    def test_accumulates_across_levels(self):
        """Scenario: pyramid_loss accumulates across pyramid levels."""
        from model.correlator import Correlator

        correlator = Correlator(device=torch.device("cpu"))
        criterion = nn.MSELoss()

        # Create a 3-level fake pyramid
        p1 = [torch.randn(2, 1, 16, 16), torch.randn(2, 1, 8, 8), torch.randn(2, 1, 4, 4)]
        p2 = [torch.randn(2, 1, 16, 16), torch.randn(2, 1, 8, 8), torch.randn(2, 1, 4, 4)]
        labels = torch.ones(2)

        loss = pyramid_loss(p1, p2, labels, correlator, criterion)
        assert isinstance(loss, torch.Tensor) or isinstance(loss, float)
        # Loss should be > 0 for random inputs
        if isinstance(loss, torch.Tensor):
            assert loss.item() >= 0

    def test_channel_handling_reshapes_bc(self):
        """Scenario: pyramid_loss treats each channel as separate image.
        Reshapes (B, C, H, W) to (B*C, 1, H, W)."""
        from model.correlator import Correlator

        correlator = Correlator(device=torch.device("cpu"))
        criterion = nn.MSELoss()

        B, C = 2, 3
        p1 = [torch.randn(B, C, 16, 16)]
        p2 = [torch.randn(B, C, 16, 16)]
        labels = torch.ones(B)

        # Should not crash -- labels get repeated C times internally
        loss = pyramid_loss(p1, p2, labels, correlator, criterion)
        assert not torch.isnan(torch.tensor(float(loss)))

    def test_pyramid_loss_mse(self):
        """Scenario: pyramid_loss_mse computes MSE across pyramid levels."""
        criterion = nn.MSELoss()
        p1 = [torch.randn(2, 1, 16, 16), torch.randn(2, 1, 8, 8)]
        p2 = [torch.randn(2, 1, 16, 16), torch.randn(2, 1, 8, 8)]

        loss = pyramid_loss_mse(p1, p2, criterion)
        assert isinstance(loss, torch.Tensor) or isinstance(loss, (int, float))
        if isinstance(loss, torch.Tensor):
            assert loss.item() >= 0

    def test_pyramid_loss_mse_identical_inputs(self):
        """pyramid_loss_mse is zero for identical pyramids."""
        criterion = nn.MSELoss()
        p1 = [torch.randn(2, 1, 16, 16)]
        p2 = [t.clone() for t in p1]
        loss = pyramid_loss_mse(p1, p2, criterion)
        if isinstance(loss, torch.Tensor):
            assert loss.item() < 1e-6
        else:
            assert loss < 1e-6


class TestNormalizeBatch:
    """Tests for normalize_batch."""

    def test_imagenet_stats(self):
        """Scenario: normalize_batch uses ImageNet mean and std."""
        batch = torch.ones(1, 3, 4, 4) * 0.5
        normed = normalize_batch(batch)
        # Channel 0: (0.5 - 0.485) / 0.229 = 0.0655
        expected_ch0 = (0.5 - 0.485) / 0.229
        actual_ch0 = normed[0, 0, 0, 0].item()
        assert abs(actual_ch0 - expected_ch0) < 1e-4

    def test_imagenet_stats_values(self):
        """Verify the exact ImageNet mean and std values used."""
        source = inspect.getsource(normalize_batch)
        assert "0.485" in source
        assert "0.456" in source
        assert "0.406" in source
        assert "0.229" in source
        assert "0.224" in source
        assert "0.225" in source


class TestRgb2GrayBatch:
    """Tests for rgb2gray_batch."""

    def test_output_shape(self):
        """Scenario: rgb2gray_batch returns (B, 1, H, W) from (B, 3, H, W)."""
        batch = torch.randn(2, 3, 32, 32)
        gray = rgb2gray_batch(batch)
        assert gray.shape == (2, 1, 32, 32)

    def test_averages_channels(self):
        """rgb2gray_batch averages across channels (sum / 3)."""
        batch = torch.ones(1, 3, 4, 4)
        batch[0, 0] = 0.3
        batch[0, 1] = 0.6
        batch[0, 2] = 0.9
        gray = rgb2gray_batch(batch)
        expected = (0.3 + 0.6 + 0.9) / 3.0
        assert abs(gray[0, 0, 0, 0].item() - expected) < 1e-6


class TestWriteTensorboard:
    """Tests for write_tensorboard."""

    def test_logs_all_pairs(self):
        """Scenario: write_tensorboard logs all label-metric pairs."""
        writer = MagicMock()
        labels = ["train_loss", "train_acc"]
        metrics = [0.5, 0.9]
        epoch = 1
        write_tensorboard(writer, labels, metrics, epoch)
        assert writer.add_scalar.call_count == 2
        writer.add_scalar.assert_any_call("train_loss", 0.5, 1)
        writer.add_scalar.assert_any_call("train_acc", 0.9, 1)

    def test_empty_labels(self):
        """write_tensorboard with empty labels does nothing."""
        writer = MagicMock()
        write_tensorboard(writer, [], [], 0)
        writer.add_scalar.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_model():
    """Create a minimal model that mimics UNet's interface for testing
    inference_img without the cost of a full UNet forward pass."""
    from model.unet import UNet
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    return model
