"""Tests for pre-trained weight compatibility.

BDD Feature: Pre-trained Weight Compatibility (weights2try/)

Pre-trained weights must remain loadable after any model changes.
These tests act as a regression guard — if UNet architecture changes
break the state dict, these tests fail immediately.
"""

import os

import pytest
import torch

from model.unet import UNet

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights2try")


def _weights_available(subdir, filename="best_test_weights.pt"):
    return os.path.isfile(os.path.join(WEIGHTS_DIR, subdir, filename))


# -- Scenario: ctCo300dx1 weights load into default UNet -------------------
class TestCtCo300dx1Weights:
    """BDD: Given ctCo300dx1-weights/best_test_weights.pt · When loaded
    into UNet(1,1,bilinear=True) · Then all keys match without error.

    If the encoder/decoder channel counts or layer names change, this
    test catches the incompatibility.
    """

    @pytest.mark.skipif(
        not _weights_available("ctCo300dx1-weights"),
        reason="ctCo300dx1 weights not present",
    )
    def test_load_ctCo300dx1(self):
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        path = os.path.join(WEIGHTS_DIR, "ctCo300dx1-weights", "best_test_weights.pt")
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)


# -- Scenario: mtCo150ax1 weights load into default UNet -------------------
class TestMtCo150ax1Weights:
    """BDD: Given mtCo150ax1-weights/best_test_weights.pt · When loaded
    into default UNet · Then all keys match without error.
    """

    @pytest.mark.skipif(
        not _weights_available("mtCo150ax1-weights"),
        reason="mtCo150ax1 weights not present",
    )
    def test_load_mtCo150ax1(self):
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        path = os.path.join(WEIGHTS_DIR, "mtCo150ax1-weights", "best_test_weights.pt")
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)


# -- Scenario: Inference with pre-trained weights produces valid output ----
class TestPretrainedInference:
    """BDD: Given a loaded model with pre-trained weights and a sample
    image · When inference runs · Then output values are in [0, 1].

    The sigmoid activation guarantees [0,1] range regardless of weights,
    but this test verifies that the loaded weights don't produce NaN or
    Inf due to corrupted parameters.
    """

    @pytest.mark.skipif(
        not _weights_available("ctCo300dx1-weights"),
        reason="ctCo300dx1 weights not present",
    )
    def test_valid_output_range(self):
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        path = os.path.join(WEIGHTS_DIR, "ctCo300dx1-weights", "best_test_weights.pt")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert out.min() >= 0.0
        assert out.max() <= 1.0
