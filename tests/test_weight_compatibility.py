"""Tests for pre-trained weight compatibility.

Maps to BDD Feature: Pre-trained Weight Compatibility.
"""

import os

import pytest
import torch

from model.unet import UNet

WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights2try",
)

CT_WEIGHTS = os.path.join(WEIGHTS_DIR, "ctCo300dx1-weights", "best_test_weights.pt")
MT_WEIGHTS = os.path.join(WEIGHTS_DIR, "mtCo150ax1-weights", "best_test_weights.pt")


class TestWeightCompatibility:
    """Tests that pre-trained weights load correctly into the UNet."""

    @pytest.mark.skipif(not os.path.isfile(CT_WEIGHTS), reason="ctCo300dx1 weights not found")
    def test_ctco300dx1_weights_load(self):
        """Scenario: ctCo300dx1 weights load into default UNet."""
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        state = torch.load(CT_WEIGHTS, map_location="cpu")
        model.load_state_dict(state)
        # If we get here, all keys matched

    @pytest.mark.skipif(not os.path.isfile(MT_WEIGHTS), reason="mtCo150ax1 weights not found")
    def test_mtco150ax1_weights_load(self):
        """Scenario: mtCo150ax1 weights load into default UNet."""
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        state = torch.load(MT_WEIGHTS, map_location="cpu")
        model.load_state_dict(state)

    @pytest.mark.skipif(not os.path.isfile(CT_WEIGHTS), reason="ctCo300dx1 weights not found")
    def test_pretrained_inference_output_range(self):
        """Scenario: Inference with pre-trained weights produces valid output.
        Output values should be in [0, 1] due to sigmoid activation."""
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        state = torch.load(CT_WEIGHTS, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert out.shape == (1, 1, 64, 64)
