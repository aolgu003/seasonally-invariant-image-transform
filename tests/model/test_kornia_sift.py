"""Tests for model/kornia_sift.py

Maps to BDD Feature: SIFT Descriptor Extraction.
"""

import inspect

import torch
import pytest

from model.kornia_sift import KorniaSift


class TestKorniaSift:
    """Tests for the KorniaSift descriptor module."""

    def test_descriptors_have_128_dimensions(self):
        """Scenario: Descriptors have 128 dimensions."""
        sift = KorniaSift(num_features=10)
        # Create input large enough for detection
        x = torch.randn(1, 1, 128, 128).abs()  # positive values for detection
        desc, laf = sift(x)
        # desc shape should be (B, N_keypoints, 128)
        assert desc.shape[0] == 1
        assert desc.shape[2] == 128

    def test_detection_without_laf(self):
        """Scenario: Keypoints are detected when no LAF is provided."""
        sift = KorniaSift(num_features=10)
        x = torch.randn(1, 1, 128, 128).abs()
        desc, laf = sift(x, laf=None)
        # LAF should be returned with shape (B, N, 2, 3)
        assert laf is not None
        assert laf.dim() == 4
        assert laf.shape[0] == 1
        assert laf.shape[2] == 2
        assert laf.shape[3] == 3

    def test_precomputed_laf_bypass(self):
        """Scenario: Pre-computed LAFs bypass detection."""
        sift = KorniaSift(num_features=10)
        x = torch.randn(1, 1, 128, 128).abs()
        # First detect to get valid LAFs
        _, laf_detected = sift(x, laf=None)
        # Now pass the LAFs back - should skip detection
        desc2, laf2 = sift(x, laf=laf_detected)
        assert desc2.shape[2] == 128
        # The returned LAF should be the same we passed in
        assert torch.equal(laf2, laf_detected)

    def test_single_channel_assertion(self):
        """Scenario: Single-channel input is required (assert PC == 1)."""
        source = inspect.getsource(KorniaSift.forward)
        assert "assert(PC == 1)" in source or "assert PC == 1" in source, (
            "Expected assertion that PC == 1 in KorniaSift.forward"
        )

    def test_patch_size_32(self):
        """Scenario: Patch extraction uses fixed 32x32 patches (PS=32)."""
        sift = KorniaSift()
        # The SIFTDescriptor is initialized with patch_size=32
        assert sift.get_descriptor.patch_size == 32
