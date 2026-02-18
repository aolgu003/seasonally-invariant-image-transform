"""Tests for SIFT descriptor extraction.

BDD Feature: SIFT Descriptor Extraction (model/kornia_sift.py)

KorniaSift wraps Kornia's ScaleSpaceDetector + SIFTDescriptor.  These
tests verify descriptor shape, detection vs pre-computed LAFs, the
single-channel assertion, and the fixed 32x32 patch size.
"""

import torch
import pytest

from model.kornia_sift import KorniaSift


@pytest.fixture
def sift():
    return KorniaSift(num_features=50)


# -- Scenario: Descriptors have 128 dimensions ----------------------------
class TestDescriptor128:
    """BDD: Given an input tensor with detected keypoints · When SIFT
    descriptors are extracted · Then each descriptor has 128 elements.

    The SIFT standard is 128-dim (4x4 spatial bins x 8 orientation bins).
    """

    def test_descriptor_dim(self, sift):
        x = torch.rand(1, 1, 64, 64)
        desc, laf = sift(x)
        assert desc.shape[2] == 128


# -- Scenario: Keypoints are detected when no LAF is provided -------------
class TestDetectionWithoutLAF:
    """BDD: Given laf=None · When forward is called · Then ScaleSpaceDetector
    runs and LAFs are returned alongside descriptors.

    We verify that laf is a tensor with shape (B, N, 2, 3).
    """

    def test_laf_returned(self, sift):
        x = torch.rand(1, 1, 64, 64)
        desc, laf = sift(x)
        assert laf.dim() == 4
        assert laf.shape[0] == 1
        assert laf.shape[2] == 2
        assert laf.shape[3] == 3


# -- Scenario: Pre-computed LAFs bypass detection --------------------------
class TestPrecomputedLAF:
    """BDD: Given a pre-computed laf tensor · When forward is called ·
    Then detection is skipped and descriptors are extracted at those
    LAF locations.

    We provide a synthetic LAF and check that the descriptor count
    matches the LAF count.
    """

    def test_uses_provided_laf(self, sift):
        x = torch.rand(1, 1, 64, 64)
        # Create a synthetic LAF: 5 keypoints
        n_kpts = 5
        laf = torch.zeros(1, n_kpts, 2, 3)
        # Identity-like LAF at various positions
        for i in range(n_kpts):
            laf[0, i, 0, 0] = 4.0  # scale
            laf[0, i, 1, 1] = 4.0
            laf[0, i, 0, 2] = 20.0 + i * 5  # x
            laf[0, i, 1, 2] = 20.0 + i * 5  # y

        desc, laf_out = sift(x, laf=laf)
        assert desc.shape[1] == n_kpts
        assert desc.shape[2] == 128


# -- Scenario: Single-channel input is required ----------------------------
class TestSingleChannelAssertion:
    """BDD: Given patches with PC != 1 · When descriptors are extracted ·
    Then an AssertionError is raised at line 53.

    The current code asserts PC == 1 after patch extraction.  Multi-channel
    inputs cause the patches to retain their channel count, triggering the
    assertion.  This test documents that restriction.
    """

    def test_multichannel_raises(self, sift):
        x = torch.rand(1, 3, 64, 64)
        with pytest.raises(AssertionError):
            sift(x)


# -- Scenario: Patch extraction uses fixed 32x32 patches ------------------
class TestPatchSize:
    """BDD: Given detected keypoints · When patches are extracted ·
    Then each patch is 32x32 (PS=32 hardcoded in forward).

    We verify indirectly: the SIFTDescriptor is initialised with
    patch_size=32, so it expects 32x32 input.  If PS changed, the
    descriptor would fail or produce wrong shapes.
    """

    def test_sift_descriptor_patch_size(self, sift):
        assert sift.get_descriptor.patch_size == 32
