"""Tests for the Difference of Gaussians pyramid.

BDD Feature: Difference of Gaussians Pyramid (model/kornia_dog.py)

The DoG module feeds into both the pyramid correlation loss and SIFT
keypoint detection.  These tests verify the pyramid structure, the
adjacent-level subtraction, and spatial downscaling across octaves.
"""

import kornia
import torch
import pytest

from model.kornia_dog import KorniaDoG, KorniaDoGScalePyr


@pytest.fixture
def dog():
    """Default KorniaDoG with a 3-level ScalePyramid."""
    sp = kornia.geometry.transform.pyramid.ScalePyramid(
        n_levels=3, init_sigma=1.6, min_size=15,
    )
    return KorniaDoG(scale_pyramid=sp)


@pytest.fixture
def dog_scale_pyr():
    """KorniaDoGScalePyr variant."""
    sp = kornia.geometry.transform.pyramid.ScalePyramid(
        n_levels=3, init_sigma=1.6, min_size=15,
    )
    return KorniaDoGScalePyr(scale_pyramid=sp)


# -- Scenario: KorniaDoG produces multi-octave DoG outputs ----------------
class TestDoGMultiOctaveOutput:
    """BDD: Given (B,1,H,W) input · When passed through KorniaDoG ·
    Then it returns a list of DoG tensors, sigmas, distances, and pyramids.

    The number of octaves depends on the input size and min_size.  We
    just verify the return types and that we get at least 1 octave.
    """

    def test_return_structure(self, dog):
        x = torch.randn(1, 1, 64, 64)
        dogs, sigmas, dists, pyramids = dog(x)
        assert isinstance(dogs, list) and len(dogs) >= 1
        assert isinstance(sigmas, list) and len(sigmas) == len(dogs)
        assert isinstance(dists, list) and len(dists) == len(dogs)
        assert isinstance(pyramids, list) and len(pyramids) >= 1

    def test_dog_tensors_are_4d(self, dog):
        x = torch.randn(1, 1, 64, 64)
        dogs, _, _, _ = dog(x)
        for d in dogs:
            assert d.dim() == 4  # (B, n_levels-1, H, W) after squeeze


# -- Scenario: DoG is computed as difference of adjacent pyramid levels ----
class TestAdjacentLevelDifference:
    """BDD: Given n_levels=3 · When DoG is computed · Then each octave has
    n_levels - 1 = 2 DoG layers.

    The subtraction pyr[:,:,1:] - pyr[:,:,:-1] removes one level.
    """

    def test_dog_has_n_levels_minus_one(self, dog):
        x = torch.randn(1, 1, 64, 64)
        dogs, _, _, pyramids = dog(x)
        for d, pyr in zip(dogs, pyramids):
            # pyr shape: (B, 1, n_levels, H, W) — after squeeze the DoG
            # should have n_levels-1 along the level dimension.
            n_pyr_levels = pyr.shape[2]
            n_dog_layers = d.shape[1]
            assert n_dog_layers == n_pyr_levels - 1


# -- Scenario: KorniaDoGScalePyr uses different dimension ordering ---------
class TestScalePyrVariant:
    """BDD: Given the same input · When KorniaDoGScalePyr runs ·
    Then subtraction operates on dim=1 instead of dim=2, producing
    contiguous tensors.

    We verify the output is still a valid list of DoGs and that the
    tensor is contiguous (the source calls .contiguous()).
    """

    def test_output_is_contiguous(self, dog_scale_pyr):
        x = torch.randn(1, 1, 64, 64)
        dogs, sigmas, dists = dog_scale_pyr(x)
        for d in dogs:
            assert d.is_contiguous()

    def test_returns_three_items(self, dog_scale_pyr):
        x = torch.randn(1, 1, 64, 64)
        result = dog_scale_pyr(x)
        assert len(result) == 3  # dogs, sigmas, dists — no pyramids


# -- Scenario: Spatial dimensions decrease across octaves ------------------
class TestSpatialDownscaling:
    """BDD: Given a multi-octave pyramid · Then each successive octave has
    approximately half the spatial dimensions.

    We compare heights across consecutive octaves.
    """

    def test_octaves_shrink(self, dog):
        x = torch.randn(1, 1, 128, 128)
        dogs, _, _, _ = dog(x)
        if len(dogs) >= 2:
            h0 = dogs[0].shape[2]
            h1 = dogs[1].shape[2]
            # Second octave should be roughly half the first
            assert h1 <= h0 * 0.75
