"""Tests for model/kornia_dog.py

Maps to BDD Feature: Difference of Gaussians Pyramid.
"""

import kornia
import torch

from model.kornia_dog import KorniaDoG, KorniaDoGScalePyr


class TestKorniaDoG:
    """Tests for the KorniaDoG module."""

    def test_multi_octave_output_structure(self):
        """Scenario: KorniaDoG produces multi-octave DoG outputs.
        Returns (dogs, sigmas, dists, pyramids) where dogs is a list."""
        dog = KorniaDoG()
        x = torch.randn(1, 1, 128, 128)
        dogs, sigmas, dists, pyramids = dog(x)
        assert isinstance(dogs, list)
        assert len(dogs) > 0
        assert isinstance(sigmas, list)
        assert isinstance(dists, list)
        assert isinstance(pyramids, list)

    def test_dog_is_difference_of_adjacent_levels(self):
        """Scenario: DoG is computed as difference of adjacent levels.
        The code computes pyr[:,:,1:,:,:] - pyr[:,:,:-1,:,:], so the
        number of DoG layers per octave is (num_pyramid_levels - 1).
        Kornia's ScalePyramid(n_levels=N) produces N+3 levels per octave,
        yielding N+2 DoG layers after subtraction."""
        n_levels = 3
        scale_pyr = kornia.geometry.transform.pyramid.ScalePyramid(n_levels=n_levels)
        dog = KorniaDoG(scale_pyramid=scale_pyr)
        x = torch.randn(1, 1, 128, 128)
        dogs, _, _, pyramids = dog(x)

        # Verify DoG layers = pyramid_levels_per_octave - 1
        for d, pyr in zip(dogs, pyramids):
            pyr_levels = pyr.shape[2]  # (B, C, levels, H, W)
            assert d.shape[1] == pyr_levels - 1, (
                f"DoG layers {d.shape[1]} should be pyramid levels {pyr_levels} - 1"
            )

    def test_spatial_dimensions_decrease_across_octaves(self):
        """Scenario: Spatial dimensions decrease across octaves."""
        dog = KorniaDoG()
        x = torch.randn(1, 1, 256, 256)
        dogs, _, _, _ = dog(x)
        if len(dogs) > 1:
            # Each successive octave should have smaller spatial dims
            for i in range(len(dogs) - 1):
                h_curr = dogs[i].shape[-2]
                h_next = dogs[i + 1].shape[-2]
                assert h_next < h_curr, (
                    f"Octave {i+1} spatial dim {h_next} not smaller than octave {i} dim {h_curr}"
                )

    def test_output_tensors_are_valid(self):
        """DoG output tensors contain no NaN values."""
        dog = KorniaDoG()
        x = torch.randn(1, 1, 128, 128)
        dogs, _, _, _ = dog(x)
        for d in dogs:
            assert not torch.isnan(d).any()

    def test_batch_support(self):
        """DoG handles batch_size > 1."""
        dog = KorniaDoG()
        x = torch.randn(2, 1, 128, 128)
        dogs, _, _, _ = dog(x)
        for d in dogs:
            assert d.shape[0] == 2


class TestKorniaDoGScalePyr:
    """Tests for the KorniaDoGScalePyr variant."""

    def test_scale_pyr_variant_returns_three_elements(self):
        """Scenario: KorniaDoGScalePyr uses different dimension ordering.
        Returns (dogs, sigmas, dists) - no pyramids."""
        dog_sp = KorniaDoGScalePyr()
        x = torch.randn(1, 1, 128, 128)
        result = dog_sp(x)
        assert len(result) == 3  # dogs, sigmas, dists (no pyramids)
        dogs, sigmas, dists = result
        assert isinstance(dogs, list)
        assert len(dogs) > 0
