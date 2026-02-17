"""Tests for createTiledDataset.py

Maps to BDD Feature: Tile Creation.
"""

import glob
import os

import numpy as np
import pytest
from PIL import Image

from createTiledDataset import createTiles


class TestCreateTiledDataset:
    """Tests for the tile creation utility."""

    def test_non_overlapping_tile_count(self, tmp_path):
        """Scenario: Non-overlapping tiles cover the image.
        A 1200x1800 image with 600x600 crops and 0 overlap -> 2*3 = 6 tiles."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        # Create a 1200x1800 RGB image (3-channel to avoid the grayscale bug)
        arr = np.random.RandomState(42).randint(0, 256, (1200, 1800, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(src_dir / "test.png")
        createTiles(str(src_dir), str(out_dir), crop_width=600, crop_height=600, overlap_ratio=0)
        tiles = glob.glob(os.path.join(str(out_dir), "*.png"))
        assert len(tiles) == 6

    def test_overlapping_tiles_increase_count(self, tmp_path):
        """Scenario: Overlapping tiles increase tile count."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        arr = np.random.RandomState(42).randint(0, 256, (1200, 1200, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(src_dir / "test.png")

        # Non-overlapping: 1200/600 = 2 per dim -> 4 tiles
        createTiles(str(src_dir), str(out_dir), crop_width=600, crop_height=600, overlap_ratio=0)
        no_overlap_count = len(glob.glob(os.path.join(str(out_dir), "*.png")))

        out_dir2 = tmp_path / "out2"
        out_dir2.mkdir()
        # 50% overlap: step = 600 * 0.5 = 300. Positions: 0, 300, 600 -> 3 per dim -> 9 tiles
        createTiles(str(src_dir), str(out_dir2), crop_width=600, crop_height=600, overlap_ratio=0.5)
        overlap_count = len(glob.glob(os.path.join(str(out_dir2), "*.png")))

        assert overlap_count > no_overlap_count

    def test_edge_pixels_discarded(self, tmp_path):
        """Scenario: Edge pixels are discarded if not tile-aligned.
        A 700x700 image with 600x600 crops -> only 1 tile."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        arr = np.random.RandomState(42).randint(0, 256, (700, 700, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(src_dir / "test.png")
        createTiles(str(src_dir), str(out_dir), crop_width=600, crop_height=600, overlap_ratio=0)
        tiles = glob.glob(os.path.join(str(out_dir), "*.png"))
        assert len(tiles) == 1

    def test_tile_filename_format(self, tmp_path):
        """Scenario: Tile filenames encode source image and index.
        Tiles named foo_000000.png, foo_000001.png, etc."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        arr = np.random.RandomState(42).randint(0, 256, (600, 1200, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(src_dir / "myimage.png")
        createTiles(str(src_dir), str(out_dir), crop_width=600, crop_height=600, overlap_ratio=0)
        tiles = sorted(glob.glob(os.path.join(str(out_dir), "*.png")))
        basenames = [os.path.basename(t) for t in tiles]
        assert "myimage_000000.png" in basenames
        assert "myimage_000001.png" in basenames

    def test_grayscale_crash_known_bug(self, tmp_path):
        """Scenario: Grayscale images crash (known bug).
        np.asarray(img).shape unpacked as (h, w, c) raises ValueError
        on a 2D grayscale array."""
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()
        # Create a grayscale image (2D array, no channel dimension)
        arr = np.random.RandomState(42).randint(0, 256, (64, 64), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(src_dir / "gray.png")
        with pytest.raises((ValueError, Exception)):
            createTiles(str(src_dir), str(out_dir), crop_width=32, crop_height=32, overlap_ratio=0)

    def test_output_dir_must_exist(self, tmp_path):
        """The createTiles function writes to save_dir which must exist.
        The main script creates it via os.makedirs, but the function
        itself does not create it."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        arr = np.random.RandomState(42).randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(src_dir / "test.png")

        nonexistent = str(tmp_path / "nonexistent_out")
        # Saving to a nonexistent directory should raise
        with pytest.raises((FileNotFoundError, OSError)):
            createTiles(str(src_dir), nonexistent, crop_width=32, crop_height=32, overlap_ratio=0)
