"""Tests for tile creation preprocessing.

BDD Feature: Tile Creation (createTiledDataset.py)

The createTiles function crops large images into fixed-size tiles for
training.  These tests verify geometry, overlap, edge handling, filename
encoding, the grayscale crash bug, and auto-directory creation.
"""

import glob
import os

import numpy as np
import pytest
from PIL import Image

from createTiledDataset import createTiles


def _write_image(path, size, mode="RGB"):
    """Write a test image of the given (width, height) size."""
    img = Image.new(mode, size, color=(128, 128, 128) if mode == "RGB" else 128)
    img.save(path)


# -- Scenario: Non-overlapping tiles cover the image -----------------------
class TestNonOverlappingTiles:
    """BDD: Given a 1800x1200 image and crop 600x600 with overlap=0 ·
    When tiles are created · Then 6 tiles are produced (2 rows x 3 cols).

    The while-loops step by crop_width*(1-0) = crop_width, so we get
    floor(1200/600) * floor(1800/600) = 2*3 = 6 tiles.
    """

    def test_tile_count(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        _write_image(str(src / "test.png"), (1800, 1200))

        createTiles(str(src), str(out), crop_width=600, crop_height=600, overlap_ratio=0)
        tiles = glob.glob(str(out / "*.png"))
        assert len(tiles) == 6


# -- Scenario: Overlapping tiles increase tile count -----------------------
class TestOverlappingTiles:
    """BDD: Given overlap_ratio=0.5 · When tiles are created ·
    Then more tiles are produced because step = crop_width * 0.5.
    """

    def test_more_tiles_with_overlap(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        _write_image(str(src / "test.png"), (1200, 1200))

        createTiles(str(src), str(out), crop_width=600, crop_height=600, overlap_ratio=0.5)
        tiles = glob.glob(str(out / "*.png"))
        # With overlap=0.5, step=300.  Positions: 0, 300, 600 → 3 per axis → 9
        assert len(tiles) == 9


# -- Scenario: Edge pixels are discarded if not tile-aligned ---------------
class TestEdgeDiscard:
    """BDD: Given a 700x700 image and crop 600x600 · When tiles are created ·
    Then only 1 tile is produced (remaining 100px strip is discarded).

    The loop condition `curr_w + crop_width <= w` prevents partial tiles.
    """

    def test_single_tile(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        _write_image(str(src / "test.png"), (700, 700))

        createTiles(str(src), str(out), crop_width=600, crop_height=600, overlap_ratio=0)
        tiles = glob.glob(str(out / "*.png"))
        assert len(tiles) == 1


# -- Scenario: Tile filenames encode source image and index ----------------
class TestTileFilenames:
    """BDD: Given source image foo.png · When tiles are created ·
    Then tiles are named foo_000000.png, foo_000001.png, etc.
    """

    def test_filename_format(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        _write_image(str(src / "foo.png"), (1200, 600))

        createTiles(str(src), str(out), crop_width=600, crop_height=600, overlap_ratio=0)
        tiles = sorted(os.listdir(str(out)))
        assert tiles[0] == "foo_000000.png"
        assert tiles[1] == "foo_000001.png"


# -- Scenario: Grayscale images crash (known bug) --------------------------
class TestGrayscaleCrash:
    """BDD: Given a grayscale PNG loaded as 2D array · When shape is
    unpacked as h, w, c · Then ValueError is raised.

    The code does `h, w, c = np.asarray(img).shape` which fails for
    2D arrays (grayscale with no channel dim).  This test documents the
    bug; update it to test the fix once implemented.
    """

    def test_grayscale_raises(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        _write_image(str(src / "gray.png"), (600, 600), mode="L")

        with pytest.raises(ValueError):
            createTiles(str(src), str(out), crop_width=300, crop_height=300, overlap_ratio=0)


# -- Scenario: Output directory is created automatically -------------------
class TestOutputDirCreation:
    """BDD: Given save_data_dir does not exist · When the main block runs ·
    Then the directory is created via os.makedirs(exist_ok=True).

    Note: createTiles itself does NOT create the directory — the __main__
    block does.  We test the function with a pre-existing directory.
    """

    def test_function_works_with_existing_dir(self, tmp_path):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        _write_image(str(src / "img.png"), (600, 600))
        # Should not crash
        createTiles(str(src), str(out), crop_width=600, crop_height=600, overlap_ratio=0)
        assert len(os.listdir(str(out))) == 1
