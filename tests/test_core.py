from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visual_path.core import FeatureIndex, export_sequence, find_path, pairwise_distance


def make_collection(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32), "red").save(images / "red.png")
    Image.new("RGB", (32, 32), "green").save(images / "green.png")
    Image.new("RGB", (32, 32), "blue").save(images / "blue.png")
    return images


def test_build_indexes_supported_images_and_caches(tmp_path):
    images = make_collection(tmp_path)
    index = FeatureIndex(images, tmp_path / "cache").build()

    assert len(index.paths) == 3
    assert index.color.shape == (3, 48)
    assert index.texture.shape == (3, 16)
    assert index.shape.shape == (3, 256)
    assert index.semantic.shape == (3, 384)
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 3

    rebuilt = FeatureIndex(images, tmp_path / "cache").build()
    assert rebuilt.color.shape == (3, 48)
    assert np.allclose(index.color, rebuilt.color)


def test_find_path_has_no_cycles_and_includes_endpoints(tmp_path):
    images = make_collection(tmp_path)
    index = FeatureIndex(images).build()

    route = find_path(index, 0, 2, sequence_length=3)

    assert route[0] == 0
    assert route[-1] == 2
    assert len(route) == len(set(route))


def test_pairwise_distance_is_symmetric(tmp_path):
    images = make_collection(tmp_path)
    index = FeatureIndex(images).build()

    distance = pairwise_distance(index, (0.25, 0.25, 0.25, 0.25))

    assert distance.shape == (3, 3)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0)


def test_export_creates_frames_gif_and_originals(tmp_path):
    images = make_collection(tmp_path)
    index = FeatureIndex(images).build()
    output = tmp_path / "export"

    created = export_sequence(index, [0, 1, 2], output, fps=4)

    assert (output / "visual-path.gif").exists()
    assert len(list((output / "frames").glob("*.png"))) == 3
    assert len(list((output / "originals").glob("*"))) == 3
    assert len(created) >= 7


def test_invalid_image_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        FeatureIndex(tmp_path / "missing").build()
