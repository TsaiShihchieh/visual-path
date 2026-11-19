from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def conventional_features(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    # Independent low-memory perceptual descriptors.
    hist = np.stack([
        np.histogram(rgb[..., channel], bins=16, range=(0, 256))[0]
        for channel in range(3)
    ]).astype(np.float32).reshape(-1)
    hist = _l2_normalize(hist)

    gray = np.asarray(image.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    magnitude = np.sqrt(gx * gx + gy * gy)
    texture = np.histogram(magnitude, bins=16, range=(0, 256))[0].astype(np.float32)
    texture = _l2_normalize(texture)

    dx = np.diff(gray, axis=1, prepend=gray[:, :1])
    dy = np.diff(gray, axis=0, prepend=gray[:1, :])
    edge = np.sqrt(dx * dx + dy * dy)
    edge = (edge > 18).astype(np.float32)
    shape = np.resize(edge, (16, 16)).astype(np.float32)
    shape = _l2_normalize(shape.reshape(-1))
    return hist, texture, shape


class FeatureIndex:
    def __init__(self, directory: str | Path, cache_dir: str | Path | None = None,
                 use_dino: bool = False) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)
        self.cache_dir = Path(cache_dir) if cache_dir else self.directory / ".visual-path-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_dino = use_dino
        self.paths: list[Path] = []
        self.color: np.ndarray = np.empty((0, 48), dtype=np.float32)
        self.texture: np.ndarray = np.empty((0, 16), dtype=np.float32)
        self.shape: np.ndarray = np.empty((0, 256), dtype=np.float32)
        self.semantic: np.ndarray = np.empty((0, 384), dtype=np.float32)

    def build(self, force: bool = False) -> "FeatureIndex":
        self.paths = image_files(self.directory)
        colors, textures, shapes, semantics = [], [], [], []
        dino_model = None
        dino_processor = None
        if self.use_dino:
            try:
                from transformers import AutoImageProcessor, AutoModel
                import torch
                dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
                dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
                dino_model.eval()
            except Exception as exc:
                raise RuntimeError(
                    "DINOv2 requires the optional torch and transformers dependencies."
                ) from exc

        for path in self.paths:
            mode = "dino" if dino_model is not None else "conventional"
            relative = path.relative_to(self.directory).as_posix().replace("/", "__")
            cache_path = self.cache_dir / (
                relative + "." + str(path.stat().st_mtime_ns) + "." + mode + ".npz"
            )
            cache_path = cache_path.with_suffix(".npz")
            if not force and cache_path.exists():
                data = np.load(cache_path)
                colors.append(data["color"])
                textures.append(data["texture"])
                shapes.append(data["shape"])
                semantics.append(data["semantic"])
                continue

            with Image.open(path) as source:
                image = source.convert("RGB")
                color, texture, shape = conventional_features(image)
                semantic = np.zeros(384, dtype=np.float32)
                if dino_model is not None:
                    import torch
                    inputs = dino_processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        output = dino_model(**inputs)
                    semantic = _l2_normalize(
                        output.last_hidden_state[:, 0].cpu().numpy()[0].astype(np.float32)
                    )
                    semantic = semantic[:384]

            np.savez_compressed(cache_path, color=color, texture=texture, shape=shape, semantic=semantic)
            colors.append(color)
            textures.append(texture)
            shapes.append(shape)
            semantics.append(semantic)

        if not self.paths:
            return self
        self.color = np.vstack(colors)
        self.texture = np.vstack(textures)
        self.shape = np.vstack(shapes)
        self.semantic = np.vstack(semantics)
        return self

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            paths=np.asarray([str(p) for p in self.paths]),
            color=self.color,
            texture=self.texture,
            shape=self.shape,
            semantic=self.semantic,
        )

    def weights(self, shape: float = 1.0, color: float = 1.0,
                texture: float = 1.0, semantic: float = 1.0) -> np.ndarray:
        total = shape + color + texture + semantic
        if total <= 0:
            raise ValueError("At least one similarity weight must be positive")
        values = np.asarray([shape, color, texture, semantic], dtype=np.float32)
        return values / total


def pairwise_distance(index: FeatureIndex, weights: Sequence[float]) -> np.ndarray:
    if len(index.paths) == 0:
        return np.empty((0, 0), dtype=np.float32)
    descriptors = [index.shape, index.color, index.texture, index.semantic]
    distance = np.zeros((len(index.paths), len(index.paths)), dtype=np.float32)
    for descriptor, weight in zip(descriptors, weights):
        normalized = descriptor / np.maximum(
            np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-8
        )
        similarity = normalized @ normalized.T
        distance += float(weight) * np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def nearest_neighbors(distance: np.ndarray, count: int = 12) -> list[list[int]]:
    graph: list[list[int]] = []
    count = max(1, count)
    for row in distance:
        order = np.argsort(row)
        graph.append([])
    for i, row in enumerate(distance):
        graph[i] = [int(j) for j in np.argsort(row) if j != i][:count]
    return graph


def find_path(index: FeatureIndex, start: int, end: int, sequence_length: int = 10,
              weights: Sequence[float] | None = None, neighbor_count: int = 12) -> list[int]:
    if len(index.paths) < 2:
        raise ValueError("Index at least two images")
    if not 0 <= start < len(index.paths) or not 0 <= end < len(index.paths):
        raise ValueError("Start and end indices are outside the collection")
    if start == end:
        return [start]
    sequence_length = max(2, min(sequence_length, len(index.paths)))
    if weights is None:
        weights = (0.25, 0.25, 0.25, 0.25)
    distance = pairwise_distance(index, weights)
    graph = nearest_neighbors(distance, neighbor_count)
    graph[start].append(end)
    graph[end].append(start)

    states: dict[int, tuple[float, list[int]]] = {start: (0.0, [start])}
    for step in range(1, sequence_length):
        next_states: dict[int, tuple[float, list[int]]] = {}
        for current, (score, route) in states.items():
            candidates = graph[current] if current != end else [end]
            for candidate in candidates:
                if candidate in route:
                    continue
                progress = distance[candidate, end]
                combined = score + float(distance[current, candidate]) + 0.15 * float(progress)
                if candidate not in next_states or combined < next_states[candidate][0]:
                    next_states[candidate] = (combined, route + [candidate])
        states = next_states
        if end in states:
            if len(states[end][1]) == sequence_length:
                return states[end][1]

    if end in states:
        return states[end][1]
    return [start, end]


def export_sequence(index: FeatureIndex, sequence: Iterable[int], output_dir: str | Path,
                    include_originals: bool = True, fps: int = 8) -> list[Path]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = index.directory.resolve()
    created: list[Path] = []
    frames: list[Image.Image] = []
    for position, index_number in enumerate(sequence, 1):
        if not 0 <= index_number < len(index.paths):
            raise IndexError("Sequence contains an image index outside the collection")
        source = index.paths[index_number].resolve()
        if not source.is_relative_to(source_root):
            raise ValueError("Image is outside the indexed directory")
        if include_originals:
            original_copy = output / "originals" / f"{position:04d}_{source.name}"
            original_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, original_copy)
            created.append(original_copy)
        with Image.open(source) as image:
            frame = image.convert("RGB").copy()
        png_path = output / "frames" / f"{position:04d}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        frame.save(png_path)
        created.append(png_path)
        frames.append(frame)

    if frames:
        gif_path = output / "visual-path.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000 // fps,
                        loop=0)
        created.append(gif_path)
        try:
            import imageio.v3 as iio
            mp4_path = output / "visual-path.mp4"
            iio.imwrite(mp4_path, [np.asarray(frame) for frame in frames],
                        fps=fps, plugin="pyav")
            created.append(mp4_path)
        except Exception:
            pass
    return created
