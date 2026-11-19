# visual-path
Visual Path is a small, local-first creative tool for finding visual paths through
your own image collection. It combines DINOv2 semantic embeddings (optional) with
conventional color, texture, and shape features. No frames are generated or
morphed: every sequence frame is an original image from the imported archive.

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[ui,video,dev]"
```

For CPU-only or Apple Silicon machines, the conventional features work without
PyTorch. To enable DINOv2:

```bash
python3 -m pip install -e ".[ui,torch,video]"
```

## Run the local interface

```bash
visual-path --images /path/to/images --dino
```

Omit `--dino` for lower-memory conventional indexing. Open the local URL printed
by Gradio (normally `http://127.0.0.1:7860`).

## Command-line workflow

```bash
python3 -m visual_path.cli /path/to/images --start 0 --end 20 --length 10 --export output
```

The CLI prints the route and writes PNG frames, copied originals, a GIF, and MP4
output when FFmpeg support is available.

## Tests

```bash
python3 -m pytest -q
```

Features are cached under `<image directory>/.visual-path-cache` or in a directory
chosen with `--cache`. The cache is designed to avoid reprocessing unchanged
files.

## Limitations

- DINOv2 Small downloads model weights when first enabled.
- Large collections require more memory; start with a few hundred images.
- Conventional shape, color, and texture descriptors are intentionally simple.
- MP4 export depends on optional `imageio-ffmpeg`; PNG and GIF exports require no
  additional export dependency.
- Leafsnap is suitable for local validation but is not included in this repository.
