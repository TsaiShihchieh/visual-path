from __future__ import annotations

from pathlib import Path
import argparse

from .core import FeatureIndex, export_sequence, find_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual Path local image explorer")
    parser.add_argument("--images", default=".", help="Directory containing source images")
    parser.add_argument("--cache", default=None, help="Feature cache directory")
    parser.add_argument("--dino", action="store_true", help="Use DINOv2 semantic embeddings")
    parser.add_argument("--share", action="store_true", help="Create a Gradio share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Install the UI extra: pip install 'visual-path[ui]'") from exc

    index = FeatureIndex(args.images, args.cache, use_dino=args.dino)
    index.build()

    def gallery_path(start_name: str, end_name: str, length: int, shape: float,
                     color: float, texture: float, semantic: float):
        if not index.paths:
            return None, "No images found"
        start = next((i for i, p in enumerate(index.paths) if p.name == start_name), 0)
        end = next((i for i, p in enumerate(index.paths) if p.name == end_name), len(index.paths) - 1)
        weights = index.weights(shape, color, texture, semantic)
        route = find_path(index, start, end, int(length), weights)
        return [str(index.paths[i]) for i in route], " → ".join(index.paths[i].name for i in route)

    names = [p.name for p in index.paths]
    with gr.Blocks(title="Visual Path") as demo:
        gr.Markdown("# Visual Path\nFind perceptual paths between images in a local collection.")
        start = gr.Dropdown(names, label="Start image")
        end = gr.Dropdown(names, label="End image")
        length = gr.Slider(2, max(2, len(names)), value=min(10, len(names)), step=1,
                           label="Desired sequence length")
        fps = gr.Slider(1, 30, value=8, step=1, label="Preview/export frame rate")
        shape = gr.Slider(0, 1, value=0.25, step=0.05, label="Shape")
        color = gr.Slider(0, 1, value=0.25, step=0.05, label="Color")
        texture = gr.Slider(0, 1, value=0.25, step=0.05, label="Texture")
        semantic = gr.Slider(0, 1, value=0.25, step=0.05, label="Semantic")
        find = gr.Button("Find visual path")
        gallery = gr.Gallery(label="Sequence", columns=5)
        route_text = gr.Textbox(label="Route")
        export_dir = gr.Textbox(value="visual-path-export", label="Export directory")
        export_button = gr.Button("Export PNG sequence, originals, GIF, and MP4")
        export_status = gr.Textbox(label="Export status")
        find.click(gallery_path, [start, end, length, shape, color, texture, semantic],
                   [gallery, route_text])

        def export_path(start_name: str, end_name: str, length: int, fps_value: float,
                         shape_value: float, color_value: float, texture_value: float,
                         semantic_value: float, output_name: str):
            if not index.paths:
                return "No images found"
            start_index = next((i for i, p in enumerate(index.paths) if p.name == start_name), 0)
            end_index = next(
                (i for i, p in enumerate(index.paths) if p.name == end_name),
                len(index.paths) - 1,
            )
            weights = index.weights(shape_value, color_value, texture_value, semantic_value)
            route = find_path(index, start_index, end_index, int(length), weights)
            created = export_sequence(
                index, route, output_name, include_originals=True, fps=max(1, int(fps_value))
            )
            return f"Exported {len(created)} files to {Path(output_name).resolve()}"

        export_button.click(
            export_path,
            [start, end, length, fps, shape, color, texture, semantic, export_dir],
            export_status,
        )
    demo.launch(server_port=args.port, share=args.share)
