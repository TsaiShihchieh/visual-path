from __future__ import annotations

import argparse
from pathlib import Path

from .core import FeatureIndex, export_sequence, find_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index images and find a visual path")
    parser.add_argument("images", help="Image directory")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--length", type=int, default=10)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--export", default=None)
    args = parser.parse_args()

    index = FeatureIndex(args.images, args.cache).build()
    if len(index.paths) < 2:
        raise SystemExit("Need at least two images")
    end = args.end if args.end is not None else len(index.paths) - 1
    route = find_path(index, args.start, end, args.length)
    print("visual-path:")
    for item in route:
        print(index.paths[item])
    if args.export:
        export_sequence(index, route, args.export)
        print(f"Exported to {Path(args.export)}")


if __name__ == "__main__":
    main()
