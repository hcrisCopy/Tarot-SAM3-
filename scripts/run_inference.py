"""Run Tarot-SAM3 on one image."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tarot_sam3.pipeline import SingleImagePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tarot-SAM3 on one image.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default="outputs/visualizations/single_image.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = SingleImagePipeline(args.config)
    result = pipeline.run(args.image, args.query, args.output)
    print(f"Saved visualization: {result.output_path}")
    print(f"Saved intermediate JSON: {result.json_path}")


if __name__ == "__main__":
    main()
