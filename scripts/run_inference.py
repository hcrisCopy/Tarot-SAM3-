"""Single-image inference entrypoint placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tarot-SAM3 on one image.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default="outputs/visualizations/single_image.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "Pipeline implementation is pending. "
        f"Received image={args.image!r}, query={args.query!r}, config={args.config!r}."
    )


if __name__ == "__main__":
    main()

