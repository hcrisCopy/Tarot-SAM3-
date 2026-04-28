"""RefCOCO evaluation entrypoint placeholder."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tarot-SAM3 on RefCOCO-style datasets.")
    parser.add_argument("--config", default="configs/refcoco.yaml")
    parser.add_argument("--split", default="testA")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(
        "RefCOCO evaluation is pending. "
        f"Received config={args.config!r}, split={args.split!r}, max_samples={args.max_samples!r}."
    )


if __name__ == "__main__":
    main()

