#!/usr/bin/env python3
"""
Preprocess raw sEEG .npy into high-gamma envelope (or your project's high-gamma preprocessing).

Usage:
  python preprocess.py --input /path/raw.npy --output /path/processed.npy --fs 1024
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
import argparse
from pathlib import Path

import numpy as np
from src.preprocess.filter import preprocess_high_gamma


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess raw sEEG (.npy) into processed high-gamma (.npy).")
    p.add_argument("--input", "-i", required=True, help="Path to raw EEG .npy")
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save processed .npy (default: <input>_processed.npy)",
    )
    p.add_argument("--fs", type=int, default=1024, help="Sampling rate (default: 1024)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    return p.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_processed.npy")


def run(input_path: Path, output_path: Path, fs: int, overwrite: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            f"Use --overwrite or choose a different --output."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw data from: {input_path}")
    raw = np.load(str(input_path))
    logger.info(f"Raw shape: {raw.shape} dtype={raw.dtype}")

    logger.info(f"Running preprocess_high_gamma(fs={fs}) ...")
    clean = preprocess_high_gamma(raw, fs=fs)
    logger.info(f"Processed shape: {clean.shape} dtype={clean.dtype}")

    np.save(str(output_path), clean)
    logger.info(f"✅ Saved processed file to: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    run(input_path=input_path, output_path=output_path, fs=args.fs, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
