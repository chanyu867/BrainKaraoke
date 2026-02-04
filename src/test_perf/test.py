#!/usr/bin/env python3
"""
Inference on TEST split (10%) using existing project code:
- main.py style TOML flags
- dataset.py get_data(split="test", hop=FLAGS.hop_in_ms)
- pipeline.py Model
- loads a Lightning checkpoint (.ckpt)
- saves predicted + ground-truth mels for each window

Run from repo root:
PYTHONPATH=./src python3 infer_test.py --config ./src/config/config.toml --ckpt_path ./checkpoints/xxx.ckpt --out_dir ./test_infer_out
"""
import logging
logger = logging.getLogger(__name__)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import argparse
import tomllib
import numpy as np
import torch
from torch.utils.data import DataLoader
from absl import flags
from src import dataset
from src.pipeline import Model

# -------------------------
# TOML -> absl FLAGS helpers
# -------------------------
def define_flags_from_toml(toml_path: str):
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    cfg_flags = cfg.get("flags", cfg)
    if not isinstance(cfg_flags, dict):
        raise ValueError("TOML must have a [flags] table (dict).")

    def _maybe_define(name, value):
        if name in flags.FLAGS:
            return
        if isinstance(value, bool):
            flags.DEFINE_bool(name, value, "")
        elif isinstance(value, int):
            flags.DEFINE_integer(name, value, "")
        elif isinstance(value, float):
            flags.DEFINE_float(name, value, "")
        elif isinstance(value, str):
            flags.DEFINE_string(name, value, "")
        elif isinstance(value, list):
            # define as list[str] for robustness
            flags.DEFINE_list(name, [str(v) for v in value], "")
        else:
            flags.DEFINE_string(name, str(value), "")

    for k, v in cfg_flags.items():
        _maybe_define(k, v)


def apply_toml_values_to_FLAGS(toml_path: str):
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    cfg_flags = cfg.get("flags", cfg)

    for k, v in cfg_flags.items():
        if k not in flags.FLAGS:
            continue
        fl = flags.FLAGS[k]
        # CLI overrides TOML
        if fl.present:
            continue

        if isinstance(v, list):
            setattr(flags.FLAGS, k, [str(x) for x in v])
        else:
            setattr(flags.FLAGS, k, v)


# -------------------------
# Main inference
# -------------------------
@torch.no_grad()
def run_inference(model, test_ds, FLAGS, out_dir: str, num_workers: int, limit_windows: int, save_wav: bool):
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(
        test_ds,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    device = next(model.parameters()).device
    model.eval()

    saved = 0
    for b_idx, (x, y_wav) in enumerate(loader):
        x = x.to(device)
        y_wav = y_wav.to(device)

        # --- GT mel (same as in pipeline training/val) ---
        if not FLAGS.use_MFCCs:
            raise ValueError("This inference script expects FLAGS.use_MFCCs=true (mel/MFCC path).")

        gt_mel = model.mel_transformer.mel_spectrogram(y_wav).transpose(1, 2)  # [B,T,80]

        # --- Prediction ---
        logits, attn, enc = model.seq2seq(x)
        pred_mel = logits  # regression output in your current setup

        # Save per-window examples
        B = pred_mel.shape[0]
        for i in range(B):
            if limit_windows > 0 and saved >= limit_windows:
                logger.info(f"[infer] Reached limit_windows={limit_windows}. Stopping.")
                return

            # Save as torch .pt to preserve exact tensors
            # Shapes saved:
            #   pred_mel: [T, 80]
            #   gt_mel:   [T, 80]
            item = {
                "pred_mel": pred_mel[i].detach().cpu().float(),
                "gt_mel": gt_mel[i].detach().cpu().float(),
            }

            if save_wav:
                item["gt_wav"] = y_wav[i].detach().cpu().float()

            out_path = os.path.join(out_dir, f"test_win_{saved:06d}.pt")
            torch.save(item, out_path)
            saved += 1

        if (b_idx + 1) % 10 == 0:
            logger.info(f"[infer] processed batches={b_idx+1}, saved_windows={saved}")

    logger.info(f"[infer] Done. Total saved_windows={saved} -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to config.toml (your project config)")
    p.add_argument("--ckpt_path", required=True, help="Path to best Lightning checkpoint (.ckpt)")
    p.add_argument("--out_dir", default="./test_infer_out", help="Output directory for saved .pt files")
    p.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 safest locally)")
    p.add_argument("--limit_windows", type=int, default=-1, help="Limit number of test windows saved (debug)")
    p.add_argument("--save_wav", action="store_true", help="Also save gt wav segment for each window")
    args, passthrough = p.parse_known_args()
    return args, passthrough


def main():
    args, passthrough = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    # Define flags before importing project modules inside ./src
    define_flags_from_toml(args.config)

    # Parse absl flags (allow overrides after the script args)
    flags.FLAGS(["infer_test.py"] + passthrough)

    # Apply TOML defaults for flags not overridden on CLI
    apply_toml_values_to_FLAGS(args.config)

    FLAGS = flags.FLAGS

    # Build datasets exactly like main.py
    train_ds = dataset.get_data(split="train", hop=FLAGS.hop_in_ms)
    val_ds   = dataset.get_data(split="val",   hop=FLAGS.hop_in_ms)
    test_ds  = dataset.get_data(split="test",  hop=FLAGS.hop_in_ms)

    num_classes = train_ds.num_audio_classes
    sampling_rate_audio = round(FLAGS.sampling_rate_eeg * train_ds.audio_eeg_sample_ratio)

    model = Model(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        num_classes=num_classes,
        sampling_rate_audio=sampling_rate_audio,
    )

    device = torch.device("cuda" if (getattr(FLAGS, "gpus", 0) and torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    run_inference(
        model=model,
        test_ds=test_ds,
        FLAGS=FLAGS,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        limit_windows=args.limit_windows,
        save_wav=args.save_wav,
    )


if __name__ == "__main__":
    main()
