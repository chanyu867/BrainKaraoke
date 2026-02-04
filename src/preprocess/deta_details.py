#!/usr/bin/env python3
"""
Sanity checks utilities:

1) CAR sanity checks:
   - per-channel variance before/after CAR
   - mean pairwise correlation before/after CAR
   - line-noise power ratio around 50Hz / 60Hz (band/1–200Hz) before/after CAR

2) Audio sample-rate forensics:
   - estimate audio sample rate from (audio_samples / eeg_samples) * eeg_rate

This is a refactor of the attached deta_details.py, keeping the same behavior but simpler/cleaner.
Reference: :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import os
import numpy as np


# ---------------------------
# Loading / shape helpers
# ---------------------------

def load_eeg_time_channels(path: str) -> np.ndarray:
    """Load EEG .npy and ensure shape is (time, channels)."""
    eeg = np.load(path)
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D EEG array, got shape {eeg.shape}")

    # Heuristic: time dimension is usually larger than channels
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T
    return eeg


def get_time_length_from_npy(path: str) -> int:
    """
    For quick length checks (mmap), return the largest dimension as time.
    Works for 1D audio or 2D eeg (time, channels) / (channels, time).
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim == 2:
        return int(max(arr.shape))
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


# ---------------------------
# CAR + metrics
# ---------------------------

def apply_car(data_tc: np.ndarray) -> np.ndarray:
    """
    Common average reference (CAR):
    subtract the mean across channels at each time point.
    data_tc: (time, channels)
    """
    return data_tc - data_tc.mean(axis=1, keepdims=True)


def per_channel_variance(data_tc: np.ndarray) -> np.ndarray:
    """Variance per channel over time."""
    return data_tc.var(axis=0)


def mean_pairwise_corr(data_tc: np.ndarray, max_channels: int = 64, seed: int = 0) -> float:
    """
    Mean off-diagonal correlation across channels.
    To keep compute manageable, subsample channels if C > max_channels.
    """
    rng = np.random.default_rng(seed)
    T, C = data_tc.shape

    if C > max_channels:
        idx = rng.choice(C, size=max_channels, replace=False)
        data_tc = data_tc[:, idx]

    R = np.corrcoef(data_tc, rowvar=False)  # (C, C)
    mask = ~np.eye(R.shape[0], dtype=bool)
    return float(np.nanmean(R[mask]))


def line_noise_power_ratio(
    data_tc: np.ndarray,
    fs: float,
    f0: float,
    bw: float = 1.0,
    max_channels: int = 64,
    seed: int = 0,
) -> float:
    """
    Average over channels of:
      bandpower around f0 (+/- bw) divided by total power in 1–200 Hz.

    Uses rFFT power (periodogram-like). Matches your original logic.
    """
    rng = np.random.default_rng(seed)
    T, C = data_tc.shape

    if C > max_channels:
        idx = rng.choice(C, size=max_channels, replace=False)
        data_tc = data_tc[:, idx]

    # Remove DC per-channel to reduce leakage
    d = data_tc - data_tc.mean(axis=0, keepdims=True)

    X = np.fft.rfft(d, axis=0)             # (F, C)
    Pxx = (np.abs(X) ** 2) / max(T, 1)     # (F, C)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    line_mask = (freqs >= (f0 - bw)) & (freqs <= (f0 + bw))
    total_mask = (freqs >= 1.0) & (freqs <= 200.0)

    line = Pxx[line_mask, :].sum(axis=0)
    total = Pxx[total_mask, :].sum(axis=0) + 1e-12

    return float(np.mean(line / total))


def summarize_array(a: np.ndarray, name: str) -> None:
    a = np.asarray(a)
    print(
        f"{name}: mean={a.mean():.4g}  median={np.median(a):.4g}  "
        f"p05={np.percentile(a, 5):.4g}  p95={np.percentile(a, 95):.4g}  "
        f"min={a.min():.4g}  max={a.max():.4g}"
    )


def run_car_checks(
    eeg_path: str,
    fs: int,
    seconds: int,
    max_channels: int,
    seed: int,
    bw: float,
) -> None:
    eeg = load_eeg_time_channels(eeg_path)
    print("Loaded EEG:", eeg.shape, eeg.dtype)

    T, C = eeg.shape
    print(f"T={T} samples, C={C} channels")

    N = min(T, fs * seconds)
    x = eeg[:N].astype(np.float64)  # stable stats
    print("Using segment:", x.shape, f"({N/fs:.1f} s)")

    x_car = apply_car(x)

    var_before = per_channel_variance(x)
    var_after = per_channel_variance(x_car)
    ratio = (var_after + 1e-12) / (var_before + 1e-12)

    corr_before = mean_pairwise_corr(x, max_channels=max_channels, seed=seed)
    corr_after = mean_pairwise_corr(x_car, max_channels=max_channels, seed=seed)

    ln50_before = line_noise_power_ratio(x, fs=fs, f0=50.0, bw=bw, max_channels=max_channels, seed=seed)
    ln50_after = line_noise_power_ratio(x_car, fs=fs, f0=50.0, bw=bw, max_channels=max_channels, seed=seed)

    ln60_before = line_noise_power_ratio(x, fs=fs, f0=60.0, bw=bw, max_channels=max_channels, seed=seed)
    ln60_after = line_noise_power_ratio(x_car, fs=fs, f0=60.0, bw=bw, max_channels=max_channels, seed=seed)

    print("\n=== Per-channel variance ===")
    summarize_array(var_before, "var_before")
    summarize_array(var_after, "var_after")

    print("\n=== Variance ratio (after/before) ===")
    summarize_array(ratio, "var_ratio")
    print("Channels with variance >2x after CAR :", int(np.sum(ratio > 2.0)), "/", C)
    print("Channels with variance <0.5x after CAR:", int(np.sum(ratio < 0.5)), "/", C)

    print("\n=== Mean pairwise correlation (subsampled) ===")
    print("corr_before:", corr_before)
    print("corr_after :", corr_after)
    print("delta_corr :", corr_after - corr_before)

    print("\n=== Line-noise power ratio (band/1–200Hz), subsampled ===")
    print(f"50Hz before={ln50_before:.6f}  after={ln50_after:.6f}  change={(ln50_after-ln50_before):.6f}")
    print(f"60Hz before={ln60_before:.6f}  after={ln60_after:.6f}  change={(ln60_after-ln60_before):.6f}")

    print("\n=== Quick interpretation heuristics ===")
    if (ln50_after < ln50_before) or (ln60_after < ln60_before):
        print("✅ Line-noise ratio decreased for at least one of 50/60 Hz (CAR may help denoise).")
    else:
        print("⚠️ Line-noise ratio did not decrease (CAR may not help line noise for this segment).")

    if np.sum(ratio > 2.0) > 0:
        print("⚠️ Some channels increased variance >2× after CAR (bad channels may contaminate CAR).")

    if corr_after < corr_before - 0.1:
        print("⚠️ Pairwise correlation dropped a lot; CAR may be over-subtracting shared signal.")
    else:
        print("✅ Correlation change is moderate; CAR effect seems reasonable.")


# ---------------------------
# Audio SR forensics
# ---------------------------

def estimate_audio_sr(audio_path: str, eeg_path: str, eeg_rate: int) -> float:
    audio_samples = get_time_length_from_npy(audio_path)
    eeg_samples = get_time_length_from_npy(eeg_path)
    ratio = audio_samples / max(eeg_samples, 1)
    return ratio * eeg_rate


def run_audio_sr_forensics(audio_path: str, eeg_path: str, eeg_rate: int) -> None:
    print("--- 🕵️ AUDIO SAMPLE RATE FORENSICS ---")

    if not os.path.exists(audio_path) or not os.path.exists(eeg_path):
        raise FileNotFoundError("Audio or EEG file not found. Check your paths.")

    audio_samples = get_time_length_from_npy(audio_path)
    eeg_samples = get_time_length_from_npy(eeg_path)
    ratio = audio_samples / max(eeg_samples, 1)
    estimated_sr = ratio * eeg_rate

    print("\n📝 THE MATH:")
    print(f"1. Audio Samples (N) = {audio_samples:,}")
    print(f"2. EEG Samples (M)   = {eeg_samples:,}")
    print(f"3. EEG Rate (Hz)     = {eeg_rate}")
    print("-" * 40)
    print("Formula:  (Audio_Samples / EEG_Samples) * EEG_Rate = Audio_SR")
    print(f"Calc:     ({audio_samples} / {eeg_samples}) * {eeg_rate}")
    print(f"Ratio:    {ratio:.4f} audio samples per 1 EEG sample")
    print(f"Result:   {ratio:.4f} * {eeg_rate} = {estimated_sr:.2f} Hz")
    print("-" * 40)

    print("\n⚖️ VERDICT:")
    if 44000 < estimated_sr < 50000:
        print(f"🚨 FOUND: ~48,000 Hz ({int(estimated_sr)} Hz)")
        print("   -> STATUS: likely RAW / NOT DOWNSAMPLED (for a 22.05k pipeline).")
    elif 20000 < estimated_sr < 24000:
        print(f"✅ FOUND: ~22,050 Hz ({int(estimated_sr)} Hz)")
        print("   -> STATUS: likely correctly downsampled.")
    else:
        print(f"⚠️ FOUND: {int(estimated_sr)} Hz")
        print("   -> STATUS: unknown / nonstandard; double check EEG rate and preprocessing.")


# ---------------------------
# CLI
# ---------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CAR checks + audio sample-rate forensics")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_car = sub.add_parser("car", help="Run CAR sanity checks on EEG.")
    p_car.add_argument("--eeg_path", required=True, help="Path to processed EEG .npy")
    p_car.add_argument("--fs", type=int, default=1024, help="EEG sample rate (default 1024)")
    p_car.add_argument("--seconds", type=int, default=60, help="Analyze first N seconds (default 60)")
    p_car.add_argument("--max_channels", type=int, default=64, help="Subsample channels for speed (default 64)")
    p_car.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling (default 0)")
    p_car.add_argument("--bw", type=float, default=1.0, help="Line-noise bandwidth +/- bw (default 1.0)")

    p_sr = sub.add_parser("sr", help="Estimate audio sample rate from audio/eeg lengths.")
    p_sr.add_argument("--audio_path", required=True, help="Path to audio .npy")
    p_sr.add_argument("--eeg_path", required=True, help="Path to EEG .npy")
    p_sr.add_argument("--eeg_rate", type=int, default=1024, help="Known EEG rate (default 1024)")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.cmd == "car":
        run_car_checks(
            eeg_path=args.eeg_path,
            fs=args.fs,
            seconds=args.seconds,
            max_channels=args.max_channels,
            seed=args.seed,
            bw=args.bw,
        )
    elif args.cmd == "sr":
        run_audio_sr_forensics(
            audio_path=args.audio_path,
            eeg_path=args.eeg_path,
            eeg_rate=args.eeg_rate,
        )
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
