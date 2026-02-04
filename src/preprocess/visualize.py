#!/usr/bin/env python3
"""
Simple sEEG .npy inspector + optional PSD proof plot.

Examples:
  python visualize.py --data_path /path/to/p1_sEEG.npy
  python visualize.py --data_path /path/to/p1_sEEG_processed.npy

  # PSD proof (raw -> bandpass filtered)
  python visualize.py --data_path /path/to/p1_sEEG.npy --psd --raw_path /path/to/p1_sEEG.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import welch


def load_npy(path: str) -> np.ndarray:
    """Load a .npy file."""
    return np.load(path)


def ensure_time_by_channels(x: np.ndarray) -> np.ndarray:
    """
    Ensure shape is (time, channels).
    If input looks like (channels, time), transpose it.
    """
    if x.ndim == 1:
        # (time,) -> (time, 1)
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")

    # Heuristic from your script: if channels < time, assume (channels, time)
    if x.shape[0] < x.shape[1]:
        return x.T
    return x


def print_stats(path: str, x: np.ndarray) -> None:
    print(f"--- File: {path} ---")
    print(f"Shape: {x.shape}  (time, channels)")
    print(f"Min Value: {x.min():.4f}")
    print(f"Max Value: {x.max():.4f}")

    if x.min() < 0:
        print("\nanalysis result: RAW DATA DETECTED ⚠️")
        print("The data contains negative values. This is likely raw voltage.")
        print("Action needed: You MUST run the preprocessing script.")
    else:
        print("\nanalysis result: PREPROCESSED DATA ✅")
        print("The data is strictly positive. This is likely the High-Gamma Envelope.")
        print("Action needed: None. You can run the training directly.")


def get_channel_segment(x: np.ndarray, channel: int, n_samples: int) -> np.ndarray:
    """Return a 1D segment from a given channel."""
    if channel < 0 or channel >= x.shape[1]:
        raise ValueError(f"channel must be in [0, {x.shape[1]-1}], got {channel}")
    return x[:n_samples, channel]


def bandpass_gamma(sig_1d: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    sos = signal.butter(N=order, Wn=[low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig_1d)


def compute_psd(sig_1d: np.ndarray, fs: float, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    f, p = welch(sig_1d, fs=fs, nperseg=nperseg)
    return f, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sEEG .npy and optionally plot PSD proof.")
    parser.add_argument("--data_path", required=True, help="Path to .npy file to inspect/plot.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to plot (default: 0).")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples for time plot (default: 1000).")

    # Optional PSD proof settings
    parser.add_argument("--psd", action="store_true", help="Also plot PSD before/after gamma bandpass filtering.")
    parser.add_argument("--raw_path", default=None, help="Path to RAW .npy for PSD (defaults to --data_path).")
    parser.add_argument("--fs", type=float, default=1024.0, help="Sampling rate for PSD/filter (default: 1024).")
    parser.add_argument("--bp_low", type=float, default=70.0, help="Bandpass low cutoff (default: 70).")
    parser.add_argument("--bp_high", type=float, default=170.0, help="Bandpass high cutoff (default: 170).")
    parser.add_argument("--bp_order", type=int, default=8, help="Butterworth order (default: 8).")
    parser.add_argument("--nperseg", type=int, default=1024, help="Welch nperseg (default: 1024).")
    parser.add_argument("--psd_xlim", type=float, default=300.0, help="PSD x-axis limit in Hz (default: 300).")

    args = parser.parse_args()

    # ---- Load + standardize shape ----
    data = ensure_time_by_channels(load_npy(args.data_path))

    # ---- Print stats + quick interpretation ----
    print_stats(args.data_path, data)

    # ---- Prepare figure (1 or 2 subplots) ----
    nrows = 2 if args.psd else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 7 if args.psd else 4), constrained_layout=True)

    if nrows == 1:
        axes = [axes]  # make it list-like

    # ---- Plot 1: time series ----
    seg = get_channel_segment(data, channel=args.channel, n_samples=args.n_samples)
    axes[0].plot(seg)
    axes[0].set_title(f"First {len(seg)} samples of Channel {args.channel}")
    axes[0].set_xlabel("Time (samples)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    # ---- Plot 2: PSD proof (optional) ----
    if args.psd:
        raw_path = args.raw_path or args.data_path
        raw = ensure_time_by_channels(load_npy(raw_path))
        raw_sig = get_channel_segment(raw, channel=args.channel, n_samples=max(args.nperseg * 10, 10000))

        filt_sig = bandpass_gamma(raw_sig, fs=args.fs, low=args.bp_low, high=args.bp_high, order=args.bp_order)

        f_raw, p_raw = compute_psd(raw_sig, fs=args.fs, nperseg=args.nperseg)
        f_flt, p_flt = compute_psd(filt_sig, fs=args.fs, nperseg=args.nperseg)

        ax = axes[1]
        ax.semilogy(f_raw, p_raw, label="Original (raw)")
        ax.semilogy(f_flt, p_flt, label="After bandpass filter")
        ax.axvspan(args.bp_low, args.bp_high, alpha=0.2, label=f"Target band ({args.bp_low}-{args.bp_high} Hz)")
        ax.set_title("PSD: before vs after filtering")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (log)")
        ax.set_xlim(0, args.psd_xlim)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
