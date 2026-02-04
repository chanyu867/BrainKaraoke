import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert



def preprocess_high_gamma(data, fs=1024):
    print(f"--- Processing Data (Shape: {data.shape}) ---")

    # Ensure shape is (Time, Channels)
    if data.shape[0] < data.shape[1]:
        print("Transposing input to (Time, Channels)...")
        data = data.T

    n_samples, n_channels = data.shape
    processed_data = np.zeros_like(data)

    # 1. Design Filters
    # Notch filters for 50Hz harmonics (Line noise)
    def elliptic_notch_sos(f0, fs, order=8, half_width_hz=1.0, rp=0.5, rs=60):
      low = f0 - half_width_hz
      high = f0 + half_width_hz
      sos = signal.iirfilter(
          N=order,
          Wn=[low, high],
          btype="bandstop",
          ftype="ellip",
          rp=rp,
          rs=rs,
          fs=fs,
          output="sos"
      )
      return sos

    # High-Gamma Bandpass (70-170 Hz) -> confirmed
    sos_bp = signal.butter(N=8, Wn=[70, 170], btype='bandpass', fs=fs, output='sos')

    # 2. Process Channel by Channel
    print(f"Applying Filters & Hilbert Transform...")
    for ch in range(n_channels):
        sig = data[:, ch]

        # Apply Notch
        for harmonics in (50, 100):
            sos50 = elliptic_notch_sos(harmonics, FS, order=8, half_width_hz=0.5, rp=0.5, rs=60)
            sig = signal.sosfiltfilt(sos50, sig)

        # Apply Bandpass
        sig = signal.sosfiltfilt(sos_bp, sig)

        # Apply Hilbert (Envelope)
        analytic_sig = hilbert(sig)
        processed_data[:, ch] = np.abs(analytic_sig)

        if (ch+1) % 20 == 0:
            print(f"  Channel {ch+1}/{n_channels} done.")

    return processed_data