# Let brain talk! (sEEG → Audio)

This project trains a **sequence-to-sequence (Seq2Seq) neural model** that learns a mapping from **EEG/sEEG time-series windows** to an **audio representation**. The goal is to predict audio content aligned to brain activity, using an encoder–decoder model with attention.

The codebase is structured as:
- an entry script that launches training (`main.py`)
- a dataset/preprocessing layer (EEG + audio loading, alignment, transforms)
- a model definition (encoder/decoder + attention)
- a training pipeline (PyTorch Lightning module, loss/metrics, checkpoints)

---

## Concept Overview

### Inputs
- **ECoG/EEG (or sEEG) signal**: a 1D or multi-channel brain signal sampled over time.
- **Audio signal**: raw waveform aligned (in time) to the EEG.

The dataset layer handles:
- reading data from files (typically `.npy` arrays)
- splitting into train/val/test
- preparing synchronized EEG/audio segments suitable for a Seq2Seq model

### Model
A typical flow is:
1. **Encoder**: converts EEG sequences into latent features (often conv + RNN/GRU).
2. **Attention**: lets the decoder focus on the most relevant encoder time steps.
3. **Decoder**: predicts an audio representation step-by-step.

### Training
The training pipeline handles:
- loss computation (classification or regression depending on mode)
- logging (e.g., TensorBoard)
- saving checkpoints (best model by validation loss)
- optional saving of predictions for later evaluation/vocoding

---

## Supported Modes

This repository supports **Regression mode** for the audio target:

Mel-spectrogram (Regression mode)
- The audio waveform is converted into a **mel spectrogram** (Tacotron-style mel extraction).
- The model predicts **continuous mel values**.
- Typical loss: **MSE / L1** on mel frames.
- Useful if you want a representation that can later be vocoded back to waveform.

> Even if some code labels this as “MFCC”, the actual transform is usually **mel spectrogram** (common in Tacotron/WaveGlow workflows).

---

## Outputs
Depending on your config, the pipeline can produce:
- **Model checkpoints** (best by validation loss)
- **TensorBoard logs**
- **Saved predictions** (e.g., numpy arrays of target vs predicted audio/mel)
- In mel mode, it may optionally save example mel tensors for external vocoders

---

## How to Run

```bash
python main.py --config config.toml
```