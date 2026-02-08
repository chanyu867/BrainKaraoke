<img width="1529" height="743" alt="Brainkaraoke_title_image" src="https://github.com/user-attachments/assets/8d523230-c2db-432d-880e-62904bb70316" />

# Let brain talk! (sEEG → Audio)

This project trains a **sequence-to-sequence (Seq2Seq) neural model** that learns a mapping from **EEG/sEEG time-series windows** to an **audio representation**. The goal is to predict audio content aligned to brain activity, using an encoder–decoder model with attention.

The codebase is structured as:
- A preprocessing script for both sEEG data and audio data. EEG preprocessing is not supported.
- A model training script for RNN model, which infer mel-spectrogram from SEEG data.
- A simple script to run waveglow pipeline with pre-trained weights to convert inferred mel-spectrogram into audio format(mp3).
- A simple script to show the data details and visualize the results.

## 🎤Concept Overview

### Inputs
- **ECoG/EEG (or sEEG) signal**: a 1D or multi-channel brain signal sampled over time.
- **Audio signal**: raw waveform aligned (in time) to the EEG.

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

## 🔘Supported Modes
Mel-spectrogram (Regression mode)
- The audio waveform is converted into a **mel spectrogram** (Tacotron-style mel extraction).
- The model predicts **continuous mel values**.
- Typical loss: **MSE / L1** on mel frames.
- Useful if you want a representation that can later be vocoded back to waveform.

> Even if some code labels this as “MFCC”, the actual transform is usually **mel spectrogram** (common in Tacotron/WaveGlow workflows).
> During the training, checkpoints for each epoch will be saved in specified directory.

## 🏃‍♀️How to Run

```bash
## 0. set environments
git clone https://github.com/chanyu867/BrainKaraoke.git
cd Brainkaraoke
pip install -r requirements.txt
chmod +x preprocess.sh show_details.sh train.sh waveglow.sh

## 1. preprocessing data and check data details
./preprocess.sh
./show_details.sh

## 2. start training
./train.sh

## 3. convert inferred mel into audio format(mp3)
./waveglow.sh

## 4. calculate performance of trained model
./test_performance.sh

## Optional: you can run pytest on our main programs, stored in "BrainKaraoke/test_perf"
cd tests # please refer the README.md in the subfolder for more details
```

## 🗣️Output examples

<img width="1298" height="396" alt="GT_vs_inferred_mel_spectrogram" src="https://github.com/user-attachments/assets/563ab609-4a08-450c-b501-05bc0553c7da" />

## 📕Reference
> Articles:
> - Kohler et. al., 2022, NBDT, "Synthesizing Speech from Intracranial Depth Electrodes using an Encoder-Decoder Framework"
> - Prenger et. al., 2018, "WAVEGLOW: A FLOW-BASED GENERATIVE NETWORK FOR SPEECH SYNTHESIS
>
> Github repository:
> - StereoEEG2speech: https://github.com/jonaskohler/stereoEEG2speech.git
> - Waveglow: https://github.com/NVIDIA/waveglow.git
