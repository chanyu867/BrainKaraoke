<img width="1529" height="743" alt="Brainkaraoke_title_image" src="https://github.com/user-attachments/assets/8d523230-c2db-432d-880e-62904bb70316" />

# Let brain talk! (sEEG → Audio)

This project trains a **sequence-to-sequence (Seq2Seq) neural model** that learns a mapping from **EEG/sEEG time-series windows** to an **audio representation**. The goal is to predict audio content aligned to brain activity, using an encoder–decoder model with attention. The codebase is based on the original study by Kohler et al., but has been substantially extended and refactored. In addition to improving the released implementation, we added missing components needed to run the full end-to-end pipeline, starting from preprocessing to mel-transformation to audio format.

The codebase is structured as:
- A full preprocessing script for both sEEG data and audio data.
- A model training script for RNN model, which infer mel-spectrogram from sEEG data.
- A simple script to run waveglow pipeline with pre-trained weights to convert inferred mel-spectrogram into audio format(mp3).
- A simple script to show the data details and visualize the results.
- The code support two attention modes, Bahdanau attention and normal attention. First is set by default.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

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

## 📦 Data Download and Organization

The data used in this project is publicly available on the **Open Science Framework (OSF)**:

🔗 https://osf.io/7wf6n/

Please note that the available audio file are anonymized; thus, current implementation will not reproduce the same results due to data quality.

### Dataset contents
The dataset contains recordings from **three patients**.  
For **each patient**, you need to download:

- **sEEG data**: a `.npy` file used as input for preprocessing and model training  
- **Audio data**: the corresponding audio waveform file aligned to the sEEG recording

These files are required for running preprocessing, training, and inference.

### How to download
1. Go to the OSF project page: https://osf.io/7wf6n/
2. Navigate to the data section for each patient
3. Download:
   - the `.npy` file containing the sEEG data
   - the corresponding audio file for that patient
4. Repeat this process for all **three patients**

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
cd src/tests # please refer the README.md in the subfolder for more details
```

## 🗣️Output examples

<img width="1298" height="396" alt="GT_vs_inferred_mel_spectrogram" src="https://github.com/user-attachments/assets/563ab609-4a08-450c-b501-05bc0553c7da" />

## 📕Reference
> Articles:
> This project includes and is derived from code by Prem Seetharaman (2017), licensed under the BSD 3-Clause License. Significant modifications have been made.
> 
> - Kohler et. al., 2022, NBDT, "Synthesizing Speech from Intracranial Depth Electrodes using an Encoder-Decoder Framework"
> - Prenger et. al., 2018, "WAVEGLOW: A FLOW-BASED GENERATIVE NETWORK FOR SPEECH SYNTHESIS
>
> Github repository:
> - StereoEEG2speech: https://github.com/jonaskohler/stereoEEG2speech.git
> - Waveglow: https://github.com/NVIDIA/waveglow.git
