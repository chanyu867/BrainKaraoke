python -m src.waveglow.waveglow \
  --eeg_path /content/drive/MyDrive/Advance_python_project/p2_sEEG_processed.npy \
  --gt_audio_npy /content/drive/MyDrive/Advance_python_project/p2_audio_final.npy \
  --ckpt_path /content/drive/MyDrive/Advance_python_project/checkpoints_p2_1d/model-epoch=35-val_loss=2.0383.ckpt \
  --out_wav /content/drive/MyDrive/Advance_python_project/full_infer_p2_1d.wav
