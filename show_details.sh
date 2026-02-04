# check CAR
python -m src.preprocess.deta_details car --eeg_path /content/drive/MyDrive/Advance_python_project/p3_sEEG_processed.npy

# check audio data
python -m src.preprocess.deta_details sr \
  --audio_path /content/drive/MyDrive/Advance_python_project/p2_audio_final.npy \
  --eeg_path /content/drive/MyDrive/Advance_python_project/p2_sEEG_processed.npy
  