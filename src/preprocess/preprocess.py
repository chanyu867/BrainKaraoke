import numpy as np
from src.preprocess.filter import preprocess_high_gamma

INPUT_FILE = "/content/drive/MyDrive/Advance_python_project/p3_sEEG.npy"
OUTPUT_FILE = '/content/drive/MyDrive/Advance_python_project/p3_sEEG_processed.npy' # Fixed path to save in Drive
FS = 1024

# EXECUTE
print(f"Loading raw data from: {INPUT_FILE}")
raw_data = np.load(INPUT_FILE)
clean_data = preprocess_high_gamma(raw_data, fs=FS)
np.save(OUTPUT_FILE, clean_data)
print(f"✅ Saved processed file to: {OUTPUT_FILE}")