import json
import torch
from pathlib import Path

scale = 16
batch_size = 32
epochs = 20
dropout_rate = 0.3
learning_rate = 1e-3
weight_decay = 1e-5
num_workers = 4
patience = 5
min_delta = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_PATH = Path("EEG/eeg-data.csv")
FULL_PATH = Path("EEG/eeg_data_with_features.csv")

POWER_BANDS = [
    'delta',
    'theta',
    'low_alpha', 
    'high_alpha',
    'low_beta',
    'high_beta',
    'low_gamma',
    'mid_gamma'
]

RAW_FEATURES = [
    "raw_len",
    "raw_mean",
    "raw_std",
    "raw_min",
    "raw_max",
    "raw_median",
    "raw_q05",
    "raw_q95",
    "raw_abs_mean",
    "raw_clip_ratio"
]

if 'FFT_SAMPLE_RATE_HZ' not in globals():
    FFT_SAMPLE_RATE_HZ = 512

FFT_FEATURE_COLUMNS = [
    'fft_dominant_freq_hz',
    'fft_spectral_centroid_hz',
    'fft_spectral_entropy',
    'fft_power_delta',
    'fft_power_theta',
    'fft_power_alpha',
    'fft_power_beta',
    'fft_power_gamma',
]

LABELS_JSON_PATH = Path("labels.json")
if not LABELS_JSON_PATH.exists():
    raise FileNotFoundError(f"Missing label mapping file: {LABELS_JSON_PATH}")

_RAW_LABEL_MAP = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
LABELS = 9
if LABELS == 9:
    # Map to 9 major classes.
    _RAW_LABEL_MAP = _RAW_LABEL_MAP.get("labels_9", {})
elif LABELS == 19:
    # Map to 19 classes (mostly 1-to-1 with raw labels).
    _RAW_LABEL_MAP = _RAW_LABEL_MAP.get("labels_19", {})
else:
    raise ValueError(f"Unsupported number of labels: {LABELS}")