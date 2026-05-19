import os
import json
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DUAL = False
SCALE = 16
EPOCHS = 20
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
PATIENCE = 5
MIN_DELTA = 1e-4
STOP_EARLY = True

DOUBLE_SCALE = 16
DOUBLE_BATCH_SIZE = 32
INPUT_CHANNELS = 12

if DUAL:
    SCALE = DOUBLE_SCALE
    BATCH_SIZE = DOUBLE_BATCH_SIZE
    INPUT_CHANNELS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_ID = 0 # only if cuda available, otherwise ignored

@dataclass
class Config:
    scale: int = SCALE
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    dropout_rate: float = DROPOUT_RATE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    num_workers: int = NUM_WORKERS
    patience: int = PATIENCE
    min_delta: float = MIN_DELTA
    device_id: int = DEVICE_ID
    dual: bool = DUAL
    double_scale: int = DOUBLE_SCALE
    double_batch_size: int = DOUBLE_BATCH_SIZE
    stop_early: bool = STOP_EARLY

RAW_PATH = Path("EEG/eeg-data.csv")
FULL_PATH = Path("EEG/eeg_data_with_features.csv")

OUTPUT_DIR = Path(f"outputs/results/{TIMESTAMP}")
ARCH_PATH = Path(f"outputs/results/{TIMESTAMP}/arch_{TIMESTAMP}.txt")
CONFIG_PATH = Path(f"outputs/results/{TIMESTAMP}/config_{TIMESTAMP}.json")
RESULTS_PATH = Path(f"outputs/results/{TIMESTAMP}/results_{TIMESTAMP}.txt")
LOGGING_PATH = Path(f"outputs/results/{TIMESTAMP}/training_log_{TIMESTAMP}.csv")
CHECKPOINT_PATH = Path(f"outputs/results/{TIMESTAMP}/best_model_{TIMESTAMP}.pth")


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
