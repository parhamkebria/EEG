import re
import numpy as np
import pandas as pd

from config import _RAW_LABEL_MAP

def parse_num_list(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text[0] == "[" and text[-1] == "]":
        text = text[1:-1]
    tokens = text.replace(",", " ").split()
    if not tokens:
        return None
    try:
        return np.asarray(tokens, dtype=np.float32)
    except ValueError:
        return None

def fix_len(arr, n):
    if arr is None:
        return None
    if arr.size == n:
        return arr.astype(np.float32)
    if arr.size > n:
        return arr[:n].astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: arr.size] = arr
    return out

def power_to_seq(power, n=512):
    if power is None:
        return None
    power = fix_len(power, 8)
    x_old = np.arange(8, dtype=np.float32)
    x_new = np.linspace(0, 7, n, dtype=np.float32)
    return np.interp(x_new, x_old, power).astype(np.float32)

def _resolve_label_once(src: str, label_map: dict, cache: dict) -> str:
    if src in cache:
        return cache[src]

    seen = set()
    cur = src
    while cur in label_map and cur not in seen:
        seen.add(cur)
        cur = label_map[cur]

    cache[src] = cur
    return cur

# Flatten any chained mappings exactly once for O(1) runtime lookup.
_RESOLVED_LABEL_MAP = {}
for _key in _RAW_LABEL_MAP:
    _resolve_label_once(_key, _RAW_LABEL_MAP, _RESOLVED_LABEL_MAP)

# Optional case-insensitive support.
_RESOLVED_LABEL_MAP_LOWER = {k.lower(): v for k, v in _RESOLVED_LABEL_MAP.items()}

def to_major_class(label: str) -> str:
    s = str(label).strip()
    if not s:
        return s

    mapped = _RESOLVED_LABEL_MAP.get(s)
    if mapped is not None:
        return mapped

    mapped = _RESOLVED_LABEL_MAP_LOWER.get(s.lower())
    if mapped is not None:
        return mapped

    # Fast fallback for unseen labels.
    base = re.split(r"[-_]", s)[0]
    base = re.sub(r"\d+$", "", base)
    return base or s

def fft_feature_row(raw_signal, fs_hz):
    feature_template = {
        'fft_dominant_freq_hz': np.nan,
        'fft_spectral_centroid_hz': np.nan,
        'fft_spectral_entropy': np.nan,
        'fft_power_delta': np.nan,
        'fft_power_theta': np.nan,
        'fft_power_alpha': np.nan,
        'fft_power_beta': np.nan,
        'fft_power_gamma': np.nan,
    }

    if raw_signal is None:
        return feature_template

    x = np.asarray(raw_signal, dtype=np.float32)
    if x.ndim != 1 or x.size < 4:
        return feature_template

    x = x - x.mean()
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    psd = (np.abs(np.fft.rfft(x)) ** 2) / n

    if psd.size < 2:
        return feature_template

    # Ignore DC component for descriptive spectral features.
    psd_no_dc = psd.copy()
    psd_no_dc[0] = 0.0
    total_power = float(psd_no_dc.sum())

    if total_power <= 0.0:
        return feature_template

    dominant_idx = int(np.argmax(psd_no_dc))
    dominant_freq = float(freqs[dominant_idx])
    spectral_centroid = float(np.sum(freqs * psd_no_dc) / total_power)

    p = psd_no_dc / total_power
    p = p[p > 0]
    spectral_entropy = float(-(p * np.log2(p)).sum() / np.log2(len(psd_no_dc))) if len(psd_no_dc) > 1 else np.nan

    def band_power(lo_hz, hi_hz):
        mask = (freqs >= lo_hz) & (freqs < hi_hz)
        if not np.any(mask):
            return np.nan
        return float(psd_no_dc[mask].sum())

    return {
        'fft_dominant_freq_hz': dominant_freq,
        'fft_spectral_centroid_hz': spectral_centroid,
        'fft_spectral_entropy': spectral_entropy,
        'fft_power_delta': band_power(0.5, 4.0),
        'fft_power_theta': band_power(4.0, 8.0),
        'fft_power_alpha': band_power(8.0, 13.0),
        'fft_power_beta': band_power(13.0, 30.0),
        'fft_power_gamma': band_power(30.0, 45.0),
    }

def compute_fft(raw_signal, fs_hz):
    x = np.asarray(raw_signal, dtype=np.float32)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("raw_signal must be a 1D array-like with at least 2 samples.")

    # Remove DC offset so low-frequency energy is easier to interpret.
    x = x - x.mean()
    n = x.size

    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    spectrum = np.fft.rfft(x)
    amplitude = np.abs(spectrum) / n
    power = amplitude ** 2
    return freqs, amplitude, power

def summarize_vector(values, prefix):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_len": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_abs_mean": np.nan,
            f"{prefix}_clip_ratio": np.nan,
        }

    return {
        f"{prefix}_len": float(arr.size),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q05": float(np.quantile(arr, 0.05)),
        f"{prefix}_q95": float(np.quantile(arr, 0.95)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(arr))),
        f"{prefix}_clip_ratio": float(np.mean(np.abs(arr) >= 2047.0)),
    }

def class_weights(dataframe, label_col='label', method='complementary'):
    num_classes = len(dataframe[label_col].unique())
    # print(f"Number of unique classes: {num_classes}")
    sample_per_class = dataframe[label_col].value_counts()
    class_proportions = dataframe[label_col].value_counts(normalize=True)

    if method == 'complementary':
        # Method 1: class weight = 1 - class proportion 
        complementary_class_weights = (1.0 - class_proportions)
        print(f'Mean of complementary: {complementary_class_weights.mean():.4f} \t STD of complementary: {complementary_class_weights.std():.4f}')
        return complementary_class_weights

    elif method == 'normalized':
        # Method 2: class weight = (1 - class proportion) / (num_classes - 1)
        normalized_class_weights = complementary_class_weights / (num_classes - 1)
        print(f'Mean of normalized: {normalized_class_weights.mean():.4f} \t STD of normalized   : {normalized_class_weights.std():.4f}')
        return normalized_class_weights

    elif method == 'inverse':
        # Method 3: class weight = 1 / class proportion
        normalized_inverse_proportions = 1.0 / (class_proportions + 1e-8)
        normalized_inverse_proportions = normalized_inverse_proportions / normalized_inverse_proportions.sum()
        print(f'Mean of inverse: {normalized_inverse_proportions.mean():.4f} \t STD of inverse     : {normalized_inverse_proportions.std():.4f}')
        return normalized_inverse_proportions
    else:
        raise ValueError(f"Unsupported method: {method}")

