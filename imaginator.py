import numpy as np
import torch
from torch import Dataset
import torch.nn.functional as F

def imaginator(data, raw_features, power_bands, fft_features):
    raw_cols = list(raw_features.keys())[1:-1]
    R_feat = data[raw_cols].values.astype(np.float32)
    P_feat = data[power_bands].values.astype(np.float32)
    F_feat = data[fft_features].values.astype(np.float32)
    
    def normalize_matrix(mat):
        """
        Normalize matrix values adaptively:
        - signed range -> [-1, 1] (preserve sign)
        - nonnegative range -> [0, 1]
        """
        m = np.asarray(mat, dtype=np.float32)
        if not np.all(np.isfinite(m)):
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

        min_v = float(m.min())
        max_v = float(m.max())
        if np.isclose(max_v, min_v):
            return np.zeros_like(m, dtype=np.float32)

        # Signed data: scale symmetrically to preserve direction.
        if min_v < 0.0 and max_v > 0.0:
            denom = float(np.max(np.abs(m)))
            return (m / denom).astype(np.float32) if denom > 0 else np.zeros_like(m, dtype=np.float32)

        # Nonnegative (or nonpositive) data: min-max to [0, 1].
        return ((m - min_v) / (max_v - min_v)).astype(np.float32)

    R_outer = normalize_matrix(np.outer(R_feat, R_feat))
    R_dist  = normalize_matrix(np.abs(R_feat[:, :, None] - R_feat[:, None, :]))
    P_outer = normalize_matrix(np.outer(P_feat, P_feat))
    P_dist  = normalize_matrix(np.abs(P_feat[:, :, None] - P_feat[:, None, :]))
    F_outer = normalize_matrix(np.outer(F_feat, F_feat))
    F_dist  = normalize_matrix(np.abs(F_feat[:, :, None] - F_feat[:, None, :]))
    
    RF_outer = normalize_matrix(np.outer(R_feat, F_feat))
    RF_dist  = normalize_matrix(np.abs(R_feat[:, :, None] - F_feat[:, None, :]))
    PR_outer = normalize_matrix(np.outer(P_feat, R_feat))
    PR_dist  = normalize_matrix(np.abs(P_feat[:, :, None] - R_feat[:, None, :]))
    FP_outer = normalize_matrix(np.outer(F_feat, P_feat))
    FP_dist  = normalize_matrix(np.abs(F_feat[:, :, None] - P_feat[:, None, :]))
    
    return np.stack([R_outer, R_dist, 
                    P_outer, P_dist, 
                    F_outer, F_dist,
                    RF_outer, RF_dist,
                    PR_outer, PR_dist,
                    FP_outer, FP_dist], axis=1)

# Stores 8×8 arrays in RAM; upscales each sample on-the-fly in __getitem__.
class EEGMatrixDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, scale: int = 8):
        # Keep as a numpy float32 array — no second tensor copy of the full data.
        self.X = X.astype(np.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.scale = scale

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)          # (1, C, 8, 8)
        if self.scale != self.X.shape[-1]:
            x = F.interpolate(x, size=(self.scale, self.scale), mode='nearest')
        return x.squeeze(0), self.y[idx]                         # (C, SCALE, SCALE)
