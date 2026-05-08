import numpy as np

def imaginator(data, raw_features, power_bands, fft_features):
    raw_cols = list(raw_features.keys())[1:-1]
    R_feat = data[raw_cols].values.astype(np.float32)
    P_feat = data[power_bands].values.astype(np.float32)
    F_feat = data[fft_features].values.astype(np.float32)

    R_outer = np.einsum('ni,nj->nij', R_feat, R_feat)
    R_dist  = np.abs(R_feat[:, :, None] - R_feat[:, None, :])
    P_outer = np.einsum('ni,nj->nij', P_feat, P_feat)
    P_dist  = np.abs(P_feat[:, :, None] - P_feat[:, None, :])
    F_outer = np.einsum('ni,nj->nij', F_feat, F_feat)
    F_dist  = np.abs(F_feat[:, :, None] - F_feat[:, None, :])
    
    RF_outer = np.einsum('ni,nj->nij', R_feat, F_feat)
    RF_dist  = np.abs(R_feat[:, :, None] - F_feat[:, None, :])
    PR_outer = np.einsum('ni,nj->nij', P_feat, R_feat)
    PR_dist  = np.abs(P_feat[:, :, None] - R_feat[:, None, :])
    FP_outer = np.einsum('ni,nj->nij', F_feat, P_feat)
    FP_dist  = np.abs(F_feat[:, :, None] - P_feat[:, None, :])
    
    return np.stack([R_outer, R_dist, 
                    P_outer, P_dist, 
                    F_outer, F_dist,
                    RF_outer, RF_dist,
                    PR_outer, PR_dist,
                    FP_outer, FP_dist], axis=1)


