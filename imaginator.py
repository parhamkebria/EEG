import torch
import torchinfo
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from time import time

from  utils import *
from config import *

def imaginator(data, raw_features, power_bands, fft_features):
    raw_cols = list(raw_features[1:-1])
    power_cols = list(power_bands)
    fft_cols = list(fft_features)

    required_columns = raw_cols + power_cols + fft_cols + ['id', 'label']
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required feature columns: {missing_columns}")
    R_feat = data[raw_cols].values.astype(np.float32)
    P_feat = data[power_cols].values.astype(np.float32)
    F_feat = data[fft_cols].values.astype(np.float32)
    
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
    
    X_img = np.stack([R_outer, R_dist, 
                    P_outer, P_dist, 
                    F_outer, F_dist,
                    RF_outer, RF_dist,
                    PR_outer, PR_dist,
                    FP_outer, FP_dist], axis=1)
    
    print(f"\nIMAGINATOR tensor shape: {X_img.shape}\n")
    return X_img

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
        return x.squeeze(0), self.y[idx]                        # (C, SCALE, SCALE)

class EEGClassifier():
    class EEGCNN(nn.Module):
        def __init__(self, dropout_rate, in_channels=12, num_classes=9):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 4, kernel_size=7, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                nn.Conv2d(          4, 4, kernel_size=3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),                               
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(4, 2), nn.ReLU(), nn.Dropout(dropout_rate),
                nn.Linear(2, num_classes),
            )
            
            self.timestamp = TIMESTAMP
            self.disp = None

        def forward(self, x):
            return self.classifier(self.features(x))

    @staticmethod
    def load_csv_data(csv_path: Path | None = None) -> pd.DataFrame:
        csv_path = csv_path or FULL_PATH
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        eeg_data = pd.read_csv(csv_path, 
                            converters={"eeg_power": parse_num_list, 
                                        "raw_values": parse_num_list},
                            low_memory=False)
        print(f"\nNumber of IDs: {len(list(eeg_data.id.value_counts().keys()))}")
        print(f"Number of labels: {len(list(eeg_data.label.value_counts().keys()))}")
        print("Shape after reloading from CSV:", eeg_data.shape)
        print("-" * 20 + "CSV Data Loaded" + "-" * 20)
        return eeg_data

    @classmethod
    def data_loader( data, raw_features, power_bands, fft_features, batch_size=batch_size, scale=scale, num_workers=num_workers):
        X_img = imaginator(data, raw_features, power_bands, fft_features)
        le = LabelEncoder()
        y = le.fit_transform(data['label'].astype(str).values)
        groups = data['id'].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(gss.split(X_img, y, groups=groups))
        X_train, X_val = X_img[tr_idx], X_img[va_idx]
        y_train, y_val = y[tr_idx], y[va_idx]
        n_tr_subj = len(np.unique(groups[tr_idx]))
        n_va_subj = len(np.unique(groups[va_idx]))
        print(f"Train: {X_train.shape}  ({n_tr_subj} subjects)")
        print(f"Val:   {X_val.shape}  ({n_va_subj} subjects)")
        
        # Normalize using training set stats; apply same transform to val set.
        ch_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        ch_std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        X_train = (X_train - ch_mean) / ch_std
        X_val = (X_val - ch_mean) / ch_std
        
        train_loader = DataLoader(EEGMatrixDataset(X_train, y_train, scale=scale), 
                                batch_size=batch_size, 
                                shuffle=True,
                                drop_last=False,
                                num_workers=num_workers,
                                pin_memory=False)
        val_loader = DataLoader(EEGMatrixDataset(X_val, y_val, scale=scale),
                                batch_size=batch_size, 
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers,
                                pin_memory=False)
        print(f"\nLoader - train: {len(train_loader)} batches, val: {len(val_loader)} batches")
        
        # Class weights for imbalanced data: inverse frequency, normalized to num_classes.
        num_classes = len(np.unique(y_train))
        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        cw = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
        cw = (cw / cw.sum() * num_classes).to(DEVICE)
        _bytes_per_batch = batch_size * 12 * scale * scale * 4
        print(f"\nSCALE={scale}  |  batch_size={batch_size}")
        print(f"Dataset RAM (8×8, both splits): {(X_train.nbytes + X_val.nbytes) / 1e6:.1f} MB")
        print(f"Peak RAM per batch (approx):    {_bytes_per_batch / 1e6:.1f} MB")
        print(f"Loader — train: {len(train_loader)} batches, val: {len(val_loader)} batches")
        print("-" * 20 + "DataLoader ready" + "-" * 20)
        return train_loader, val_loader, num_classes, cw, le
    
    def build_model(self, input_channels, spatial_size, num_classes, cw, learning_rate, weight_decay, dropout_rate):
        self.model = self.EEGCNN(dropout_rate, in_channels=input_channels, num_classes=num_classes).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(weight=cw)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)

        print(f"\nTrainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(torchinfo.summary(self.model, input_size=(1, input_channels, spatial_size, spatial_size), verbose=0))
        print("-" * 20 + "Model built" + "-" * 20)
        return self.model, self.optimizer, self.criterion, self.scheduler

    def train(self, train_loader, val_loader, epochs, learning_rate, weight_decay, device=DEVICE):
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        input_channels = train_loader.dataset.X.shape[1]
        spatial_size = train_loader.dataset.X.shape[2]
        
        self.model, self.optimizer, self.criterion, self.scheduler = self.build_model(
            input_channels=input_channels,
            spatial_size=spatial_size,
            num_classes=self.num_classes,
            cw=self.cw,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=0.5
        )
        
        best_f1 = float('-inf')
        best_epoch = 0
        best_weights = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, epochs + 1):
            start_time = time()
            self.model.train()
            tr_loss, tr_preds, tr_targets = 0.0, [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss   = self.criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tr_loss += loss.item() * len(yb)
                tr_preds.extend(logits.argmax(1).detach().cpu().numpy())
                tr_targets.extend(yb.cpu().numpy())
            
            # Compute epoch metrics
            tr_loss /= len(train_loader.dataset)
            tr_f1 = f1_score(tr_targets, tr_preds, average='macro', zero_division=0)
            
            # Validation step
            self.model.eval()
            va_loss, va_preds, va_targets = 0.0, [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits  = self.model(xb)
                    va_loss += self.criterion(logits, yb).item() * len(yb)
                    va_preds.extend(logits.argmax(1).cpu().numpy())
                    va_targets.extend(yb.cpu().numpy())

            va_loss /= len(val_loader.dataset)
            va_f1    = f1_score(va_targets, va_preds, average='macro', zero_division=0)
            self.scheduler.step(va_f1)
            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(va_loss)
            self.history['train_f1'].append(tr_f1)
            self.history['val_f1'].append(va_f1)

            improved = va_f1 > best_f1 + min_delta
            if improved:
                best_f1, best_epoch = va_f1, epoch
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(
                {
                    'epoch': epoch,
                    'val_f1_macro': float(va_f1),
                    'model_state_dict': best_weights,
                },
                CHEKPOINT_PATH,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            tag = '  <- best' if improved else ''
            print(f"Epoch {epoch:3d}/{epochs} | "
                f"TLoss: {tr_loss:.4f}  TF1: {tr_f1:.4f} | "
                f"VLoss: {va_loss:.4f}  VF1: {va_f1:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}{tag}")

            print(f"Epoch {epoch} completed in {(time() - start_time):.1f} seconds.")

            if patience_counter >= patience:
                print(f"\nEarly stop at epoch {epoch}  "
                    f"(best: epoch {best_epoch}, val F1: {best_f1:.4f})")
                break
        
        if best_weights is None:
            best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        print(f"\nRestored best checkpoint — Val F1-macro: {best_f1:.4f}")
        print(f"Saved/loaded checkpoint path: {CHEKPOINT_PATH}")

        return self.model
    
    def evaluate(self, model, val_loader, device=DEVICE, le=None):
        start_time = time()
        model.eval()
        va_preds, va_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                va_preds.extend(logits.argmax(1).cpu().numpy())
                va_targets.extend(yb.cpu().numpy())
        print("\n" + "-" * 20 + "Validation Results" + "-" * 20)
        report_kwargs = {'zero_division': 0}
        if le is not None:
            report_kwargs['target_names'] = le.classes_
        print(classification_report(va_targets, va_preds, **report_kwargs))
        _cm = confusion_matrix(va_targets, va_preds, normalize='true')
        self.disp = ConfusionMatrixDisplay.from_predictions(confusion_matrix=_cm, 
                                                            display_labels=le.classes_,
                                                            xticks_rotation=90,
                                                            values_format='.1f',
                                                            cmap='Blues')
        print("-" * 20 + f"Evaluation completed in {(time() - start_time):.1f} seconds." + "-" * 20)

    def save_config(self, config_path: Path = CONFIG_PATH):
        config_data = {
            'timestamp': self.timestamp,
            'batch_size': batch_size,
            'epochs': epochs,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_workers': num_workers,
            'patience': patience,
            'min_delta': min_delta,
            'DEVICE': str(DEVICE),
            'SINGLE_SCALE': SINGLE_SCALE,
            'SINGLE_BATCH_SIZE': SINGLE_BATCH_SIZE,
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"\nConfig saved to {config_path}")
    
    def plot_results(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss over Epochs'); plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_f1'], label='Train F1-macro')
        plt.plot(epochs, self.history['val_f1'], label='Val F1-macro')
        plt.xlabel('Epoch'); plt.ylabel('F1-macro'); plt.title('F1-macro over Epochs'); plt.legend()
        plt.tight_layout()
        plt.savefig(f"training_results_{self.timestamp}.png", dpi=300)
        
        if self.disp is not None:
            self.disp.plot(cmap='Blues', xticks_rotation=90, values_format='.1f')
            plt.title('CNN Confusion Matrix - val set (normalized)')
            plt.tight_layout()
            plt.savefig(f"ConfusionMatrix_{self.timestamp}.png", dpi=300)