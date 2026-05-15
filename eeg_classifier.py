from pathlib import Path
import torch
import torchinfo
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from utils import *
from config import *
from imaginator import *

class EEGClassifier():
    class EEGCNNClassifier(nn.Module):
        def __init__(self, in_channels=6, num_classes=9):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32,  kernel_size=3, padding=1),
                nn.BatchNorm2d(32),  nn.ReLU(),
                nn.Conv2d(32,         64,  kernel_size=3, padding=1),
                nn.BatchNorm2d(64),  nn.ReLU(),
                nn.Conv2d(64,         128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(64, num_classes),
            )

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
        print(f"Number of IDs: {len(list(eeg_data.id.value_counts().keys()))}")
        print(f"Number of labels: {len(list(eeg_data.label.value_counts().keys()))}")
        print("Shape after reloading from CSV:", eeg_data.shape)
        return eeg_data

    @staticmethod
    def _build_feature_matrices(data, raw_features, power_bands, fft_features):
        raw_cols = list(raw_features[1:-1])
        power_cols = list(power_bands)
        fft_cols = list(fft_features)

        required_columns = raw_cols + power_cols + fft_cols + ['id', 'label']
        missing_columns = [column for column in required_columns if column not in data.columns]
        if missing_columns:
            raise KeyError(f"Missing required feature columns: {missing_columns}")

        raw_values = data[raw_cols].to_numpy(dtype=np.float32)
        power_values = data[power_cols].to_numpy(dtype=np.float32)
        fft_values = data[fft_cols].to_numpy(dtype=np.float32)

        raw_outer = np.einsum('ni,nj->nij', raw_values, raw_values)
        raw_dist = np.abs(raw_values[:, :, None] - raw_values[:, None, :])
        power_outer = np.einsum('ni,nj->nij', power_values, power_values)
        power_dist = np.abs(power_values[:, :, None] - power_values[:, None, :])
        fft_outer = np.einsum('ni,nj->nij', fft_values, fft_values)
        fft_dist = np.abs(fft_values[:, :, None] - fft_values[:, None, :])
        raw_fft_outer = np.einsum('ni,nj->nij', raw_values, fft_values)
        raw_fft_dist = np.abs(raw_values[:, :, None] - fft_values[:, None, :])
        power_raw_outer = np.einsum('ni,nj->nij', power_values, raw_values)
        power_raw_dist = np.abs(power_values[:, :, None] - raw_values[:, None, :])
        fft_power_outer = np.einsum('ni,nj->nij', fft_values, power_values)
        fft_power_dist = np.abs(fft_values[:, :, None] - power_values[:, None, :])

        return np.stack(
            [
                raw_outer,
                raw_dist,
                power_outer,
                power_dist,
                fft_outer,
                fft_dist,
                raw_fft_outer,
                raw_fft_dist,
                power_raw_outer,
                power_raw_dist,
                fft_power_outer,
                fft_power_dist,
            ],
            axis=1,
        )

    @classmethod
    def data_loader(cls, data, raw_features, power_bands, fft_features, batch_size=32):
        X = cls._build_feature_matrices(data, raw_features, power_bands, fft_features)
        le = LabelEncoder()
        y = le.fit_transform(data['label'].astype(str).values)
        groups = data['id'].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, va_idx = next(gss.split(X, y, groups=groups))
        X_train, X_val = X[tr_idx], X[va_idx]
        y_train, y_val = y[tr_idx], y[va_idx]
        n_tr_subj = len(np.unique(groups[tr_idx]))
        n_va_subj = len(np.unique(groups[va_idx]))
        print(f"Train: {X_train.shape}  ({n_tr_subj} subjects)")
        print(f"Val:   {X_val.shape}  ({n_va_subj} subjects)")
        ch_mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        ch_std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
        X_train = (X_train - ch_mean) / ch_std
        X_val = (X_val - ch_mean) / ch_std
        train_loader = DataLoader(EEGMatrixDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EEGMatrixDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        print(f"\nLoader - train: {len(train_loader)} batches, val: {len(val_loader)} batches")
        num_classes = len(np.unique(y_train))
        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        cw = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32)
        cw = (cw / cw.sum() * num_classes).to(DEVICE)
        return train_loader, val_loader, num_classes, cw, le

    def build_model(self, input_channels, spatial_size, num_classes, cw):
        self.model     = self.EEGCNNClassifier(in_channels=input_channels, num_classes=num_classes).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss(weight=cw)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5)

        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(torchinfo.summary(self.model, input_size=(1, input_channels, spatial_size, spatial_size), verbose=0))
        return self.model, self.optimizer, self.criterion, self.scheduler

    def train(self, train_loader, val_loader, epochs, learning_rate, weight_decay, device):
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        input_channels = train_loader.dataset.X.shape[1]
        spatial_size = train_loader.dataset.X.shape[2]
        self.model, self.optimizer, self.criterion, self.scheduler = self.build_model(
            input_channels=input_channels,
            spatial_size=spatial_size,
            num_classes=self.num_classes,
            cw=self.cw,
        )
        best_f1 = float('-inf')
        best_epoch = 0
        best_weights = None
        patience_counter = 0
        for epoch in range(1, epochs + 1):
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

            tr_loss /= len(train_loader.dataset)
            tr_f1 = f1_score(tr_targets, tr_preds, average='macro', zero_division=0)
            # — Validate —
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
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1 or improved:
                tag = '  <- best' if improved else ''
                print(f"Epoch {epoch:3d}/{epochs} | "
                    f"TLoss: {tr_loss:.4f}  TF1: {tr_f1:.4f} | "
                    f"VLoss: {va_loss:.4f}  VF1: {va_f1:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}{tag}")

            if patience_counter >= patience:
                print(f"\nEarly stop at epoch {epoch}  "
                    f"(best: epoch {best_epoch}, val F1: {best_f1:.4f})")
                break

        if best_weights is None:
            best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        print(f"\nRestored best checkpoint — Val F1-macro: {best_f1:.4f}")

        return self.model

    def evaluate(self, model, val_loader, device, le=None):
        model.eval()
        self.final_preds, self.final_targets = [], []
        self.le = le
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                self.final_preds.extend(logits.argmax(1).cpu().numpy())
                self.final_targets.extend(yb.numpy())

        print("\nClassification Report (CNN — 6-matrix images):")
        report_kwargs = {'zero_division': 0}
        if le is not None:
            report_kwargs['target_names'] = le.classes_
        print(classification_report(self.final_targets, self.final_preds, **report_kwargs))

    def reaulrs(self):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].plot(self.history['train_loss'], label='Train')
        axs[0].plot(self.history['val_loss'],   label='Val')
        axs[0].set_title('Loss');    axs[0].set_xlabel('Epoch')
        axs[0].legend(); axs[0].grid(alpha=0.3)

        axs[1].plot(self.history['train_f1'], label='Train')
        axs[1].plot(self.history['val_f1'],   label='Val')
        axs[1].set_title('F1-Macro'); axs[1].set_xlabel('Epoch')
        axs[1].legend(); axs[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        ConfusionMatrixDisplay.from_predictions(
            self.final_targets, self.final_preds,
            display_labels=self.le.classes_,
            xticks_rotation=90,
            normalize='true',
            values_format='.2f',
            cmap='Blues',
        )
        plt.title('Confusion Matrix — val set (normalized)')
        plt.tight_layout()
        plt.show()

    def main(self):
        data = self.load_csv_data()
        train_loader, val_loader, self.num_classes, self.cw, self.le = self.data_loader(
            data,
            raw_features=RAW_FEATURES,
            power_bands=POWER_BANDS,
            fft_features=FFT_FEATURE_COLUMNS,
            batch_size=batch_size,
        )
        self.model = self.train(
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=DEVICE,
        )
        self.evaluate(self.model, val_loader, device=DEVICE, le=self.le)
        self.reaulrs()

# if __name__ == "__main__":
#     eeg_classifier = EEGClassifier()
#     eeg_classifier.main()