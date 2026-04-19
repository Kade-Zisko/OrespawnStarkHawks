import torch
import torch.nn as nn


class HiveNetHeavy(nn.Module):
    """
    Beehive stress classifier — heavy variant.

    Architecture: 41 → 128 → 64 → 32 → 16 → 1
    Each hidden layer (except the last pair) uses BatchNorm + ReLU + Dropout.
    Output is a single Sigmoid unit — probability of hive stress.

    Trained with:
      - Focal loss (α=0.25, γ=2.0) to handle class imbalance
      - SMOTE oversampling on the minority (stressed) class
      - Adam + ReduceLROnPlateau, early stopping at patience=15
      - Per-hive z-score normalisation + StandardScaler on all 41 features

    41 input features
    -----------------
    Sensor-derived (37):
      temperature, humidity
      temp/hum rolling stats at 12h, 24h, 72h windows (mean, std, min, max, range)
      temp/hum trend diffs at 6h and 48h
      temp_hum_ratio  (temperature / humidity)

    Temporal encodings (4):
      hour_sin, hour_cos   — circadian cycle
      doy_sin,  doy_cos    — seasonal cycle
    """

    def __init__(self, n_features: int = 41, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
