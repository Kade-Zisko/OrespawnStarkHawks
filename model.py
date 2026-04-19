import torch
import torch.nn as nn


class HiveNet(nn.Module):
    """input(14) → Linear(32) → ReLU → [Dropout] → Linear(16) → ReLU → [Dropout] → Linear(1) → Sigmoid"""

    def __init__(self, n_features: int = 14, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # shape (N, 1) — no squeeze so ONNX output is well-defined
