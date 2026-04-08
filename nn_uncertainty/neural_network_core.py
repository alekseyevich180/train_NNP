from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dims: list[int]
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 1.0e-5
    seed: int = 42


if nn is not None:
    class SelectorMLP(nn.Module):
        """Simple MLP used by nn_uncertainty to score each structure frame."""

        def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(-1)
else:
    class SelectorMLP:  # pragma: no cover
        pass


def standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalization used before training."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def split_indices(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random 80/20 train-validation split."""
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_size = max(1, int(0.8 * num_samples))
    if train_size >= num_samples:
        train_size = max(1, num_samples - 1)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:] if train_size < num_samples else indices[:0]
    return train_indices, val_indices


def train_selector(
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig | dict[str, Any],
) -> tuple[SelectorMLP, dict[str, float]]:
    """Train the same selector network used in nn_uncertainty/run.py."""
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required to train the selector model.")

    if isinstance(config, dict):
        cfg = TrainingConfig(
            hidden_dims=[int(value) for value in config["hidden_dims"]],
            dropout=float(config["dropout"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 1.0e-5)),
            seed=int(config.get("seed", 42)),
        )
    else:
        cfg = config

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_indices, val_indices = split_indices(len(features), cfg.seed)
    x_train = torch.from_numpy(features[train_indices])
    y_train = torch.from_numpy(labels[train_indices])
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    model = SelectorMLP(
        input_dim=features.shape[1],
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    last_train_loss = 0.0
    last_val_loss = 0.0
    for _ in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_x)
            sample_count += len(batch_x)
        last_train_loss = running_loss / max(sample_count, 1)

        if len(val_indices) > 0:
            model.eval()
            with torch.no_grad():
                x_val = torch.from_numpy(features[val_indices])
                y_val = torch.from_numpy(labels[val_indices])
                val_logits = model(x_val)
                last_val_loss = float(criterion(val_logits, y_val).item())
        else:
            last_val_loss = last_train_loss

    return model, {"train_loss": last_train_loss, "val_loss": last_val_loss}


def score_all_frames(model: SelectorMLP, features: np.ndarray) -> np.ndarray:
    """Convert logits to 0-1 selection scores."""
    if torch is None:
        raise RuntimeError("PyTorch is required to score frames.")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores.astype(np.float32)


def demo() -> None:
    """Small runnable example for learning the network flow."""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(200, 12)).astype(np.float32)
    labels = (features[:, 0] + 0.5 * features[:, 1] - 0.2 * features[:, 2] > 0.0).astype(np.float32)

    normalized_features, mean, std = standardize_features(features)
    config = TrainingConfig(
        hidden_dims=[64, 32],
        dropout=0.1,
        epochs=20,
        batch_size=32,
        learning_rate=1.0e-3,
        weight_decay=1.0e-5,
        seed=42,
    )
    model, losses = train_selector(normalized_features, labels, config)
    scores = score_all_frames(model, normalized_features)

    print(f"features shape: {features.shape}")
    print(f"mean shape: {mean.shape}, std shape: {std.shape}")
    print(f"train loss: {losses['train_loss']:.6f}")
    print(f"val loss: {losses['val_loss']:.6f}")
    print(f"score range: {float(scores.min()):.6f} -> {float(scores.max()):.6f}")


if __name__ == "__main__":
    demo()
