from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency issue is environment-specific
    raise SystemExit("PyYAML is required to run nn_uncertainty.") from exc

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
class FrameRecord:
    frame_label: str
    frame_index: int
    structure_path: str


if nn is not None:
    class SelectorMLP(nn.Module):
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
    class SelectorMLP:  # pragma: no cover - used only when torch is missing
        pass


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def ensure_output_dirs(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    output_root = root / config["output"]["root"]
    selected_dir = output_root / config["output"]["selected_dir"]
    output_root.mkdir(exist_ok=True)
    selected_dir.mkdir(exist_ok=True)
    return output_root, selected_dir


def build_summary_path(output_root: Path, config: dict[str, Any]) -> Path:
    return output_root / config["output"]["summary_json"]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_frame_index(index_path: Path, raw_frames_dir: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            frame_label = (
                row.get("frame_label")
                or row.get("label")
                or row.get("frame")
                or row.get("frame_name")
                or f"frame_{idx:08d}"
            )
            frame_index_raw = row.get("frame_index") or row.get("index")
            frame_index = int(frame_index_raw) if frame_index_raw not in (None, "") else idx
            structure_path = row.get("structure_path") or row.get("path") or ""
            if not structure_path:
                structure_path = str((raw_frames_dir / f"{frame_label}.cif").resolve())
            records.append(
                FrameRecord(
                    frame_label=frame_label,
                    frame_index=frame_index,
                    structure_path=structure_path,
                )
            )
    return records


def load_event_labels(csv_path: Path, frame_labels: set[str]) -> set[str]:
    if not csv_path.exists():
        return set()
    selected: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_label = row.get("frame_label")
            if frame_label and frame_label in frame_labels:
                selected.add(frame_label)
    return selected


def build_labels(records: list[FrameRecord], config: dict[str, Any], root: Path) -> tuple[np.ndarray, dict[str, int]]:
    frame_labels = {record.frame_label for record in records}
    bond_labels = load_event_labels((root / config["labels"]["bond_events"]).resolve(), frame_labels)
    coordination_labels = load_event_labels(
        (root / config["labels"]["coordination_events"]).resolve(),
        frame_labels,
    )

    labels = np.zeros(len(records), dtype=np.float32)
    positives = 0
    for idx, record in enumerate(records):
        is_positive = record.frame_label in bond_labels or record.frame_label in coordination_labels
        if is_positive:
            labels[idx] = 1.0
            positives += 1

    counts = {
        "positive_labels": positives,
        "negative_labels": int(len(records) - positives),
        "bond_positive_frames": len(bond_labels),
        "coordination_positive_frames": len(coordination_labels),
    }
    return labels, counts


def standardize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def split_indices(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
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
    model_cfg: dict[str, Any],
    seed: int,
) -> tuple[SelectorMLP, dict[str, float]]:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required to train the selector model.")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_indices, val_indices = split_indices(len(features), seed)
    x_train = torch.from_numpy(features[train_indices])
    y_train = torch.from_numpy(labels[train_indices])
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(model_cfg["batch_size"]),
        shuffle=True,
    )

    model = SelectorMLP(
        input_dim=features.shape[1],
        hidden_dims=[int(value) for value in model_cfg["hidden_dims"]],
        dropout=float(model_cfg["dropout"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg.get("weight_decay", 1e-5)),
    )

    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1.0)],
        dtype=torch.float32,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    last_train_loss = 0.0
    last_val_loss = 0.0
    for _ in range(int(model_cfg["epochs"])):
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
    if torch is None:
        raise RuntimeError("PyTorch is required to score frames.")
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores.astype(np.float32)


def export_scores(
    path: Path,
    records: list[FrameRecord],
    labels: np.ndarray,
    scores: np.ndarray,
    score_column: str,
) -> None:
    fieldnames = ["frame_index", "frame_label", "structure_path", "label", score_column]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, label, score in zip(records, labels, scores):
            writer.writerow(
                {
                    "frame_index": record.frame_index,
                    "frame_label": record.frame_label,
                    "structure_path": record.structure_path,
                    "label": int(label),
                    score_column: f"{float(score):.6f}",
                }
            )


def select_records(
    records: list[FrameRecord],
    scores: np.ndarray,
    threshold: float,
    top_k: int,
) -> list[tuple[FrameRecord, float]]:
    ranked = sorted(
        zip(records, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    selected = [item for item in ranked if float(item[1]) >= threshold]
    if top_k > 0:
        selected = selected[:top_k]
    return [(record, float(score)) for record, score in selected]


def export_selected(path: Path, selected: list[tuple[FrameRecord, float]], score_column: str) -> None:
    fieldnames = ["frame_index", "frame_label", "structure_path", score_column]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, score in selected:
            writer.writerow(
                {
                    "frame_index": record.frame_index,
                    "frame_label": record.frame_label,
                    "structure_path": record.structure_path,
                    score_column: f"{score:.6f}",
                }
            )


def copy_selected_frames(selected: list[tuple[FrameRecord, float]], selected_dir: Path) -> int:
    copied = 0
    for record, _ in selected:
        source = Path(record.structure_path)
        if not source.exists():
            continue
        destination = selected_dir / source.name
        shutil.copy2(source, destination)
        copied += 1
    return copied


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    output_root, selected_dir = ensure_output_dirs(root, config)
    summary_path = build_summary_path(output_root, config)

    descriptor_path = (root / config["input"]["descriptors"]).resolve()
    frame_index_path = (root / config["input"]["frame_index"]).resolve()
    raw_frames_dir = (root / config["input"]["raw_frames"]).resolve()

    if torch is None:
        write_summary(
            summary_path,
            {
                "module": "nn_uncertainty",
                "role": "neural_network_structure_selector",
                "status": "missing_dependency",
                "message": "PyTorch is required to train the selector model.",
            },
        )
        print("PyTorch is not installed. Selector training was skipped.")
        return

    if not descriptor_path.exists() or not frame_index_path.exists():
        write_summary(
            summary_path,
            {
                "module": "nn_uncertainty",
                "role": "neural_network_structure_selector",
                "status": "missing_input",
                "descriptor_input": str(descriptor_path),
                "frame_index_input": str(frame_index_path),
                "message": "SOAP descriptors and frame index are required before training.",
            },
        )
        print("Missing descriptor inputs. Selector training was skipped.")
        return

    features = np.load(descriptor_path)
    records = load_frame_index(frame_index_path, raw_frames_dir)
    if features.ndim != 2:
        raise SystemExit("Descriptor array must be 2D: [num_frames, feature_dim].")
    if len(records) != len(features):
        raise SystemExit("Descriptor rows and frame index rows must match.")
    if len(records) < 2:
        write_summary(
            summary_path,
            {
                "module": "nn_uncertainty",
                "role": "neural_network_structure_selector",
                "status": "insufficient_samples",
                "num_samples": len(records),
                "message": "At least two samples are required to train the selector.",
            },
        )
        print("Not enough samples for selector training.")
        return

    labels, label_counts = build_labels(records, config, root)
    if int(label_counts["positive_labels"]) == 0:
        write_summary(
            summary_path,
            {
                "module": "nn_uncertainty",
                "role": "neural_network_structure_selector",
                "status": "missing_positive_labels",
                "num_samples": len(records),
                **label_counts,
                "message": "No positive labels were generated from upstream modules.",
            },
        )
        print("No positive labels found. Selector training was skipped.")
        return

    seed = int(config["model"].get("seed", 42))
    normalized_features, mean, std = standardize_features(features)
    model, losses = train_selector(normalized_features, labels, config["model"], seed)
    scores = score_all_frames(model, normalized_features)

    scores_csv = output_root / config["output"]["scores_csv"]
    selected_csv = output_root / config["output"]["selected_csv"]
    score_column = config["selection"]["score_column"]
    export_scores(scores_csv, records, labels, scores, score_column)

    selected = select_records(
        records,
        scores,
        threshold=float(config["selection"]["threshold"]),
        top_k=int(config["selection"]["top_k"]),
    )
    export_selected(selected_csv, selected, score_column)
    copied_count = copy_selected_frames(selected, selected_dir)

    summary = {
        "module": "nn_uncertainty",
        "role": "neural_network_structure_selector",
        "status": "completed",
        "descriptor_input": str(descriptor_path),
        "frame_index_input": str(frame_index_path),
        "raw_frames_input": str(raw_frames_dir),
        "num_samples": len(records),
        "feature_dim": int(features.shape[1]),
        **label_counts,
        **losses,
        "selection_threshold": float(config["selection"]["threshold"]),
        "selected_count": len(selected),
        "copied_selected_frames": copied_count,
        "normalization": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
    }
    write_summary(summary_path, summary)
    print(
        "nn_uncertainty selector completed: "
        f"{len(selected)} structures selected from {len(records)} samples."
    )


if __name__ == "__main__":
    main()
