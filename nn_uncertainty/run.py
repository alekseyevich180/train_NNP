from __future__ import annotations

import csv
import json
import math
import random
import re
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


EVENT_TOKEN_PATTERN = re.compile(r"\((formed|broken),[^)]*?,([A-Za-z]+-[A-Za-z]+),[^)]*\)")


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


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def parse_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def load_vdw_records(vdw_csv_path: Path) -> tuple[list[FrameRecord], dict[str, dict[str, float]], list[str]]:
    records: list[FrameRecord] = []
    feature_rows: dict[str, dict[str, float]] = {}
    with vdw_csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("vdw_scores.csv is missing a header row.")
        feature_columns = [
            name
            for name in reader.fieldnames
            if name not in {"frame_index", "frame_label", "structure_path"}
        ]
        for row in reader:
            frame_label = row["frame_label"]
            records.append(
                FrameRecord(
                    frame_label=frame_label,
                    frame_index=parse_int(row.get("frame_index"), len(records)),
                    structure_path=row.get("structure_path") or "",
                )
            )
            feature_rows[frame_label] = {
                column: parse_float(row.get(column), 0.0)
                for column in feature_columns
            }
    return records, feature_rows, feature_columns


def load_interface_features(csv_path: Path) -> dict[str, dict[str, float]]:
    if not csv_path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_label = row["frame_label"]
            rows[frame_label] = {
                "interface_organic_surface": parse_float(row.get("organic-surface"), 0.0),
                "interface_water_surface": parse_float(row.get("water-surface"), 0.0),
                "interface_organic_water": parse_float(row.get("organic-water"), 0.0),
            }
    return rows


def parse_bond_event_counts(encoded_events: str) -> dict[str, float]:
    counts = {
        "event_formed_count": 0.0,
        "event_broken_count": 0.0,
        "event_pair_cc": 0.0,
        "event_pair_co": 0.0,
        "event_pair_ozn": 0.0,
        "event_pair_ho": 0.0,
    }
    for event_type, pair_label in EVENT_TOKEN_PATTERN.findall(encoded_events):
        key = "event_formed_count" if event_type == "formed" else "event_broken_count"
        counts[key] += 1.0
        if pair_label == "C-C":
            counts["event_pair_cc"] += 1.0
        elif pair_label == "C-O":
            counts["event_pair_co"] += 1.0
        elif pair_label == "O-Zn":
            counts["event_pair_ozn"] += 1.0
        elif pair_label == "H-O":
            counts["event_pair_ho"] += 1.0
    return counts


def load_bond_features(csv_path: Path) -> dict[str, dict[str, float]]:
    if not csv_path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        events_column = next(
            (name for name in (reader.fieldnames or []) if name.startswith("events(")),
            "",
        )
        for row in reader:
            frame_label = row["frame_label"]
            features = {
                "event_count": parse_float(row.get("event_count"), 0.0),
            }
            features.update(parse_bond_event_counts(row.get(events_column, "")))
            rows[frame_label] = features
    return rows


def load_coordination_labels(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    labels: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_label = row.get("frame_label")
            if frame_label:
                labels.add(frame_label)
    return labels


def build_feature_matrix(records: list[FrameRecord], config: dict[str, Any], root: Path) -> tuple[np.ndarray, list[str], dict[str, float]]:
    input_cfg = config["input"]
    vdw_rows, bond_rows, interface_rows = {}, {}, {}
    _, vdw_rows, vdw_feature_columns = load_vdw_records((root / input_cfg["vdw_scores"]).resolve())
    bond_rows = load_bond_features((root / input_cfg["bond_events"]).resolve())
    interface_rows = load_interface_features((root / input_cfg["bond_interface_counts"]).resolve())

    feature_names = [
        *vdw_feature_columns,
        "event_count",
        "event_formed_count",
        "event_broken_count",
        "event_pair_cc",
        "event_pair_co",
        "event_pair_ozn",
        "event_pair_ho",
        "interface_organic_surface",
        "interface_water_surface",
        "interface_organic_water",
        "interface_total",
        "event_delta",
        "vdw_event_coupling",
    ]

    matrix: list[list[float]] = []
    max_event_count = 0.0
    max_vdw_score = 0.0
    for record in records:
        vdw = dict(vdw_rows.get(record.frame_label, {}))
        bond = dict(bond_rows.get(record.frame_label, {}))
        interface = dict(interface_rows.get(record.frame_label, {}))

        interface_total = (
            float(interface.get("interface_organic_surface", 0.0))
            + float(interface.get("interface_water_surface", 0.0))
            + float(interface.get("interface_organic_water", 0.0))
        )
        event_delta = float(bond.get("event_formed_count", 0.0)) - float(bond.get("event_broken_count", 0.0))
        vdw_total_score = float(vdw.get(config["labels"]["vdw_score_column"], 0.0))
        vdw_event_coupling = vdw_total_score * (1.0 + float(bond.get("event_count", 0.0)))

        row = {
            **{name: float(vdw.get(name, 0.0)) for name in vdw_feature_columns},
            "event_count": float(bond.get("event_count", 0.0)),
            "event_formed_count": float(bond.get("event_formed_count", 0.0)),
            "event_broken_count": float(bond.get("event_broken_count", 0.0)),
            "event_pair_cc": float(bond.get("event_pair_cc", 0.0)),
            "event_pair_co": float(bond.get("event_pair_co", 0.0)),
            "event_pair_ozn": float(bond.get("event_pair_ozn", 0.0)),
            "event_pair_ho": float(bond.get("event_pair_ho", 0.0)),
            "interface_organic_surface": float(interface.get("interface_organic_surface", 0.0)),
            "interface_water_surface": float(interface.get("interface_water_surface", 0.0)),
            "interface_organic_water": float(interface.get("interface_organic_water", 0.0)),
            "interface_total": interface_total,
            "event_delta": event_delta,
            "vdw_event_coupling": vdw_event_coupling,
        }
        max_event_count = max(max_event_count, row["event_count"])
        max_vdw_score = max(max_vdw_score, vdw_total_score)
        matrix.append([row[name] for name in feature_names])

    metadata = {"max_event_count": max_event_count, "max_vdw_score": max_vdw_score}
    return np.asarray(matrix, dtype=np.float32), feature_names, metadata


def build_labels(records: list[FrameRecord], config: dict[str, Any], root: Path) -> tuple[np.ndarray, dict[str, int | float]]:
    input_cfg = config["input"]
    label_cfg = config["labels"]
    bond_rows = load_bond_features((root / input_cfg["bond_events"]).resolve())
    interface_rows = load_interface_features((root / input_cfg["bond_interface_counts"]).resolve())
    _, vdw_rows, _ = load_vdw_records((root / input_cfg["vdw_scores"]).resolve())
    coordination_labels = load_coordination_labels((root / input_cfg["coordination_events"]).resolve())

    vdw_values = [
        float(row.get(label_cfg["vdw_score_column"], 0.0))
        for row in vdw_rows.values()
    ]
    percentile = float(label_cfg.get("vdw_positive_percentile", 90.0))
    vdw_threshold = float(np.percentile(vdw_values, percentile)) if vdw_values else math.inf

    labels = np.zeros(len(records), dtype=np.float32)
    positives = 0
    bond_positive = 0
    interface_positive = 0
    vdw_positive = 0
    coordination_positive = 0

    for idx, record in enumerate(records):
        bond = bond_rows.get(record.frame_label, {})
        interface = interface_rows.get(record.frame_label, {})
        vdw = vdw_rows.get(record.frame_label, {})

        has_bond_change = float(bond.get("event_count", 0.0)) >= float(label_cfg.get("min_event_count", 1.0))
        interface_total = (
            float(interface.get("interface_organic_surface", 0.0))
            + float(interface.get("interface_water_surface", 0.0))
            + float(interface.get("interface_organic_water", 0.0))
        )
        has_interface_change = interface_total >= float(label_cfg.get("min_interface_count", 1.0))
        has_vdw_signal = float(vdw.get(label_cfg["vdw_score_column"], 0.0)) >= vdw_threshold
        has_coordination_signal = record.frame_label in coordination_labels

        is_positive = has_bond_change or has_interface_change or has_vdw_signal or has_coordination_signal
        labels[idx] = 1.0 if is_positive else 0.0
        positives += int(is_positive)
        bond_positive += int(has_bond_change)
        interface_positive += int(has_interface_change)
        vdw_positive += int(has_vdw_signal)
        coordination_positive += int(has_coordination_signal)

    counts = {
        "positive_labels": positives,
        "negative_labels": int(len(records) - positives),
        "bond_positive_frames": bond_positive,
        "interface_positive_frames": interface_positive,
        "vdw_positive_frames": vdw_positive,
        "coordination_positive_frames": coordination_positive,
        "vdw_positive_threshold": float(vdw_threshold if math.isfinite(vdw_threshold) else 0.0),
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
    feature_matrix: np.ndarray,
    feature_names: list[str],
    score_column: str,
) -> None:
    fieldnames = ["frame_index", "frame_label", "structure_path", "label", score_column, *feature_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, label, score, features in zip(records, labels, scores, feature_matrix, strict=True):
            row = {
                "frame_index": record.frame_index,
                "frame_label": record.frame_label,
                "structure_path": record.structure_path,
                "label": int(label),
                score_column: f"{float(score):.6f}",
            }
            for name, value in zip(feature_names, features, strict=True):
                row[name] = f"{float(value):.6f}"
            writer.writerow(row)


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
    summary_path = output_root / config["output"]["summary_json"]

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

    vdw_csv_path = (root / config["input"]["vdw_scores"]).resolve()
    if not vdw_csv_path.exists():
        write_summary(
            summary_path,
            {
                "module": "nn_uncertainty",
                "role": "neural_network_structure_selector",
                "status": "missing_input",
                "vdw_scores_input": str(vdw_csv_path),
                "message": "Run vdw_energy_predictor before training the selector.",
            },
        )
        print("Missing vdw_scores.csv. Selector training was skipped.")
        return

    records, _, _ = load_vdw_records(vdw_csv_path)
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

    feature_matrix, feature_names, feature_metadata = build_feature_matrix(records, config, root)
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
                "message": "No positive labels were generated from the heuristic rules.",
            },
        )
        print("No positive labels found. Selector training was skipped.")
        return

    seed = int(config["model"].get("seed", 42))
    normalized_features, mean, std = standardize_features(feature_matrix)
    model, losses = train_selector(normalized_features, labels, config["model"], seed)
    scores = score_all_frames(model, normalized_features)

    scores_csv = output_root / config["output"]["scores_csv"]
    selected_csv = output_root / config["output"]["selected_csv"]
    score_column = config["selection"]["score_column"]
    export_scores(scores_csv, records, labels, scores, feature_matrix, feature_names, score_column)

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
        "vdw_scores_input": str(vdw_csv_path),
        "num_samples": len(records),
        "feature_dim": int(feature_matrix.shape[1]),
        "feature_names": feature_names,
        **feature_metadata,
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
