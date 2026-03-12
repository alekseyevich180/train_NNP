from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    output_root = root / "outputs"
    output_root.mkdir(exist_ok=True)
    (output_root / "train").mkdir(exist_ok=True)
    (output_root / "val").mkdir(exist_ok=True)
    (output_root / "test").mkdir(exist_ok=True)

    summary = {
        "module": "final_training_dataset",
        "status": "scaffold",
        "next_step": "Implement candidate merge, deduplication, provenance tracking, and split export.",
    }
    (output_root / "provenance.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("final_training_dataset scaffold is ready.")


if __name__ == "__main__":
    main()
