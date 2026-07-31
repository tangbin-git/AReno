"""Dataset loader for the Battleship agentic training example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
from game import DEFAULT_MAX_TURNS, FLEET, make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL records and normalize them into Areno prompt records.

    When the dataset file is missing the loader falls back to the generator so
    the example can run without a pre-built dataset.
    """

    del default_loader
    records = _load_records(dataset_path)
    return [_format_record(raw, idx) for idx, raw in enumerate(records, start=1)]


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "battleship.jsonl"
    if not path.exists():
        return dataset_generator.generate_records()
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _format_record(raw: dict, index: int) -> dict:
    record = {
        "id": str(raw.get("id", f"battleship-{index:05d}")),
        "seed": int(raw["seed"]),
        "fleet": [int(s) for s in raw.get("fleet", FLEET)],
        "max_turns": int(raw.get("max_turns", DEFAULT_MAX_TURNS)),
    }
    record["prompt"] = make_prompt(record)
    return record

