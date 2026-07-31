"""Generate reproducible Battleship tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from game import DEFAULT_MAX_TURNS, FLEET


def generate_records(count: int = 256, *, seed: int = 2026) -> list[dict]:
    """Return deterministic seeded fleet layouts.

    Each record carries only the seed and fleet config; the actual placement is
    derived deterministically at game time so records stay compact.
    """

    return [
        {
            "id": f"battleship-{index:05d}",
            "seed": seed + index,
            "fleet": list(FLEET),
            "max_turns": DEFAULT_MAX_TURNS,
        }
        for index in range(1, count + 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    records = generate_records(args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

