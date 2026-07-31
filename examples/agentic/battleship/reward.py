"""Outcome and process reward for Battleship trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import DEFAULT_MAX_TURNS, FLEET, BattleshipGame  # noqa: E402


def reward_fn(record) -> float:
    """Reward efficient sinking, hits, and penalize invalid or wasted shots.

    Replays the tool-call sequence against a fresh ``BattleshipGame`` so the
    reward reflects the actual game state transitions, not just surface text.
    """

    source = dict(record.source_record)
    source.setdefault("fleet", list(FLEET))
    source.setdefault("max_turns", DEFAULT_MAX_TURNS)
    game = BattleshipGame(source)

    invalid_count = 0
    fire_found = False
    for call in record.tool_calls:
        if call.get("name") != "fire":
            continue
        fire_found = True
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
        if not isinstance(arguments, dict) or "coordinate" not in arguments:
            invalid_count += 1
            continue
        result = game.fire(arguments["coordinate"])
        if result["status"] == "invalid":
            invalid_count += 1

    if not fire_found:
        return -1.0

    hits = sum(len(h) for h in game.hits)
    sunk = sum(1 for s in game.sunk if s)
    efficiency = -0.002 * game.turns_used
    invalid_penalty = -0.02 * invalid_count
    if game.won:
        return 1.0 + efficiency + invalid_penalty
    return 0.05 * hits + 0.15 * sunk + efficiency + invalid_penalty

