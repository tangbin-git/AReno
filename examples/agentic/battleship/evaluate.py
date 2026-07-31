"""CPU evaluation harness comparing a random baseline with a trained policy.

Run the random baseline only (no network needed)::

    python examples/agentic/battleship/evaluate.py --count 64 --seed 2026

Evaluate a trained policy served by ``areno serve`` against the same boards::

    python examples/agentic/battleship/evaluate.py --count 64 --seed 2026 \
        --base-url http://127.0.0.1:8000/v1 --api-key token
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_loader  # noqa: E402
from game import FIRE_TOOL, GRID_SIZE, ROWS, BattleshipGame  # noqa: E402

logger = logging.getLogger(__name__)


def run_random_baseline(record: dict, *, seed: int) -> dict:
    """Play one board with uniform random legal shot selection."""

    game = BattleshipGame(record)
    rng = random.Random(seed)
    all_coords = [f"{row}{col}" for row in ROWS for col in range(1, GRID_SIZE + 1)]
    rng.shuffle(all_coords)
    for coord in all_coords:
        if game.terminated:
            break
        game.fire(coord)
    return _summarize(game)


async def run_policy_episode(record: dict, client) -> dict:
    """Play one board using a trained policy via an OpenAI-compatible endpoint."""

    game = BattleshipGame(record)
    messages = [
        {"role": "system", "content": "You are a Battleship player. Call fire once per turn."},
        {"role": "user", "content": record["prompt"]},
    ]
    tool_choice = {"type": "function", "function": {"name": "fire"}}
    for shot_number in range(1, game.max_turns + 1):
        turn_messages = [
            *messages,
            {"role": "user", "content": f"Shot {shot_number} of {game.max_turns}: call fire now."},
        ]
        response = await client.chat.completions.create(
            model="policy",
            messages=turn_messages,
            tools=[FIRE_TOOL],
            tool_choice=tool_choice,
            stream=False,
        )
        assistant = response.choices[0].message
        calls = assistant.tool_calls or []
        if len(calls) != 1 or calls[0].function.name != "fire":
            break
        try:
            arguments = json.loads(calls[0].function.arguments or "")
        except (json.JSONDecodeError, TypeError):
            break
        if not isinstance(arguments, dict) or "coordinate" not in arguments:
            break
        result = game.fire(arguments["coordinate"])
        messages.append(
            {
                "role": "assistant",
                "content": assistant.content,
                "tool_calls": [
                    {
                        "id": calls[0].id,
                        "type": calls[0].type,
                        "function": {"name": "fire", "arguments": calls[0].function.arguments},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": calls[0].id, "name": "fire", "content": json.dumps(result)})
        if game.terminated:
            break
    return _summarize(game)


def _summarize(game: BattleshipGame) -> dict:
    return {
        "won": game.won,
        "turns_used": game.turns_used,
        "shots_to_win": game.turns_used if game.won else None,
        "ships_sunk": sum(1 for s in game.sunk if s),
        "total_ships": len(game.ships),
    }


def aggregate(results: list[dict]) -> dict:
    """Compute completion rate and shots-to-win metrics."""

    wins = [r for r in results if r["won"]]
    completed = len(wins)
    completion_rate = completed / len(results) if results else 0.0
    shots_to_win = [r["turns_used"] for r in wins]
    avg_shots = sum(shots_to_win) / len(shots_to_win) if shots_to_win else 0.0
    return {
        "episodes": len(results),
        "completed": completed,
        "completion_rate": round(completion_rate, 4),
        "avg_shots_to_win": round(avg_shots, 2),
        "min_shots_to_win": min(shots_to_win) if shots_to_win else None,
        "max_shots_to_win": max(shots_to_win) if shots_to_win else None,
    }


def print_table(baseline: dict, policy: dict | None) -> None:
    """Print a human-readable comparison table."""

    rows = [
        ("Episodes", baseline["episodes"], policy["episodes"] if policy else "-"),
        ("Completed", baseline["completed"], policy["completed"] if policy else "-"),
        ("Completion rate", f"{baseline['completion_rate']:.4f}", f"{policy['completion_rate']:.4f}" if policy else "-"),
        ("Avg shots to win", f"{baseline['avg_shots_to_win']:.2f}", f"{policy['avg_shots_to_win']:.2f}" if policy else "-"),
        (
            "Min shots to win",
            baseline["min_shots_to_win"] or "-",
            policy["min_shots_to_win"] if policy else "-",
        ),
        (
            "Max shots to win",
            baseline["max_shots_to_win"] or "-",
            policy["max_shots_to_win"] if policy else "-",
        ),
    ]
    print(f"{'Metric':<20} {'Random baseline':<20} {'Trained policy':<20}")
    print("-" * 60)
    for metric, base_val, pol_val in rows:
        print(f"{metric:<20} {str(base_val):<20} {str(pol_val):<20}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="battleship", help="JSONL path or directory; defaults to generator")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint for trained policy evaluation")
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--output", type=Path, help="Write JSON results to this path")
    args = parser.parse_args()

    records = dataset_loader.load_training_dataset(args.dataset_path, default_loader=None)
    records = records[: args.count]

    baseline_results = [run_random_baseline(rec, seed=args.seed + i) for i, rec in enumerate(records)]
    baseline_summary = aggregate(baseline_results)

    policy_summary = None
    policy_results = None
    if args.base_url:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("Policy evaluation requires `pip install openai`") from exc
        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

        async def _run_all():
            return await asyncio.gather(*(run_policy_episode(rec, client) for rec in records))

        policy_results = asyncio.run(_run_all())
        policy_summary = aggregate(policy_results)

    print_table(baseline_summary, policy_summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(
                {"baseline": baseline_summary, "trained_policy": policy_summary},
                handle,
                indent=2,
            )
        print(f"\nJSON results written to {args.output}")


if __name__ == "__main__":
    main()

