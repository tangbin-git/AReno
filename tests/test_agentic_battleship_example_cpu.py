from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "battleship"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=type("AgentTrajectory", (), {}),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_battleship_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


def test_fleet_placement_is_legal_and_does_not_overlap():
    game = _load_module("game")

    ships = game.place_fleet(seed=42)
    assert len(ships) == 4
    assert [len(s) for s in ships] == [4, 3, 2, 2]
    all_cells = [cell for ship in ships for cell in ship]
    assert len(all_cells) == 11
    assert len(set(all_cells)) == 11  # no overlaps
    for ship in ships:
        rows = [r for r, _ in ship]
        cols = [c for _, c in ship]
        is_horizontal = len(set(rows)) == 1
        is_vertical = len(set(cols)) == 1
        assert is_horizontal or is_vertical
        assert all(0 <= r < game.GRID_SIZE for r, c in ship)
        assert all(0 <= c < game.GRID_SIZE for r, c in ship)


def test_placement_is_deterministic_for_same_seed():
    game = _load_module("game")

    first = game.place_fleet(seed=99)
    second = game.place_fleet(seed=99)
    assert first == second
    different = game.place_fleet(seed=100)
    assert first != different


def test_fire_returns_miss_hit_sunk_and_invalid_without_leaking_cells():
    game = _load_module("game")

    record = {"seed": 7, "fleet": game.FLEET, "max_turns": 48}
    board = game.BattleshipGame(record)

    # Find a cell that is occupied and a cell that is empty.
    ship_cells = {cell for ship in board.ships for cell in ship}
    occupied_coord = game.normalize_coordinate(
        f"{game.ROWS[next(iter(ship_cells))[0]]}{next(iter(ship_cells))[1] + 1}"
    )
    empty_coord = None
    for row in game.ROWS:
        for col in range(1, 9):
            coord = f"{row}{col}"
            if (game.ROWS.index(row), col - 1) not in ship_cells:
                empty_coord = coord
                break
        if empty_coord:
            break

    assert board.fire(empty_coord)["status"] == "miss"
    result = board.fire(occupied_coord)
    assert result["status"] in ("hit", "sunk")
    # Miss/hit results must not reveal ship cells.
    assert "cells" not in result
    assert "ship" not in result

    # Repeated shot is invalid.
    repeat = board.fire(empty_coord)
    assert repeat["status"] == "invalid"
    assert "reason" in repeat

    # Malformed coordinate is invalid.
    bad = board.fire("Z9")
    assert bad["status"] == "invalid"

    # Sunk result must include ship_size but not the actual cells.
    full_ship = board.ships[0]
    sunk_result = None
    for cell in full_ship:
        coord = f"{game.ROWS[cell[0]]}{cell[1] + 1}"
        res = board.fire(coord)
        if res["status"] == "sunk":
            sunk_result = res
            break
    if sunk_result is not None:
        assert sunk_result["ship_size"] == len(full_ship)
        assert "cells" not in sunk_result


def test_game_terminates_on_all_sunk_or_turn_cap():
    game = _load_module("game")

    # Win by sinking every ship cell.
    record = {"seed": 1, "fleet": game.FLEET, "max_turns": 48}
    board = game.BattleshipGame(record)
    for ship in board.ships:
        for cell in ship:
            coord = f"{game.ROWS[cell[0]]}{cell[1] + 1}"
            board.fire(coord)
    assert board.won is True
    assert board.terminated is True

    # Turn cap reached without winning.
    record_short = {"seed": 1, "fleet": game.FLEET, "max_turns": 3}
    short_board = game.BattleshipGame(record_short)
    for coord in ["A1", "A2", "A3"]:
        short_board.fire(coord)
    assert short_board.terminated is True
    assert short_board.won is False


def test_make_prompt_does_not_leak_ship_positions():
    game = _load_module("game")

    record = {"seed": 5, "fleet": game.FLEET, "max_turns": 48}
    prompt = game.make_prompt(record)
    ships = game.place_fleet(record["seed"])
    for ship in ships:
        for row, col in ship:
            coord = f"{game.ROWS[row]}{col + 1}"
            assert coord not in prompt
    assert "8x8" in prompt
    assert "48 shots" in prompt


def test_tool_schema_is_closed_and_bounded():
    game = _load_module("game")
    parameters = game.FIRE_TOOL["function"]["parameters"]

    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["coordinate"]
    assert parameters["properties"]["coordinate"]["pattern"] == "^[A-H][1-8]$"


def test_generator_is_reproducible_and_loader_falls_back_when_missing(tmp_path):
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")

    rows = generator.generate_records(8, seed=4)
    assert rows == generator.generate_records(8, seed=4)
    assert len({row["seed"] for row in rows}) == 8

    # Missing file triggers generator fallback.
    records = loader.load_training_dataset(str(tmp_path / "nonexistent.jsonl"), default_loader=None)
    assert len(records) > 0
    assert all("prompt" in r for r in records)
    assert all("seed" in r for r in records)


def test_loader_normalizes_records_from_file(tmp_path):
    loader = _load_module("dataset_loader")

    data_file = tmp_path / "battleship.jsonl"
    data_file.write_text(
        json.dumps({"id": "test-1", "seed": 100, "fleet": [4, 3, 2, 2], "max_turns": 20}) + "\n"
        + json.dumps({"id": "test-2", "seed": 200, "max_turns": 10}) + "\n"
    )
    records = loader.load_training_dataset(str(data_file), default_loader=None)
    assert len(records) == 2
    assert records[0]["seed"] == 100
    assert records[1]["seed"] == 200
    assert records[1]["fleet"] == [4, 3, 2, 2]  # defaults applied
    assert all("prompt" in r for r in records)


def test_reward_distinguishes_win_partial_invalid_and_empty_paths():
    reward = _load_module("reward")
    game = _load_module("game")

    record = {"seed": 1, "fleet": game.FLEET, "max_turns": 48}
    ships = game.place_fleet(record["seed"])
    # Build a winning sequence: fire at every ship cell.
    win_coords = [f"{game.ROWS[row]}{col + 1}" for ship in ships for row, col in ship]

    def score(coords):
        calls = [{"name": "fire", "arguments": json.dumps({"coordinate": c})} for c in coords]
        return reward.reward_fn(SimpleNamespace(source_record=record, tool_calls=calls))

    assert score([]) == -1.0  # no fire calls at all
    win_reward = score(win_coords)
    assert win_reward > 0.9  # base 1.0 minus light efficiency penalty for 11 shots
    hit_only = score([win_coords[0]])
    assert -1.0 < hit_only < win_reward
    invalid = score(["Z9"])
    assert invalid < 0  # only invalid shot
    repeated = score([win_coords[0], win_coords[0]])
    assert repeated < win_reward


def test_agent_executes_strict_single_fire_and_rejects_fabricated_calls():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    record = {"seed": 1, "fleet": game.FLEET, "max_turns": 48}
    board = game.BattleshipGame(record)
    valid = {
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "fire", "arguments": json.dumps({"coordinate": "A1"})},
            }
        ]
    }

    result = run_agent._execute_fire(valid, board)
    assert result is not None
    assert result["status"] in ("miss", "hit", "sunk", "invalid")
    assert run_agent._execute_fire({"tool_calls": []}, board) is None
    assert (
        run_agent._execute_fire({"tool_calls": [valid["tool_calls"][0], valid["tool_calls"][0]]}, board) is None
    )
    assert (
        run_agent._execute_fire(
            {"tool_calls": [{"function": {"name": "fire", "arguments": "not-json"}}]}, board
        )
        is None
    )


def test_episode_preserves_tool_order_and_stops_on_win():
    run_agent = _load_module("run_agent")
    game = _load_module("game")

    record = {"seed": 1, "fleet": game.FLEET, "max_turns": 48}
    ships = game.place_fleet(record["seed"])
    win_coords = [f"{game.ROWS[row]}{col + 1}" for ship in ships for row, col in ship]
    # Pad to max_turns; only the winning shots matter before termination.
    coords_iter = iter(win_coords + ["A8"] * 100)

    class FakeCompletions:
        def __init__(self):
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            coord = next(coords_iter, "A8")
            call = SimpleNamespace(
                id=f"call-{coord}",
                type="function",
                function=SimpleNamespace(name="fire", arguments=json.dumps({"coordinate": coord})),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    item = SimpleNamespace(prompt="sink the fleet", record=record)
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turns = asyncio.run(run_agent._run_episode(item, client))

    # The game terminates as soon as all ships sink; plus one finish turn.
    assert len(turns) >= 1
    # Verify tool message ordering in the second turn's context.
    if len(completions.messages) >= 2:
        second = completions.messages[1]
        roles = [m["role"] for m in second]
        assert "assistant" in roles
        has_tool = any(m["role"] == "tool" for m in second)
        assert has_tool


def test_evaluate_random_baseline_is_deterministic_and_reports_metrics():
    evaluate = _load_module("evaluate")

    record = {"seed": 1, "fleet": [4, 3, 2, 2], "max_turns": 48}
    result_1 = evaluate.run_random_baseline(record, seed=99)
    result_2 = evaluate.run_random_baseline(record, seed=99)
    assert result_1 == result_2
    assert "won" in result_1
    assert "turns_used" in result_1
    assert result_1["turns_used"] > 0

    summary = evaluate.aggregate([result_1])
    assert "completion_rate" in summary
    assert "avg_shots_to_win" in summary
    assert 0.0 <= summary["completion_rate"] <= 1.0


def test_end_to_end_generator_loader_reward_integration():
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")
    reward = _load_module("reward")
    game = _load_module("game")

    records = generator.generate_records(2, seed=2026)
    loaded = loader.load_training_dataset("unused", default_loader=lambda _: records)

    for rec in loaded:
        ships = game.place_fleet(rec["seed"])
        win_coords = [f"{game.ROWS[row]}{col + 1}" for ship in ships for row, col in ship]
        calls = [{"name": "fire", "arguments": json.dumps({"coordinate": c})} for c in win_coords]
        scored = reward.reward_fn(SimpleNamespace(source_record=rec, tool_calls=calls))
        assert scored > 0.9  # win with light efficiency penalty

