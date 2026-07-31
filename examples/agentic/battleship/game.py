"""Deterministic Battleship rules for the agentic RL example."""

from __future__ import annotations

import random
from typing import Any

GRID_SIZE = 8
FLEET = [4, 3, 2, 2]
DEFAULT_MAX_TURNS = 48
ROWS = "ABCDEFGH"

# Single tool: fire at one coordinate, returns miss/hit/sunk without leaking hidden cells.
FIRE_TOOL = {
    "type": "function",
    "function": {
        "name": "fire",
        "description": "Fire at one grid coordinate and receive miss, hit, or sunk feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinate": {
                    "type": "string",
                    "pattern": "^[A-H][1-8]$",
                    "description": "One grid coordinate, e.g. A1 or H8.",
                }
            },
            "required": ["coordinate"],
            "additionalProperties": False,
        },
    },
}


def normalize_coordinate(value: object) -> str:
    """Validate and normalize a board coordinate like ``A1``."""

    coord = str(value).strip().upper()
    if len(coord) != 2 or coord[0] not in ROWS or coord[1] not in "12345678":
        raise ValueError(f"coordinate must be like A1..H8, got {value!r}")
    return coord


def _cell_to_index(coord: str) -> tuple[int, int]:
    """Convert ``A1`` to a zero-based ``(row, col)`` index."""

    return (ROWS.index(coord[0]), int(coord[1]) - 1)


def place_fleet(seed: int, *, fleet: list[int] | None = None, size: int = GRID_SIZE) -> list[list[tuple[int, int]]]:
    """Return deterministic ship placements for the given seed.

    Each ship is a list of (row, col) cells.  Placement order is largest-first;
    ships never overlap and always stay on the grid.
    """

    sizes = list(fleet or FLEET)
    rng = random.Random(seed)
    ships: list[list[tuple[int, int]]] = []
    occupied: set[tuple[int, int]] = set()
    for ship_size in sizes:
        while True:
            horizontal = rng.random() < 0.5
            if horizontal:
                row = rng.randrange(size)
                col_start = rng.randrange(size - ship_size + 1)
                cells = [(row, col_start + i) for i in range(ship_size)]
            else:
                col = rng.randrange(size)
                row_start = rng.randrange(size - ship_size + 1)
                cells = [(row_start + i, col) for i in range(ship_size)]
            if not any(cell in occupied for cell in cells):
                ships.append(cells)
                occupied.update(cells)
                break
    return ships


class BattleshipGame:
    """Stateful environment for one Battleship episode.

    The fleet layout is derived from ``record["seed"]`` so identical seeds
    always produce identical games.  ``fire`` never reveals hidden cells.
    """

    def __init__(self, record: dict[str, Any]):
        self.seed = int(record["seed"])
        self.fleet = list(record.get("fleet", FLEET))
        self.max_turns = int(record.get("max_turns", DEFAULT_MAX_TURNS))
        self.ships = place_fleet(self.seed, fleet=self.fleet)
        self.shots: set[str] = set()
        self.hits: list[set[tuple[int, int]]] = [set() for _ in self.ships]
        self.sunk: list[bool] = [False] * len(self.ships)
        self.won = False
        self.terminated = False
        self.turns_used = 0

    def fire(self, coordinate: object) -> dict[str, Any]:
        """Process one shot, returning miss/hit/sunk/invalid without leaking ships."""

        if self.terminated:
            return {"status": "invalid", "reason": "game already ended"}
        try:
            coord = normalize_coordinate(coordinate)
        except ValueError as exc:
            return {"status": "invalid", "reason": str(exc)}
        if coord in self.shots:
            return {"status": "invalid", "reason": "coordinate already fired"}
        self.shots.add(coord)
        self.turns_used += 1
        row, col = _cell_to_index(coord)
        for idx, ship in enumerate(self.ships):
            if (row, col) in ship:
                self.hits[idx].add((row, col))
                if len(self.hits[idx]) == len(ship):
                    self.sunk[idx] = True
                    self._check_end()
                    return {"status": "sunk", "ship_size": len(ship)}
                return {"status": "hit"}
        self._check_end()
        return {"status": "miss"}

    def _check_end(self) -> None:
        if all(self.sunk):
            self.won = True
            self.terminated = True
        elif self.turns_used >= self.max_turns:
            self.terminated = True

    def summary(self) -> dict[str, Any]:
        """Return a non-leaking summary of the current state."""

        return {
            "won": self.won,
            "terminated": self.terminated,
            "turns_used": self.turns_used,
            "max_turns": self.max_turns,
            "shots_fired": len(self.shots),
            "ships_remaining": sum(1 for sunk in self.sunk if not sunk),
        }


def make_prompt(record: dict[str, Any]) -> str:
    """Build one prompt without leaking hidden ship cells."""

    max_turns = int(record.get("max_turns", DEFAULT_MAX_TURNS))
    total_cells = sum(record.get("fleet", FLEET))
    return (
        f"Sink the hidden fleet ({total_cells} cells across {len(record.get('fleet', FLEET))} ships) "
        f"on an 8x8 grid within {max_turns} shots. "
        "Coordinates use rows A-H and columns 1-8 (e.g. A1, H8). "
        "Call fire once per turn with one coordinate. Prior hits help you locate ships. "
        "Sunk feedback tells you a whole ship is destroyed; the exact cells are never revealed."
    )


def score_episode(game: BattleshipGame) -> float:
    """Compute the shaped reward for one completed episode.

    +1.0 for winning, +0.05 per hit, +0.15 per newly-sunk ship, -0.02 per invalid
    shot, and -0.002 per shot for efficiency shaping.  Invalid shots are counted
    via ``turns_used`` exceeding the number of legal shots attempted.
    """

    if game.won:
        return 1.0
    hits = sum(len(h) for h in game.hits)
    sunk = sum(1 for s in game.sunk if s)
    penalty = -0.002 * game.turns_used
    return 0.05 * hits + 0.15 * sunk + penalty

