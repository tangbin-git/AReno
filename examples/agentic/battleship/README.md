# Agentic Battleship RL

Battleship is a deterministic, partial-observability grid game for agentic RL.
A hidden fleet of four ships (`[4, 3, 2, 2]`, 11 cells total) is placed on an
8×8 grid using a seed. The policy calls `fire(coordinate)` once per turn and
receives `miss`, `hit`, or `sunk` feedback. Ship cells are never revealed —
only the outcome of each shot. Repeated or out-of-range shots return
`status: "invalid"`. The game ends when all ships are sunk (win) or the
48-shot turn cap is reached (loss).

Coordinates use rows A–H and columns 1–8 (e.g. `A1`, `H8`).

## Generate data

```bash
python examples/agentic/battleship/dataset_generator.py \
  --output /tmp/battleship.jsonl --count 256 --seed 2026
```

When the dataset file is missing the loader falls back to the generator, so
you can also pass any path and the default records are produced on the fly.

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/battleship.jsonl \
  --dataset-loader-fn examples/agentic/battleship/dataset_loader.py \
  --reward-fn-path examples/agentic/battleship/reward.py \
  --agent-fn examples/agentic/battleship/run_agent.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 32 \
  --n-samples 8 \
  --max-new-tokens 32
```

## Evaluate

Run the random baseline only (no GPU or network needed):

```bash
python examples/agentic/battleship/evaluate.py \
  --count 64 --seed 2026
```

Evaluate a trained policy against the random baseline by pointing at an
`areno serve` endpoint:

```bash
areno serve --model-path /path/to/trained --tp-size 1 --port 8000 &
python examples/agentic/battleship/evaluate.py \
  --count 64 --seed 2026 \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key token \
  --output /tmp/battleship_eval.json
```

Output includes a human-readable comparison table and, when `--output` is set,
a JSON file with completion rate and shots-to-win for both the random baseline
and the trained policy.

## Reward signal

| Event | Reward |
|---|---|
| Win (all ships sunk within turn cap) | +1.0 |
| Each hit | +0.05 |
| Each newly-sunk ship | +0.15 |
| Each invalid/repeated shot | -0.02 |
| Efficiency shaping | -0.002 × shots used |

No fire call at all returns -1.0.

