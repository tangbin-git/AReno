Battleship agentic RL
=====================

Battleship is a partial-observability grid game for agentic RL. A hidden fleet
of four ships (sizes ``[4, 3, 2, 2]``, 11 cells) is placed on an 8×8 grid using
a seed. The policy calls ``fire(coordinate)`` once per turn and receives
``miss``, ``hit``, or ``sunk`` feedback. Ship cells are never revealed.
Repeated or out-of-range shots return ``status: "invalid"``. The game ends
when all ships are sunk or the 48-shot turn cap is reached.

.. code-block:: bash

   python examples/agentic/battleship/dataset_generator.py \
     --output /tmp/battleship.jsonl --count 256 --seed 2026

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path /tmp/battleship.jsonl \
     --dataset-loader-fn examples/agentic/battleship/dataset_loader.py \
     --reward-fn-path examples/agentic/battleship/reward.py \
     --agent-fn examples/agentic/battleship/run_agent.py \
     --algo gspo \
     --tp-size 1 \
     --world-size 1

Evaluate a trained policy against a random baseline:

.. code-block:: bash

   python examples/agentic/battleship/evaluate.py \
     --count 64 --seed 2026 \
     --base-url http://127.0.0.1:8000/v1 \
     --api-key token

The evaluation harness reports completion rate and shots-to-win for both the
random baseline and the trained policy in a comparison table and optional JSON.

Key adaptation points:

* The game state is owned by a per-episode ``BattleshipGame`` instance so
  ``fire`` results depend on prior shots.
* The reward function replays the tool-call sequence against a fresh game
  instance to compute the actual state transitions.
* The dataset loader falls back to the generator when the dataset file is
  missing, so training can run without a pre-built dataset.

See :doc:`/reference/agentic-rollout-api` for the agentic rollout API contract.

