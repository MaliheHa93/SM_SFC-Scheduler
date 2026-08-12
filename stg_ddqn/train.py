from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from .config import load_config
from .policies import make_policy
from .runner import config_digest, run_single


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STG-DDQN with delayed migration transitions")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--policy", choices=("stg_ddqn", "graph_ddqn"), default="stg_ddqn")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/training")
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = Path(
        args.checkpoint or cfg["experiments"]["checkpoint"] or "checkpoints/stg_ddqn.pt"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in cfg["simulation"]["seeds"]]
    policy = make_policy(args.policy, cfg, seed=seeds[0] + 10_000, training=True)
    rows = []
    episodes = int(cfg["learning"]["training_episodes"])
    partial_checkpoint = checkpoint.with_name(checkpoint.name + ".partial")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for episode in range(episodes):
        seed = seeds[episode % len(seeds)] + 100_000 * (episode // len(seeds))
        row, _, policy = run_single(
            cfg, args.policy, seed, training=True, policy=policy
        )
        row["episode"] = episode + 1
        rows.append(row)
        # Keep the public checkpoint immutable until every configured episode
        # has finished.  Evaluation can therefore never load a mixture of
        # weights from an in-progress training run.
        policy.save(partial_checkpoint)
        pd.DataFrame(rows).to_csv(output / "training_log.csv", index=False)
        print(
            f"episode={episode + 1}/{episodes} seed={seed} "
            f"acceptance={row['acceptance_ratio_pct']:.2f}% "
            f"continuity={row['continuity_satisfaction_ratio_pct']:.2f}% "
            f"reward={row['mean_reward']:.4f}"
        )
    os.replace(partial_checkpoint, checkpoint)
    checkpoint_hash = _sha256(checkpoint)
    completion = {
        "status": "complete",
        "episodes": episodes,
        "config_digest": config_digest(cfg),
        "checkpoint_sha256": checkpoint_hash,
    }
    marker = checkpoint.with_name(checkpoint.name + ".complete.json")
    marker.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint={checkpoint.resolve()}")
    print(f"checkpoint_sha256={checkpoint_hash}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
