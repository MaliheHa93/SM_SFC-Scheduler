from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil

import pandas as pd

from .config import load_config
from .runner import run_single
from .stats import write_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traffic, mobility, and scalability sweeps")
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/paper")
    parser.add_argument("--experiments", default="traffic,mobility,scalability")
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated algorithm override, useful for partitioned runs",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated paired-seed override, useful for partitioned runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted sweep from the existing raw CSV",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent algorithm/seed runs to execute concurrently",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    requested = {value.strip() for value in args.experiments.split(",")}
    algorithms = (
        [value.strip() for value in args.algorithms.split(",") if value.strip()]
        if args.algorithms else list(cfg["experiments"]["algorithms"])
    )
    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds else [int(seed) for seed in cfg["simulation"]["seeds"]]
    )
    checkpoint = args.checkpoint or cfg["experiments"].get("checkpoint")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frozen_checkpoint, checkpoint_hash = _freeze_checkpoint(
        checkpoint,
        output,
        resume=bool(args.resume),
        required=any(name in {"stg_ddqn", "graph_ddqn"} for name in algorithms),
    )
    if any(name in {"stg_ddqn", "graph_ddqn"} for name in algorithms):
        if frozen_checkpoint is None:
            raise FileNotFoundError(
                "A trained checkpoint is required for the learning schemes. "
                "Run python -m stg_ddqn.train first."
            )
    raw_path = output / "evaluation_raw.csv"
    request_path = output / "request_records.csv"
    if args.resume and raw_path.exists():
        raw_rows = pd.read_csv(raw_path).to_dict("records")
        request_rows = (
            pd.read_csv(request_path).to_dict("records")
            if request_path.exists() else []
        )
    else:
        raw_rows, request_rows = [], []
    recorded_hashes = {
        str(row.get("checkpoint_sha256"))
        for row in raw_rows
        if pd.notna(row.get("checkpoint_sha256"))
    }
    if checkpoint_hash and recorded_hashes and recorded_hashes != {checkpoint_hash}:
        raise ValueError(
            "The resumed result directory was created with a different "
            "STG-DDQN checkpoint. Use a new output directory."
        )
    completed_keys = {
        (
            str(row["experiment"]),
            float(row["x_value"]),
            str(row["algorithm"]),
            int(row["seed"]),
        )
        for row in raw_rows
    }
    scenarios = list(_scenarios(cfg, requested))
    total = len(scenarios) * len(algorithms) * len(seeds)
    completed = 0
    jobs = []
    for experiment, x_name, x_value, scenario_cfg in scenarios:
        for algorithm in algorithms:
            for seed in seeds:
                key = (experiment, float(x_value), algorithm, int(seed))
                if key in completed_keys:
                    completed += 1
                    print(
                        f"[{completed}/{total}] resume-skip {experiment} "
                        f"{x_value} {algorithm} seed={seed}"
                    )
                    continue
                selected_checkpoint = (
                    str(frozen_checkpoint)
                    if algorithm in {"stg_ddqn", "graph_ddqn"} else None
                )
                jobs.append((
                    scenario_cfg,
                    algorithm,
                    seed,
                    selected_checkpoint,
                    str(Path(checkpoint)) if selected_checkpoint else None,
                    experiment,
                    x_name,
                    x_value,
                    checkpoint_hash or "not_applicable",
                ))

    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            completed = _record_result(
                _run_job(job), raw_rows, request_rows, completed_keys,
                output, completed, total,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                completed = _record_result(
                    future.result(), raw_rows, request_rows, completed_keys,
                    output, completed, total,
                )
    paths = write_results(raw_rows, request_rows, output)
    print("\n".join(str(path.resolve()) for path in paths))


def _freeze_checkpoint(
    checkpoint: str | None,
    output: Path,
    resume: bool,
    required: bool,
) -> tuple[Path | None, str | None]:
    if not required:
        return None, None
    if not checkpoint:
        return None, None
    source = Path(checkpoint)
    frozen = output / "frozen_stg_ddqn.pt"
    if resume and frozen.exists():
        return frozen, _sha256(frozen)
    if not source.exists():
        return None, None
    temporary = frozen.with_name(frozen.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, frozen)
    return frozen, _sha256(frozen)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenarios(cfg: dict, requested: set[str]):
    experiments = cfg["experiments"]
    if "traffic" in requested:
        for arrival_rate in experiments["traffic_arrival_rates"]:
            scenario = deepcopy(cfg)
            scenario["traffic"]["arrival_rate_req_s"] = float(arrival_rate)
            scenario["mobility"]["maximum_speed_m_s"] = float(
                experiments["traffic_fixed_maximum_speed_m_s"]
            )
            yield "traffic", "arrival_rate_req_s", float(arrival_rate), scenario
    if "mobility" in requested:
        for maximum_speed in experiments["mobility_speeds_m_s"]:
            scenario = deepcopy(cfg)
            scenario["mobility"]["maximum_speed_m_s"] = float(maximum_speed)
            scenario["traffic"]["arrival_rate_req_s"] = float(
                experiments["mobility_fixed_arrival_rate_req_s"]
            )
            yield "mobility", "maximum_speed_m_s", float(maximum_speed), scenario
    if "scalability" in requested:
        for nodes in experiments["scalability_nodes"]:
            scenario = deepcopy(cfg)
            scenario["network"]["nodes"] = int(nodes)
            scenario["network"]["physical_links"] = max(int(nodes) - 1, 2 * int(nodes) - 5)
            yield "scalability", "nodes", int(nodes), scenario


def _run_job(job):
    (
        scenario_cfg,
        algorithm,
        seed,
        checkpoint,
        checkpoint_source,
        experiment,
        x_name,
        x_value,
        checkpoint_hash,
    ) = job
    if checkpoint:
        selected = Path(checkpoint)
        if not selected.exists() and checkpoint_source:
            selected = Path(checkpoint_source)
        if not selected.exists():
            raise FileNotFoundError(
                "The frozen and source STG-DDQN checkpoints are both missing"
            )
        actual_hash = _sha256(selected)
        if actual_hash != checkpoint_hash:
            raise ValueError(
                "The STG-DDQN checkpoint changed during evaluation; "
                "start a new result directory"
            )
        checkpoint = str(selected)
    row, requests, _ = run_single(
        scenario_cfg,
        algorithm,
        seed,
        checkpoint=checkpoint,
        training=False,
    )
    row.update({
        "experiment": experiment,
        "x_name": x_name,
        "x_value": x_value,
        "checkpoint_sha256": checkpoint_hash,
    })
    for request_row in requests:
        request_row.update({
            "experiment": experiment,
            "x_name": x_name,
            "x_value": x_value,
        })
    return row, requests


def _record_result(
    result,
    raw_rows: list[dict],
    request_rows: list[dict],
    completed_keys: set[tuple],
    output: Path,
    completed: int,
    total: int,
) -> int:
    row, requests = result
    key = (
        str(row["experiment"]),
        float(row["x_value"]),
        str(row["algorithm"]),
        int(row["seed"]),
    )
    raw_rows.append(row)
    request_rows.extend(requests)
    completed_keys.add(key)
    completed += 1
    print(
        f"[{completed}/{total}] {row['experiment']} {row['x_value']} "
        f"{row['algorithm']} seed={row['seed']}",
        flush=True,
    )
    raw_rows.sort(key=lambda item: (
        str(item.get("experiment", "")),
        float(item.get("x_value", 0.0)),
        str(item.get("algorithm", "")),
        int(item.get("seed", 0)),
    ))
    request_rows.sort(key=lambda item: (
        str(item.get("experiment", "")),
        float(item.get("x_value", 0.0)),
        str(item.get("algorithm", "")),
        int(item.get("seed", 0)),
        int(item.get("request_id", 0)),
    ))
    # Save after every completed realization so an interrupted paper run loses
    # at most the still-running worker results.
    write_results(raw_rows, request_rows, output)
    return completed


if __name__ == "__main__":
    main()
