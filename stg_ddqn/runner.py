from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .environment import STGEnvironment
from .policies import Policy, make_policy


def run_single(cfg: dict, algorithm: str, seed: int,
               checkpoint: str | None = None, training: bool = False,
               policy: Policy | None = None) -> tuple[dict, list[dict], Policy]:
    selected_policy = policy or make_policy(
        algorithm, cfg, seed=seed + 10_000, checkpoint=checkpoint, training=training
    )
    environment = STGEnvironment(cfg, seed)
    aggregate, request_rows = environment.run(selected_policy, training=training)
    aggregate.update({
        "algorithm": algorithm,
        "config_digest": config_digest(cfg),
        "mobility_mode": cfg["mobility"]["mode"],
        "network_nodes": cfg["network"]["nodes"],
        "network_physical_links": cfg["network"]["physical_links"],
        "arrival_rate_req_s": cfg["traffic"]["arrival_rate_req_s"],
        "maximum_speed_m_s": cfg["mobility"]["maximum_speed_m_s"],
    })
    for row in request_rows:
        row.update({"algorithm": algorithm, "seed": seed})
    return aggregate, request_rows, selected_policy


def config_digest(cfg: dict) -> str:
    clean = deepcopy(cfg)
    clean.pop("_config_path", None)
    clean.pop("_config_dir", None)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

