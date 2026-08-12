from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import t as student_t
except ImportError:  # pragma: no cover
    student_t = None


PAPER_METRICS = (
    "acceptance_ratio_pct",
    "continuity_satisfaction_ratio_pct",
    "migration_mb_per_admitted_sfc",
    "downtime_ms_per_admitted_sfc",
    "mean_e2e_delay_ms",
    "p95_e2e_delay_ms",
    "mean_decision_runtime_ms",
    "p95_decision_runtime_ms",
    "mean_cpu_utilization_pct",
    "mean_bandwidth_utilization_pct",
)


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        column for column in ("experiment", "x_name", "x_value", "algorithm")
        if column in raw.columns
    ]
    rows: list[dict] = []
    for keys, group in raw.groupby(grouping, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(grouping, key_values))
        for metric in PAPER_METRICS:
            if metric not in group:
                continue
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            mean = float(np.mean(values))
            if len(values) > 1:
                standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
                critical = (
                    float(student_t.ppf(0.975, len(values) - 1))
                    if student_t is not None else 1.96
                )
                half_width = critical * standard_error
            else:
                half_width = 0.0
            rows.append({
                **base,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "ci95_half_width": half_width,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            })
    return pd.DataFrame(rows)


def write_results(raw_rows: list[dict], request_rows: list[dict],
                  output_dir: str | Path) -> tuple[Path, Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(raw_rows)
    raw_path = output / "evaluation_raw.csv"
    summary_path = output / "evaluation_summary.csv"
    requests_path = output / "request_records.csv"
    raw.to_csv(raw_path, index=False)
    summarize_results(raw).to_csv(summary_path, index=False)
    pd.DataFrame(request_rows).to_csv(requests_path, index=False)
    return raw_path, summary_path, requests_path

