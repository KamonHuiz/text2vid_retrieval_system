"""Auto-generates the Phase-1-vs-Phase-2 comparison report.

Reads ``{results_dir}/{model_name}/baseline_metrics.json`` and
``{results_dir}/{model_name}/finetuned_metrics.json`` (as written by
``evaluation.evaluator.Evaluator.run``) for every model, and produces a long
table of ``(model, direction, metric, v1, v2, delta)`` rows, plus a
markdown summary grouped by model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _load_metrics_flat(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data["metrics_flat"]


def build_comparison_table(
    results_dir: str,
    model_names: List[str],
    baseline_filename: str = "baseline_metrics.json",
    finetuned_filename: str = "finetuned_metrics.json",
) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        model_dir = Path(results_dir) / model_name
        v1 = _load_metrics_flat(model_dir / baseline_filename)
        v2 = _load_metrics_flat(model_dir / finetuned_filename)

        if v1 is None and v2 is None:
            continue

        metric_keys = sorted(set((v1 or {}).keys()) | set((v2 or {}).keys()))
        for metric_key in metric_keys:
            if metric_key == "n_samples":
                continue
            v1_val = (v1 or {}).get(metric_key)
            v2_val = (v2 or {}).get(metric_key)
            delta = (v2_val - v1_val) if (v1_val is not None and v2_val is not None) else None
            direction, metric = metric_key.split("/", 1)
            rows.append(
                {
                    "model": model_name,
                    "direction": direction,
                    "metric": metric,
                    "v1_baseline": v1_val,
                    "v2_finetuned": v2_val,
                    "delta": delta,
                }
            )
    return pd.DataFrame(rows)


def write_comparison_csv(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_comparison_markdown(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Fine-Tuning Impact: Phase 1 (Baseline) vs Phase 2 (Fine-Tuned)", ""]
    for model_name, group in df.groupby("model"):
        lines.append(f"## {model_name}")
        lines.append("")
        pivot = group.pivot(index="metric", columns="direction", values=["v1_baseline", "v2_finetuned", "delta"])
        lines.append(pivot.to_markdown(floatfmt=".4f"))
        lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def build_comparison_report(
    results_dir: str,
    model_names: List[str],
    output_csv: str,
    output_markdown: str,
) -> pd.DataFrame:
    df = build_comparison_table(results_dir, model_names)
    if df.empty:
        raise FileNotFoundError(
            f"No baseline/finetuned metrics found under {results_dir} for models {model_names}. "
            "Run scripts/benchmark.py --phase baseline and --phase finetuned for each model first."
        )
    write_comparison_csv(df, output_csv)
    write_comparison_markdown(df, output_markdown)
    return df
