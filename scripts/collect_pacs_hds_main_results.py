#!/usr/bin/env python
"""Collect PACS HDS-Main experiment results from configs and training logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


TARGET_NAMES = {
    "a": "art_painting",
    "c": "cartoon",
    "p": "photo",
    "s": "sketch",
}

ACC_PATTERNS = [
    re.compile(r"'test': \{'acc_avg': ([0-9.eE+-]+)\}"),
    re.compile(r'"test"\s*:\s*\{\s*"acc_avg"\s*:\s*([0-9.eE+-]+)'),
    re.compile(r"test[/_. -]?acc(?:_avg)?[=: ]+([0-9.eE+-]+)", re.IGNORECASE),
    re.compile(r"final[_ -]?test[_ -]?acc[=: ]+([0-9.eE+-]+)", re.IGNORECASE),
]

REQUIRED_COLUMNS = [
    "target_domain",
    "seed",
    "client_method",
    "server_method",
    "partition_eta",
    "partition_lambda",
    "accuracy",
    "avg_accuracy",
    "std",
    "gain_over_HierarchicalFedAvg",
    "gain_over_HFedATM",
]


def target_from_split(split_scheme: str) -> str:
    if "-" in split_scheme:
        code = split_scheme.rsplit("-", 1)[-1]
        return TARGET_NAMES.get(code, code)
    return split_scheme


def parse_last_accuracy(log_path: Path):
    if not log_path.exists():
        return None, "missing_log"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = []
    for pattern in ACC_PATTERNS:
        matches.extend(pattern.findall(text))
    if "Traceback (most recent call last)" in text:
        status = "failed_traceback"
    else:
        status = "ok" if matches else "missing_accuracy"
    if not matches:
        return None, status
    try:
        return float(matches[-1]), status
    except ValueError:
        return None, "bad_accuracy"


def find_wandb_output_log(row, wandb_dir: Path):
    """Find wandb's redirected stdout log for a config-backed run.

    The tmux launcher redirects stdout to outputs/.../logs, but wandb wraps
    stdout after initialization. Later metrics can therefore land in
    wandb/run-*/files/output.log instead of the launcher log.
    """
    if not wandb_dir.exists():
        return None

    config_path = str(row["config_path"]).replace("\\", "/")
    config_name = Path(row["config_path"]).name
    run_id = row["run_id"]
    candidates = []
    for metadata_path in wandb_dir.glob("run-*/files/wandb-metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        args = " ".join(str(arg) for arg in metadata.get("args", [])).replace("\\", "/")
        if config_path not in args and config_name not in args and run_id not in args:
            continue
        output_log = metadata_path.parent / "output.log"
        if output_log.exists():
            candidates.append(output_log)

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_configs(config_dir: Path):
    rows = []
    for cfg_path in sorted(config_dir.glob("*.json")):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        run_id = cfg.get("id", cfg_path.stem)
        split_scheme = cfg.get("split_scheme", "")
        rows.append(
            {
                "run_id": run_id,
                "config_path": str(cfg_path),
                "split_scheme": split_scheme,
                "target_domain": target_from_split(split_scheme),
                "seed": int(cfg.get("seed", 0)),
                "client_method": cfg.get("client_method", ""),
                "server_method": cfg.get("server_method", ""),
                "partition_eta": float(cfg.get("partition_eta", cfg.get("partition_alpha_station", math.nan))),
                "partition_lambda": float(cfg.get("partition_lambda", cfg.get("partition_alpha_client", math.nan))),
            }
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["accuracy"] is None:
            continue
        key = (row["target_domain"], row["client_method"], row["server_method"])
        grouped[key].append(row["accuracy"])

    stats = {}
    for key, values in grouped.items():
        avg = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[key] = (avg, std)

    baseline_by_seed = {}
    for row in rows:
        if row["accuracy"] is None:
            continue
        key = (row["target_domain"], row["client_method"], row["seed"], row["server_method"])
        baseline_by_seed[key] = row["accuracy"]

    for row in rows:
        key = (row["target_domain"], row["client_method"], row["server_method"])
        avg, std = stats.get(key, (None, None))
        row["avg_accuracy"] = avg
        row["std"] = std
        base_key = (row["target_domain"], row["client_method"], row["seed"], "HierarchicalFedAvg")
        atm_key = (row["target_domain"], row["client_method"], row["seed"], "HFedATM")
        row["gain_over_HierarchicalFedAvg"] = (
            None if row["accuracy"] is None or base_key not in baseline_by_seed else row["accuracy"] - baseline_by_seed[base_key]
        )
        row["gain_over_HFedATM"] = (
            None if row["accuracy"] is None or atm_key not in baseline_by_seed else row["accuracy"] - baseline_by_seed[atm_key]
        )
    return rows


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_csv(rows, path: Path):
    columns = REQUIRED_COLUMNS + ["status", "run_id", "split_scheme", "config_path", "log_path", "stdout_log_path"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt(row.get(col)) for col in columns})


def write_markdown(rows, path: Path):
    columns = REQUIRED_COLUMNS + ["status", "run_id"]
    lines = ["# PACS HDS-Main Results", ""]
    completed = sum(1 for row in rows if row.get("accuracy") is not None)
    lines.append(f"- Completed runs with parsed accuracy: {completed}/{len(rows)}")
    lines.append("- Accuracy is the final target-domain test accuracy reported by the run log.")
    lines.append("- No target-domain validation or tuning is performed by this parser.")
    lines.append("")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows, path: Path):
    aggregate = {}
    for row in rows:
        if row.get("avg_accuracy") is None:
            continue
        key = (row["target_domain"], row["client_method"], row["server_method"])
        aggregate[key] = (row["avg_accuracy"], row["std"])
    lines = [
        "% Auto-generated by scripts/collect_pacs_hds_main_results.py",
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Target & Client & Server & Avg. Acc. & Std. \\\\",
        "\\midrule",
    ]
    for (target, client, server), (avg, std) in sorted(aggregate.items()):
        lines.append(f"{target} & {client} & {server} & {avg:.4f} & {std:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_dir", default="configs/pacs_hds_main")
    parser.add_argument("--log_dir", default="outputs/pacs_hds_main/logs")
    parser.add_argument("--output_dir", default="outputs/pacs_hds_main")
    parser.add_argument("--wandb_dir", default="wandb")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    wandb_dir = Path(args.wandb_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_configs(config_dir)
    for row in rows:
        log_path = log_dir / f"{row['run_id']}.log"
        accuracy, status = parse_last_accuracy(log_path)
        row["stdout_log_path"] = str(log_path)
        row["accuracy"] = accuracy
        row["status"] = status
        row["log_path"] = str(log_path)
        if accuracy is None and status != "failed_traceback":
            wandb_log_path = find_wandb_output_log(row, wandb_dir)
            if wandb_log_path is not None:
                wandb_accuracy, wandb_status = parse_last_accuracy(wandb_log_path)
                if wandb_accuracy is not None or status in {"missing_log", "missing_accuracy"}:
                    row["accuracy"] = wandb_accuracy
                    row["status"] = f"{wandb_status}_wandb"
                    row["log_path"] = str(wandb_log_path)

    rows = summarize(rows)
    write_csv(rows, output_dir / "main_results.csv")
    write_markdown(rows, output_dir / "main_results.md")
    write_latex(rows, output_dir / "main_results_latex.tex")

    completed = sum(1 for row in rows if row.get("accuracy") is not None)
    print(f"Wrote {len(rows)} rows; parsed accuracy for {completed} runs.")
    print(output_dir / "main_results.csv")
    print(output_dir / "main_results.md")
    print(output_dir / "main_results_latex.tex")


if __name__ == "__main__":
    main()
