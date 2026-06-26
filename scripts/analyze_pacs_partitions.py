#!/usr/bin/env python
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_diagnostics(input_dir):
    rows = []
    for path in Path(input_dir).rglob("partition_diagnostics.json"):
        with path.open(encoding="utf-8") as fh:
            row = json.load(fh)
        row["report_path"] = str(path.parent)
        parts = path.parts
        if len(parts) >= 4:
            row.setdefault("split_scheme", parts[-4])
        rows.append(row)
    return rows


def hard_pass(row):
    return (
        row.get("missing_sample_count", 1) == 0
        and row.get("duplicated_sample_count", 1) == 0
        and row.get("empty_client_count", 1) == 0
        and row.get("min_client_samples", 0) >= 1
        and row.get("every_client_once", False)
        and row.get("every_station_has_client", False)
        and row.get("all_source_domains_assigned", True)
        and row.get("allocation_subset", "train") == "train"
        and row.get("valid_hard_filters_passed", False)
    )


def score_main(row):
    inter = row.get("inter_station_js_mean", 0.0)
    intra = row.get("intra_station_js_mean", 0.0)
    station_cv = row.get("station_sample_cv", 999.0)
    client_cv = row.get("client_sample_cv", 999.0)
    score = 0.0
    score -= abs(inter - 0.30)
    score -= 0.5 * abs(intra - 0.20)
    score -= max(station_cv - 1.0, 0.0)
    score -= 0.5 * max(client_cv - 1.5, 0.0)
    if row.get("partition_method") == "hds_inter":
        score += 0.25
    return score


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


HDS_ETA_LAMBDA_COLUMNS = [
    "split_scheme",
    "partition_method",
    "partition_eta",
    "partition_lambda",
    "seed",
    "missing_sample_count",
    "duplicated_sample_count",
    "empty_client_count",
    "min_client_samples",
    "median_client_samples",
    "max_client_samples",
    "min_station_samples",
    "median_station_samples",
    "max_station_samples",
    "client_sample_cv",
    "station_sample_cv",
    "inter_station_js_mean",
    "inter_station_js_std",
    "intra_station_js_mean",
    "intra_station_js_std",
    "client_domain_entropy_mean",
    "client_domain_entropy_std",
    "station_domain_entropy_mean",
    "station_domain_entropy_std",
    "allocation_subset",
    "target_test_val_used_for_partitioning",
    "every_client_once",
    "every_station_has_client",
    "all_source_domains_assigned",
    "valid_hard_filters_passed",
    "report_path",
]


def _mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def _std(values):
    values = [float(v) for v in values]
    if not values:
        return 0.0
    mu = _mean(values)
    return (sum((v - mu) ** 2 for v in values) / len(values)) ** 0.5


def hds_eta_lambda_rows(rows):
    out = []
    for row in rows:
        if row.get("partition_method") == "hds_eta_lambda":
            out.append(row)
    return out


def write_hds_eta_lambda_summary_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HDS_ETA_LAMBDA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in HDS_ETA_LAMBDA_COLUMNS})


def aggregate_eta_lambda(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(float(row.get("partition_eta", row.get("alpha_station", 1.0))), float(row.get("partition_lambda", row.get("alpha_client", 1.0))))].append(row)
    summary = []
    for (eta, lam), group in sorted(groups.items()):
        valid = [row for row in group if hard_pass(row)]
        summary.append({
            "partition_eta": eta,
            "partition_lambda": lam,
            "n": len(group),
            "valid_n": len(valid),
            "all_valid": len(valid) == len(group),
            "inter_station_js_mean": _mean(row.get("inter_station_js_mean", 0.0) for row in group),
            "inter_station_js_std": _std(row.get("inter_station_js_mean", 0.0) for row in group),
            "intra_station_js_mean": _mean(row.get("intra_station_js_mean", 0.0) for row in group),
            "intra_station_js_std": _std(row.get("intra_station_js_mean", 0.0) for row in group),
            "client_sample_cv": _mean(row.get("client_sample_cv", 0.0) for row in group),
            "station_sample_cv": _mean(row.get("station_sample_cv", 0.0) for row in group),
            "missing_sample_count": sum(int(row.get("missing_sample_count", 0)) for row in group),
            "duplicated_sample_count": sum(int(row.get("duplicated_sample_count", 0)) for row in group),
            "empty_client_count": sum(int(row.get("empty_client_count", 0)) for row in group),
        })
    return summary


def _score_eta_lambda(row):
    inter = row.get("inter_station_js_mean", 0.0)
    intra = row.get("intra_station_js_mean", 0.0)
    client_cv = row.get("client_sample_cv", 999.0)
    station_cv = row.get("station_sample_cv", 999.0)
    score = 0.0
    score -= abs(inter - 0.30)
    score -= 0.6 * abs(intra - 0.20)
    score -= max(station_cv - 1.0, 0.0)
    score -= 0.5 * max(client_cv - 1.5, 0.0)
    if 0.15 <= inter <= 0.50:
        score += 0.2
    if 0.05 <= intra <= 0.40:
        score += 0.2
    return score


def _control_table(summary, field_name, metric_name):
    grouped = defaultdict(list)
    for row in summary:
        grouped[float(row[field_name])].append(float(row[metric_name]))
    return [(key, _mean(vals)) for key, vals in sorted(grouped.items(), reverse=True)]


def _lower_value_increases_metric(table):
    if len(table) < 2:
        return False
    # Table is sorted high -> low by eta/lambda. We expect the last value
    # (lowest parameter) to have larger heterogeneity than the first.
    return table[-1][1] > table[0][1]


def write_hds_eta_lambda_summary_md(path, rows):
    summary = aggregate_eta_lambda(rows)
    lines = ["# HDS Eta-Lambda PACS Summary", ""]
    lines.append("| eta | lambda | n | valid | inter JS mean | intra JS mean | client CV | station CV | missing | duplicated | empty clients |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            "| {eta:g} | {lam:g} | {n} | {valid}/{n} | {inter:.4f} | {intra:.4f} | {ccv:.4f} | {scv:.4f} | {missing} | {dup} | {empty} |".format(
                eta=row["partition_eta"],
                lam=row["partition_lambda"],
                n=row["n"],
                valid=row["valid_n"],
                inter=row["inter_station_js_mean"],
                intra=row["intra_station_js_mean"],
                ccv=row["client_sample_cv"],
                scv=row["station_sample_cv"],
                missing=row["missing_sample_count"],
                dup=row["duplicated_sample_count"],
                empty=row["empty_client_count"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_eta_lambda(path, rows):
    summary = aggregate_eta_lambda(rows)
    hard_failed = [row for row in rows if not hard_pass(row)]
    eligible = [
        row for row in summary
        if row["all_valid"]
        and row["missing_sample_count"] == 0
        and row["duplicated_sample_count"] == 0
        and row["empty_client_count"] == 0
    ]
    ranked = sorted(eligible, key=_score_eta_lambda, reverse=True)
    main = ranked[0] if ranked else None
    eta_table = _control_table(summary, "partition_eta", "inter_station_js_mean")
    lambda_table = _control_table(summary, "partition_lambda", "intra_station_js_mean")
    eta_ok = _lower_value_increases_metric(eta_table)
    lambda_ok = _lower_value_increases_metric(lambda_table)
    no_missing = all(int(row.get("missing_sample_count", 1)) == 0 for row in rows)
    no_duplicated = all(int(row.get("duplicated_sample_count", 1)) == 0 for row in rows)
    no_target = all(
        row.get("allocation_subset", "train") == "train"
        and not bool(row.get("target_test_val_used_for_partitioning", False))
        for row in rows
    )

    lines = ["# Recommended HDS Eta-Lambda Setting", ""]
    if main:
        lines.append("## Recommendation")
        lines.append("- Main eta/lambda: `eta={:g}`, `lambda={:g}`.".format(main["partition_eta"], main["partition_lambda"]))
        lines.append("- Mean inter-station JS: `{:.4f}`.".format(main["inter_station_js_mean"]))
        lines.append("- Mean intra-station JS: `{:.4f}`.".format(main["intra_station_js_mean"]))
        lines.append("- Mean client/station sample CV: `{:.4f}` / `{:.4f}`.".format(main["client_sample_cv"], main["station_sample_cv"]))
    else:
        lines.append("## Recommendation")
        lines.append("- No eta/lambda pair passed hard filters across all evaluated splits/seeds.")
    lines.append("")
    lines.append("## Control Checks")
    lines.append("- Eta controls inter-station JS: `{}`.".format(eta_ok))
    lines.append("- Lambda controls intra-station JS: `{}`.".format(lambda_ok))
    lines.append("- No samples missing: `{}`.".format(no_missing))
    lines.append("- No samples duplicated: `{}`.".format(no_duplicated))
    lines.append("- No target/test/val data used for partitioning: `{}`.".format(no_target))
    lines.append("")
    lines.append("### Mean inter-station JS by eta")
    for eta, value in eta_table:
        lines.append("- eta `{:g}`: `{:.4f}`".format(eta, value))
    lines.append("")
    lines.append("### Mean intra-station JS by lambda")
    for lam, value in lambda_table:
        lines.append("- lambda `{:g}`: `{:.4f}`".format(lam, value))
    lines.append("")
    lines.append("## Hard Filter Failures")
    if hard_failed:
        lines.append("- `{}` diagnostics failed hard filters.".format(len(hard_failed)))
        for row in hard_failed[:25]:
            lines.append(
                "- split `{}`, seed `{}`, eta `{}`, lambda `{}`: missing `{}`, duplicated `{}`, empty `{}`.".format(
                    row.get("split_scheme"),
                    row.get("seed"),
                    row.get("partition_eta"),
                    row.get("partition_lambda"),
                    row.get("missing_sample_count"),
                    row.get("duplicated_sample_count"),
                    row.get("empty_client_count"),
                )
            )
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("Analyzer uses only train-allocation diagnostics generated before model training; it does not read target/test/validation labels for scoring.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path, rows):
    lines = ["# PACS Partition Grid Summary", ""]
    lines.append("| method | split | seed | iid | valid | inter JS | intra JS | client CV | station CV |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {method} | {split} | {seed} | {iid} | {valid} | {inter:.4f} | {intra:.4f} | {ccv:.4f} | {scv:.4f} |".format(
                method=row.get("partition_method"),
                split=row.get("split_scheme", ""),
                seed=row.get("seed"),
                iid=row.get("iid"),
                valid=hard_pass(row),
                inter=row.get("inter_station_js_mean", 0.0),
                intra=row.get("intra_station_js_mean", 0.0),
                ccv=row.get("client_sample_cv", 0.0),
                scv=row.get("station_sample_cv", 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendations(path, rows):
    valid = [row for row in rows if hard_pass(row)]
    ranked = sorted(valid, key=score_main, reverse=True)
    main = next((row for row in ranked if row.get("partition_method") == "hds_inter"), ranked[0] if ranked else None)
    continuity = [row for row in valid if row.get("partition_method") == "paper_lambda"]
    control = [row for row in valid if row.get("partition_method") == "paper_lambda_mixed"]
    intra = [row for row in valid if row.get("partition_method") == "hds_intra"]
    stress = [row for row in valid if row.get("partition_method") in {"hds_severe", "hds_quantity"}]
    lines = ["# Recommended PACS Partitions", ""]
    if main:
        lines.append("## Main journal split")
        lines.append("- `{}` from `{}`; inter JS `{:.4f}`, intra JS `{:.4f}`.".format(
            main.get("partition_method"),
            main.get("report_path"),
            main.get("inter_station_js_mean", 0.0),
            main.get("intra_station_js_mean", 0.0),
        ))
    lines.append("")
    lines.append("## Categories")
    lines.append("- Continuity: `{}` valid Original-lambda reports.".format(len(continuity)))
    lines.append("- Low inter-station control: `{}` Original-lambda-M reports.".format(len(control)))
    lines.append("- Intra-station ablation: `{}` HDS-Intra reports.".format(len(intra)))
    lines.append("- Stress: `{}` HDS-Severe/HDS-Q reports.".format(len(stress)))
    lines.append("- Invalid hard-filter reports: `{}`.".format(len(rows) - len(valid)))
    lines.append("")
    lines.append("Analyzer scoring uses partition diagnostics only; it does not read validation, test, or target-domain samples.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze PACS partition diagnostics.")
    parser.add_argument("input_dir", nargs="?")
    parser.add_argument("--input_dir", dest="input_dir_flag")
    parser.add_argument("--output_dir", default="outputs/pacs_partition_selection")
    args = parser.parse_args(argv)
    input_dir = args.input_dir_flag or args.input_dir
    if input_dir is None:
        parser.error("Provide input_dir positionally or with --input_dir.")
    rows = load_diagnostics(input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "partition_grid_summary.csv", rows)
    write_markdown(output_dir / "partition_grid_summary.md", rows)
    write_recommendations(output_dir / "recommended_partitions.md", rows)
    hds_rows = hds_eta_lambda_rows(rows)
    if hds_rows:
        write_hds_eta_lambda_summary_csv(output_dir / "hds_eta_lambda_summary.csv", hds_rows)
        write_hds_eta_lambda_summary_md(output_dir / "hds_eta_lambda_summary.md", hds_rows)
        write_recommended_eta_lambda(output_dir / "recommended_eta_lambda.md", hds_rows)
    print("Analyzed {} diagnostics into {}".format(len(rows), output_dir))


if __name__ == "__main__":
    main()
