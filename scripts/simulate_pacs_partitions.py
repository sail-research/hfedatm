#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.partitioning import build_hierarchical_partition


DEFAULT_METHODS = [
    "paper_lambda",
    "paper_lambda_clustered",
    "paper_lambda_mixed",
    "hds_eta_lambda",
    "hds_dirichlet",
    "hds_inter",
    "hds_intra",
    "hds_severe",
    "hds_quantity",
    "hds_partial",
]
DEFAULT_SPLITS = ["pac-s", "pcs-a", "pas-c", "acs-p"]
LODO_SPLIT_RE = re.compile(r"^[acps]{3}-[acps]$")


class MetadataOnlyDataset:
    def __init__(self, metadata_array, metadata_fields):
        self.metadata_array = np.asarray(metadata_array, dtype=np.int64)
        self._metadata_array = self.metadata_array
        self._metadata_fields = list(metadata_fields)
        self.indices = np.arange(len(self.metadata_array), dtype=np.int64)

    def __len__(self):
        return int(len(self.metadata_array))

    def __getitem__(self, idx):
        return None, int(self.metadata_array[idx, 1]), self.metadata_array[idx]


def find_pacs_metadata_csvs(search_roots=None):
    search_roots = search_roots or ["resources", "data", "dataset", "datasets"]
    found = {}
    for root in search_roots:
        root_path = ROOT / root
        if not root_path.exists():
            continue
        for csv_path in root_path.rglob("*.csv"):
            if "pacs" not in str(csv_path).lower():
                continue
            found[csv_path.stem] = csv_path
    return found


def available_lodo_splits(csvs):
    return sorted(name for name in csvs if LODO_SPLIT_RE.match(name))


def load_metadata_train_dataset(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if row.get("split") != "train":
                continue
            domain = int(row.get("domain_remapped", row.get("domain", 0)))
            label = int(row.get("y", 0))
            rows.append([domain, label, idx])
    if not rows:
        raise ValueError("No train rows found in {}".format(csv_path))
    return MetadataOnlyDataset(rows, ["domain", "y", "idx"])


def _safe_float_label(value):
    return str(value).replace(".", "p").replace("-", "m")


def hparam_for(method, split_scheme, seed, iid, eta, lam, args, report_dir):
    hparam = {
        "id": "{}_{}_seed{}_iid{}_eta{}_lambda{}".format(
            split_scheme,
            method,
            seed,
            _safe_float_label(iid),
            _safe_float_label(eta),
            _safe_float_label(lam),
        ),
        "data_path": str(report_dir),
        "partition_report_dir": str(report_dir),
        "partition_save_report": True,
        "partition_plot": args.plot,
        "partition_method": method,
        "partition_eta": eta,
        "partition_lambda": lam,
        "partition_alpha_station": eta,
        "partition_alpha_client": lam,
        "partition_quantity_mode": "none",
        "partition_min_client_samples": 1,
        "partition_resample_until_nonempty": True,
        "partition_max_resample_attempts": 100,
        "num_clients": args.num_clients,
        "num_stations": args.num_stations,
        "seed": seed,
        "iid": iid,
        "split_scheme": split_scheme,
        "station_assignment": "contiguous",
        "wandb": False,
    }
    return hparam


def _write_updated_diagnostics(report_dir, diagnostics):
    report_dir = Path(report_dir)
    clean = {key: value for key, value in diagnostics.items() if not key.startswith("_")}
    (report_dir / "partition_diagnostics.json").write_text(
        json.dumps(clean, indent=2),
        encoding="utf-8",
    )
    scalar = {
        key: value
        for key, value in clean.items()
        if not isinstance(value, (list, dict))
    }
    with (report_dir / "partition_diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(scalar.keys()))
        writer.writeheader()
        writer.writerow(scalar)


def simulate_one(csv_path, split_scheme, method, seed, iid, args, eta=None, lam=None):
    dataset = load_metadata_train_dataset(csv_path)
    eta_values = getattr(args, "eta_values", getattr(args, "alpha_station_values", [1.0]))
    lambda_values = getattr(args, "lambda_values", getattr(args, "alpha_client_values", [1.0]))
    eta = eta_values[0] if eta is None else eta
    lam = lambda_values[0] if lam is None else lam
    report_dir = (
        Path(args.output_dir)
        / split_scheme
        / method
        / "eta_{}".format(_safe_float_label(eta))
        / "lambda_{}".format(_safe_float_label(lam))
        / "seed_{}_iid_{}".format(seed, _safe_float_label(iid))
    )
    hparam = hparam_for(method, split_scheme, seed, iid, eta, lam, args, report_dir)
    result = build_hierarchical_partition(dataset, domain_field=["domain"], transform=None, hparam=hparam)
    result.diagnostics.update({
        "split_scheme": split_scheme,
        "partition_eta": float(eta),
        "partition_lambda": float(lam),
        "target_test_val_used_for_partitioning": False,
    })
    _write_updated_diagnostics(report_dir, result.diagnostics)
    return result.diagnostics


def eta_lambda_pairs(method, args):
    eta_values = getattr(args, "eta_values", getattr(args, "alpha_station_values", [1.0]))
    lambda_values = getattr(args, "lambda_values", getattr(args, "alpha_client_values", [1.0]))
    if method == "hds_eta_lambda":
        return [(eta, lam) for eta in eta_values for lam in lambda_values]
    if method == "hds_dirichlet" and args.sweep_dirichlet_eta_lambda:
        return [(eta, lam) for eta in eta_values for lam in lambda_values]
    return [(eta_values[0], lambda_values[0])]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulate PACS hierarchical partitions from metadata CSVs.")
    parser.add_argument("--output_dir", default="outputs/pacs_partition_sim")
    parser.add_argument("--num_clients", default=100, type=int)
    parser.add_argument("--num_stations", default=10, type=int)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iid_values", nargs="+", type=float, default=[1.0, 0.1, 0.0])
    parser.add_argument("--eta_values", nargs="+", type=float, default=[1.0])
    parser.add_argument("--lambda_values", nargs="+", type=float, default=[1.0])
    parser.add_argument("--alpha_station_values", nargs="+", type=float)
    parser.add_argument("--alpha_client_values", nargs="+", type=float)
    parser.add_argument("--sweep_dirichlet_eta_lambda", action="store_true")
    parser.add_argument("--split_schemes", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.alpha_station_values is not None:
        args.eta_values = args.alpha_station_values
    if args.alpha_client_values is not None:
        args.lambda_values = args.alpha_client_values

    csvs = find_pacs_metadata_csvs()
    if args.split_schemes == ["all_lodo"]:
        args.split_schemes = available_lodo_splits(csvs)
    missing = [split for split in args.split_schemes if split not in csvs]
    if missing:
        raise FileNotFoundError(
            "Missing PACS metadata CSVs for {}. Found split schemes: {}".format(
                missing, sorted(csvs.keys())
            )
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for split in args.split_schemes:
        for method in args.methods:
            iid_values = args.iid_values if method.startswith("paper_lambda") else [1.0]
            for seed in args.seeds:
                for iid in iid_values:
                    for eta, lam in eta_lambda_pairs(method, args):
                        diag = simulate_one(csvs[split], split, method, seed, iid, args, eta=eta, lam=lam)
                        print(
                            "{split} {method} seed={seed} iid={iid} eta={eta} lambda={lam}: inter_js={inter:.4f} intra_js={intra:.4f} valid={valid}".format(
                                split=split,
                                method=method,
                                seed=seed,
                                iid=iid,
                                eta=eta,
                                lam=lam,
                                inter=diag["inter_station_js_mean"],
                                intra=diag["intra_station_js_mean"],
                                valid=diag["valid_hard_filters_passed"],
                            )
                        )


if __name__ == "__main__":
    main()
