import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import datasets as my_datasets
from src.hierarchy import assign_clients_to_stations, summarize_client_domains
from src.splitter import NonIIDSplitter


def load_domain_names(dataset, split_scheme):
    metadata_filename = "metadata.csv" if split_scheme == "official" else "{}.csv".format(split_scheme)
    metadata_path = Path(dataset.data_dir) / metadata_filename
    mapping = {}
    with metadata_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["domain_remapped"])] = row["domain"]
    return [mapping[idx] for idx in sorted(mapping)]


def proportions(counts):
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)


def write_counts_csv(path, counts, domain_names, prefix):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([prefix, "total"] + domain_names)
        for idx, row in enumerate(counts):
            writer.writerow([idx, int(row.sum())] + [int(v) for v in row.tolist()])


def plot_stacked(ax, counts, domain_names, title, xlabel, width=0.9):
    props = proportions(counts)
    x = np.arange(counts.shape[0])
    bottom = np.zeros(counts.shape[0])
    colors = plt.get_cmap("tab10").colors
    for domain_id, domain_name in enumerate(domain_names):
        ax.bar(
            x,
            props[:, domain_id],
            bottom=bottom,
            color=colors[domain_id % len(colors)],
            width=width,
            label=domain_name,
            linewidth=0,
        )
        bottom += props[:, domain_id]
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Domain proportion")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset.lower() != "pacs":
        raise NotImplementedError("This visualization currently supports PACS.")

    dataset = my_datasets.PACS(
        version="1.0",
        root_dir=args.dataset_path,
        download=args.download,
        split_scheme=args.split_scheme,
    )
    train_subset = dataset.get_subset("train")
    domain_names = load_domain_names(dataset, args.split_scheme)
    domain_field = ["domain"]
    minlength = len(domain_names)

    fig, axes = plt.subplots(
        len(args.lambdas),
        2,
        figsize=(18, 4.2 * len(args.lambdas)),
        constrained_layout=True,
    )
    if len(args.lambdas) == 1:
        axes = np.array([axes])

    summary = {
        "dataset": args.dataset,
        "split_scheme": args.split_scheme,
        "num_clients": args.num_clients,
        "num_stations": args.num_stations,
        "station_assignment": args.station_assignment,
        "domain_names": domain_names,
        "lambdas": {},
    }

    for row_idx, lambda_value in enumerate(args.lambdas):
        client_datasets = NonIIDSplitter(
            num_shards=args.num_clients,
            iid=lambda_value,
            seed=args.seed,
        ).split(train_subset, domain_field, transform=None)
        client_counts = summarize_client_domains(
            client_datasets,
            domain_field,
            minlength=minlength,
        )
        station_groups = assign_clients_to_stations(
            client_datasets,
            num_stations=args.num_stations,
            assignment=args.station_assignment,
            domain_field=domain_field,
            seed=args.seed,
        )
        station_counts = np.stack([
            client_counts[group].sum(axis=0)
            for group in station_groups
        ])

        stem = "pacs_{}_lambda_{}".format(
            args.split_scheme,
            str(lambda_value).replace(".", "p"),
        )
        write_counts_csv(out_dir / "{}_clients.csv".format(stem), client_counts, domain_names, "client_id")
        write_counts_csv(out_dir / "{}_stations.csv".format(stem), station_counts, domain_names, "station_id")

        summary["lambdas"][str(lambda_value)] = {
            "station_groups": station_groups,
            "client_totals": client_counts.sum(axis=1).astype(int).tolist(),
            "station_counts": station_counts.astype(int).tolist(),
        }

        plot_stacked(
            axes[row_idx, 0],
            client_counts,
            domain_names,
            "Client domains, lambda={}".format(lambda_value),
            "Client id",
            width=1.0,
        )
        clients_per_station = args.num_clients / args.num_stations
        for boundary in range(1, args.num_stations):
            axes[row_idx, 0].axvline(
                boundary * clients_per_station - 0.5,
                color="#222222",
                linewidth=0.7,
                alpha=0.45,
            )
        axes[row_idx, 0].set_xlim(-0.5, args.num_clients - 0.5)
        axes[row_idx, 0].set_xticks(np.linspace(0, args.num_clients - 1, args.num_stations + 1, dtype=int))

        plot_stacked(
            axes[row_idx, 1],
            station_counts,
            domain_names,
            "Station domains, lambda={}".format(lambda_value),
            "Station id",
        )
        axes[row_idx, 1].set_xticks(np.arange(args.num_stations))
        axes[row_idx, 1].set_xticklabels([str(i) for i in range(args.num_stations)])

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(domain_names), 4))
    fig.suptitle(
        "PACS hierarchy domain split: {} clients, {} stations, assignment={}".format(
            args.num_clients,
            args.num_stations,
            args.station_assignment,
        ),
        fontsize=15,
    )
    fig.subplots_adjust(bottom=0.08)

    image_path = out_dir / "pacs_hierarchy_{}_{}.png".format(
        args.split_scheme,
        args.station_assignment,
    )
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "pacs_hierarchy_{}_{}_summary.json".format(
        args.split_scheme,
        args.station_assignment,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved figure:", image_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize client/station domain splits for PACS.")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--dataset_path", default="data")
    parser.add_argument("--split_scheme", default="official")
    parser.add_argument("--output_dir", default="outputs/hierarchy_visualization")
    parser.add_argument("--num_clients", default=100, type=int)
    parser.add_argument("--num_stations", default=10, type=int)
    parser.add_argument("--station_assignment", default="contiguous")
    parser.add_argument("--lambdas", nargs="+", default=[1.0, 0.1, 0.0], type=float)
    parser.add_argument("--seed", default=1001, type=int)
    parser.add_argument("--download", action="store_true")
    main(parser.parse_args())
