import csv
from argparse import Namespace

import numpy as np
import pytest

from src.hierarchy import assign_clients_to_stations
from src.partitioning import (
    build_hierarchical_partition,
    compute_client_domain_counts,
    partition_diagnostics,
)
from scripts.simulate_pacs_partitions import load_metadata_train_dataset, simulate_one


class FakeMetadataDataset:
    def __init__(self, domains):
        self.metadata_array = np.asarray([[d, i % 7, i] for i, d in enumerate(domains)], dtype=np.int64)
        self._metadata_array = self.metadata_array
        self._metadata_fields = ["domain", "y", "idx"]
        self.indices = np.arange(len(domains), dtype=np.int64)

    def __len__(self):
        return int(len(self.metadata_array))

    def __getitem__(self, idx):
        return None, int(self.metadata_array[idx, 1]), self.metadata_array[idx]


def fake_dataset(n_per_domain=80, n_domains=4):
    domains = []
    for domain_id in range(n_domains):
        domains.extend([domain_id] * n_per_domain)
    return FakeMetadataDataset(domains)


def base_hparam(tmp_path, method, seed=0, num_clients=20, num_stations=4):
    return {
        "id": "test_{}".format(method),
        "data_path": str(tmp_path),
        "partition_report_dir": str(tmp_path / method),
        "partition_save_report": True,
        "partition_plot": False,
        "partition_method": method,
        "num_clients": num_clients,
        "num_stations": num_stations,
        "seed": seed,
        "iid": 0.1,
        "station_assignment": "contiguous",
        "partition_alpha_station": 1.0,
        "partition_alpha_client": 1.0,
        "partition_quantity_mode": "none",
        "partition_station_size_sigma": 0.0,
        "partition_client_size_sigma": 0.0,
        "partition_station_size_alpha": 10.0,
        "partition_client_size_alpha": 10.0,
        "partition_min_client_samples": 1,
        "partition_resample_until_nonempty": True,
        "partition_max_resample_attempts": 100,
        "wandb": False,
    }


def assigned_indices(result):
    out = []
    for dataset in result.client_datasets:
        out.extend([int(idx) for idx in dataset.indices])
    return out


def test_paper_lambda_backward_compatible_counts(tmp_path):
    dataset = fake_dataset()
    hp = base_hparam(tmp_path, "paper_lambda")
    result = build_hierarchical_partition(dataset, ["domain"], None, hp)
    assert len(result.client_datasets) == hp["num_clients"]
    assert result.diagnostics["total_assigned_samples"] == len(dataset)
    assert result.diagnostics["missing_sample_count"] == 0
    assert result.diagnostics["duplicated_sample_count"] == 0


def test_missing_partition_method_defaults_to_paper_lambda(tmp_path):
    dataset = fake_dataset()
    hp = base_hparam(tmp_path, "paper_lambda")
    del hp["partition_method"]
    result = build_hierarchical_partition(dataset, ["domain"], None, hp)
    assert result.diagnostics["partition_method"] == "paper_lambda"
    assert len(result.client_datasets) == hp["num_clients"]
    assert result.diagnostics["missing_sample_count"] == 0
    assert result.diagnostics["duplicated_sample_count"] == 0


def test_hds_sample_conservation(tmp_path):
    dataset = fake_dataset()
    result = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "hds_dirichlet"))
    ids = assigned_indices(result)
    assert len(ids) == len(dataset)
    assert len(set(ids)) == len(dataset)
    assert set(ids) == set(range(len(dataset)))


def test_hds_no_empty_clients(tmp_path):
    dataset = fake_dataset(n_per_domain=100)
    for method in ["hds_inter", "hds_severe"]:
        result = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, method, seed=2))
    assert result.diagnostics["empty_client_count"] == 0
    assert result.diagnostics["allocation_subset"] == "train"
    assert result.diagnostics["all_source_domains_assigned"]
    assert result.diagnostics["train_domain_counts"] == result.diagnostics["assigned_domain_counts"]
    assert (tmp_path / "hds_inter" / "partition_diagnostics.json").exists()
    assert (tmp_path / "hds_inter" / "client_domain_counts.csv").exists()
    assert (tmp_path / "hds_inter" / "station_domain_counts.csv").exists()
    assert (tmp_path / "hds_inter" / "partition_summary.md").exists()


def test_hds_inter_has_higher_inter_js_than_mixed(tmp_path):
    dataset = fake_dataset(n_per_domain=100)
    inter = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "hds_inter", seed=1))
    intra = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "hds_intra", seed=1))
    assert inter.diagnostics["inter_station_js_mean"] > intra.diagnostics["inter_station_js_mean"]


def test_hds_intra_has_nontrivial_intra_js(tmp_path):
    dataset = fake_dataset(n_per_domain=100)
    result = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "hds_intra", seed=3))
    assert result.diagnostics["intra_station_js_mean"] > 0.01


def test_quantity_skew_changes_sample_cv(tmp_path):
    dataset = fake_dataset(n_per_domain=120)
    base = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "hds_dirichlet", seed=4))
    quantity = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "hds_quantity", seed=4))
    assert (
        quantity.diagnostics["station_sample_cv"] > base.diagnostics["station_sample_cv"]
        or quantity.diagnostics["client_sample_cv"] > base.diagnostics["client_sample_cv"]
    )


def test_station_assignment_modes_cover_all_clients_once(tmp_path):
    dataset = fake_dataset(n_per_domain=50)
    result = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "paper_lambda"))
    for mode in ["domain_clustered", "domain_mixed", "random_balanced"]:
        groups = assign_clients_to_stations(result.client_datasets, 4, assignment=mode, domain_field=["domain"], seed=0)
        flat = sorted(idx for group in groups for idx in group)
        assert flat == list(range(20))
        assert all(len(group) == 5 for group in groups)


def test_partition_diagnostics_keys(tmp_path):
    dataset = fake_dataset()
    result = build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, "hds_inter"))
    required = [
        "partition_method",
        "paper_name",
        "total_train_samples",
        "total_assigned_samples",
        "duplicated_sample_count",
        "missing_sample_count",
        "empty_client_count",
        "client_sample_cv",
        "station_sample_cv",
        "intra_station_js_mean",
        "inter_station_js_mean",
        "source_domain_coverage_per_station",
        "valid_hard_filters_passed",
    ]
    for key in required:
        assert key in result.diagnostics


def test_simulate_pacs_partitions_importable(tmp_path):
    csv_path = tmp_path / "pac-s.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "domain_remapped", "y"])
        writer.writeheader()
        for idx in range(80):
            writer.writerow({"split": "train", "domain_remapped": idx % 4, "y": idx % 7})
    dataset = load_metadata_train_dataset(csv_path)
    assert len(dataset) == 80
    args = Namespace(
        output_dir=str(tmp_path / "out"),
        num_clients=8,
        num_stations=4,
        plot=False,
        alpha_station_values=[1.0],
        alpha_client_values=[1.0],
    )
    diag = simulate_one(csv_path, "pac-s", "hds_inter", 0, 1.0, args)
    assert diag["missing_sample_count"] == 0
    assert diag["duplicated_sample_count"] == 0


def test_label_skew_methods_are_future_or_notimplemented(tmp_path):
    dataset = fake_dataset()
    for method in ["hds_label", "hds_full"]:
        with pytest.raises(NotImplementedError):
            build_hierarchical_partition(dataset, ["domain"], None, base_hparam(tmp_path, method))
