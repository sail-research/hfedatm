import csv
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from .hierarchy import assign_clients_to_stations
from .splitter import NonIIDSplitter

try:
    from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset
except Exception:  # pragma: no cover - tests do not require WILDS.
    WILDSDataset = ()
    WILDSSubset = ()


PAPER_NAMES = {
    "paper_lambda": "Original-lambda",
    "paper_lambda_clustered": "Original-lambda-C",
    "paper_lambda_mixed": "Original-lambda-M",
    "hds_eta_lambda": "HDS-eta-lambda",
    "hds_dirichlet": "HDS",
    "hds_inter": "HDS-Inter",
    "hds_intra": "HDS-Intra",
    "hds_severe": "HDS-Severe",
    "hds_quantity": "HDS-Q",
    "hds_partial": "HDS-PP",
    "hds_label": "HDS-Label",
    "hds_full": "HDS-Full",
}


@dataclass
class PartitionConfig:
    method: str
    num_clients: int
    num_stations: int
    seed: int
    iid: float
    station_assignment: str
    domain_field: Any
    alpha_station: float = 1.0
    alpha_client: float = 1.0
    quantity_mode: str = "none"
    station_size_sigma: float = 0.0
    client_size_sigma: float = 0.0
    station_size_alpha: float = 10.0
    client_size_alpha: float = 10.0
    min_client_samples: int = 1
    resample_until_nonempty: bool = True
    max_resample_attempts: int = 100
    report_dir: Optional[str] = None
    save_report: bool = True
    plot: bool = False


@dataclass
class PartitionResult:
    client_datasets: List[Any]
    station_client_indices: List[List[int]]
    diagnostics: Dict[str, Any]


class MetadataSubset:
    """Small subset wrapper for metadata-only tests and simulator scripts."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self._metadata_fields = list(getattr(dataset, "_metadata_fields", getattr(dataset, "metadata_fields", [])))
        metadata = get_metadata_array(dataset)
        self.metadata_array = metadata[self.indices]
        self._metadata_array = self.metadata_array

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, idx):
        item = self.dataset[int(self.indices[idx])]
        if self.transform is not None and isinstance(item, tuple) and len(item) >= 2:
            return (self.transform(item[0]),) + item[1:]
        return item


def _to_numpy(values):
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    if hasattr(values, "cpu"):
        return values.cpu().numpy()
    return np.asarray(values)


def get_metadata_array(dataset):
    if hasattr(dataset, "metadata_array"):
        return _to_numpy(dataset.metadata_array)
    if hasattr(dataset, "_metadata_array"):
        return _to_numpy(dataset._metadata_array)
    if isinstance(dataset, Subset):
        parent_meta = get_metadata_array(dataset.dataset)
        return parent_meta[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent_meta = get_metadata_array(dataset.dataset)
        return parent_meta[np.asarray(dataset.indices, dtype=np.int64)]
    raise ValueError("Dataset does not expose metadata_array/_metadata_array.")


def _metadata_fields(dataset):
    if hasattr(dataset, "_metadata_fields"):
        return list(dataset._metadata_fields)
    if hasattr(dataset, "metadata_fields"):
        return list(dataset.metadata_fields)
    if hasattr(dataset, "dataset"):
        return _metadata_fields(dataset.dataset)
    return []


def get_metadata_field_index(dataset, field):
    if isinstance(field, (list, tuple)):
        field = field[0]
    if isinstance(field, str):
        fields = _metadata_fields(dataset)
        if field not in fields:
            raise ValueError("Metadata field '{}' not found in {}.".format(field, fields))
        return int(fields.index(field))
    return int(field)


def get_subset_indices(dataset):
    if hasattr(dataset, "indices"):
        return np.asarray(dataset.indices, dtype=np.int64)
    return np.arange(len(dataset), dtype=np.int64)


def make_subset_like(dataset, indices, transform=None):
    indices = [int(idx) for idx in indices]
    if WILDSSubset and isinstance(dataset, WILDSSubset):
        return WILDSSubset(dataset.dataset, indices, transform=transform)
    if WILDSDataset and isinstance(dataset, WILDSDataset):
        return WILDSSubset(dataset, indices, transform=transform)
    if isinstance(dataset, Subset):
        return Subset(dataset.dataset, indices)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "__len__"):
        return MetadataSubset(dataset.dataset, indices, transform=transform)
    return MetadataSubset(dataset, indices, transform=transform)


def extract_domain_ids(dataset, domain_field):
    metadata = get_metadata_array(dataset)
    field_idx = get_metadata_field_index(dataset, domain_field)
    return metadata[:, field_idx].astype(np.int64)


def compute_client_domain_counts(client_datasets, domain_field, num_domains=None):
    if num_domains is None:
        max_domain = 0
        for dataset in client_datasets:
            domains = extract_domain_ids(dataset, domain_field)
            if domains.size:
                max_domain = max(max_domain, int(domains.max()))
        num_domains = max_domain + 1
    counts = []
    for dataset in client_datasets:
        domains = extract_domain_ids(dataset, domain_field)
        counts.append(np.bincount(domains, minlength=num_domains).astype(np.int64))
    return np.stack(counts, axis=0) if counts else np.zeros((0, num_domains), dtype=np.int64)


def compute_station_domain_counts(client_domain_counts, station_client_indices):
    station_counts = []
    for clients in station_client_indices:
        if clients:
            station_counts.append(client_domain_counts[np.asarray(clients, dtype=np.int64)].sum(axis=0))
        else:
            station_counts.append(np.zeros(client_domain_counts.shape[1], dtype=np.int64))
    return np.stack(station_counts, axis=0) if station_counts else np.zeros((0, client_domain_counts.shape[1]), dtype=np.int64)


def _safe_prob(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return np.ones_like(values, dtype=np.float64) / max(len(values), 1)
    out = values / total
    out = np.clip(out, eps, 1.0)
    return out / out.sum()


def js_divergence(p, q):
    p = _safe_prob(p)
    q = _safe_prob(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy(p):
    p = _safe_prob(p)
    return float(-np.sum(p * np.log(p)))


def gini(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if np.any(values < 0):
        values = values - values.min()
    total = values.sum()
    if total <= 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) / (n * total)) - ((n + 1) / n))


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean()) if values.size else 0.0
    if mean == 0:
        return 0.0
    return float(values.std() / mean)


def _pairwise_js(rows):
    if len(rows) < 2:
        return np.asarray([0.0], dtype=np.float64)
    vals = [js_divergence(rows[i], rows[j]) for i, j in combinations(range(len(rows)), 2)]
    return np.asarray(vals, dtype=np.float64)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        values = np.asarray([0.0], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _station_groups_balanced(num_clients, num_stations):
    return [list(map(int, arr.tolist())) for arr in np.array_split(np.arange(num_clients), num_stations)]


def _normal_sample_weights(rng, cfg: PartitionConfig):
    S = cfg.num_stations
    C = cfg.num_clients
    station_groups = _station_groups_balanced(C, S)
    station_weights = np.ones(S, dtype=np.float64)
    client_weights = np.ones(C, dtype=np.float64)

    if cfg.quantity_mode == "none":
        return client_weights
    if cfg.quantity_mode == "lognormal":
        station_weights = rng.lognormal(mean=0.0, sigma=max(cfg.station_size_sigma, 0.0), size=S)
        for station_id, clients in enumerate(station_groups):
            local = rng.lognormal(mean=0.0, sigma=max(cfg.client_size_sigma, 0.0), size=len(clients))
            for client_id, weight in zip(clients, local):
                client_weights[client_id] = station_weights[station_id] * weight
        return client_weights
    if cfg.quantity_mode == "dirichlet":
        station_weights = rng.dirichlet(np.full(S, max(cfg.station_size_alpha, 1e-3)))
        for station_id, clients in enumerate(station_groups):
            local = rng.dirichlet(np.full(len(clients), max(cfg.client_size_alpha, 1e-3)))
            for client_id, weight in zip(clients, local):
                client_weights[client_id] = station_weights[station_id] * weight
        return client_weights
    raise ValueError("Unknown quantity_mode '{}'.".format(cfg.quantity_mode))


def _partition_config_from_hparam(hparam, domain_field):
    method = hparam.get("partition_method", "paper_lambda")
    eta = hparam.get("partition_eta", hparam.get("partition_alpha_station", 1.0))
    lam = hparam.get("partition_lambda", hparam.get("partition_alpha_client", 1.0))
    cfg = PartitionConfig(
        method=method,
        num_clients=int(hparam["num_clients"]),
        num_stations=int(hparam.get("num_stations", 1)),
        seed=int(hparam.get("seed", 0)),
        iid=float(hparam.get("iid", 1.0)),
        station_assignment=hparam.get("station_assignment", "contiguous"),
        domain_field=domain_field,
        alpha_station=float(eta),
        alpha_client=float(lam),
        quantity_mode=hparam.get("partition_quantity_mode", "none"),
        station_size_sigma=float(hparam.get("partition_station_size_sigma", 0.0)),
        client_size_sigma=float(hparam.get("partition_client_size_sigma", 0.0)),
        station_size_alpha=float(hparam.get("partition_station_size_alpha", 10.0)),
        client_size_alpha=float(hparam.get("partition_client_size_alpha", 10.0)),
        min_client_samples=int(hparam.get("partition_min_client_samples", 1)),
        resample_until_nonempty=_as_bool(hparam.get("partition_resample_until_nonempty", True)),
        max_resample_attempts=int(hparam.get("partition_max_resample_attempts", 100)),
        report_dir=hparam.get("partition_report_dir"),
        save_report=_as_bool(hparam.get("partition_save_report", True)),
        plot=_as_bool(hparam.get("partition_plot", False)),
    )

    if method == "hds_inter" or method == "hds_partial":
        cfg.alpha_station = 0.2
        cfg.alpha_client = 10.0
        cfg.quantity_mode = "none"
    elif method == "hds_intra":
        cfg.alpha_station = 10.0
        cfg.alpha_client = 0.2
        cfg.quantity_mode = "none"
    elif method == "hds_severe":
        cfg.alpha_station = 0.2
        cfg.alpha_client = 0.2
        cfg.quantity_mode = "none"
    elif method == "hds_quantity":
        cfg.alpha_station = 0.3
        cfg.alpha_client = 0.5
        cfg.quantity_mode = "lognormal"
        cfg.station_size_sigma = 0.8
        cfg.client_size_sigma = 0.8
    return cfg


def _legacy_lambda_partition_generic(train_subset, domain_field, transform, cfg: PartitionConfig):
    rng = np.random.default_rng(cfg.seed)
    domains = extract_domain_ids(train_subset, domain_field)
    train_indices = get_subset_indices(train_subset)
    unique_domains = np.unique(domains).astype(np.int64)
    num_domains = int(unique_domains.max()) + 1 if unique_domains.size else 0
    counts_per_domain = np.bincount(domains, minlength=num_domains)
    C = cfg.num_clients

    non_empty = counts_per_domain > 0
    main_shards_per_domain = non_empty.astype(np.int64)
    while main_shards_per_domain.sum() < C:
        ratios = np.divide(
            counts_per_domain.astype(np.float64),
            np.maximum(main_shards_per_domain, 1),
            out=np.zeros_like(counts_per_domain, dtype=np.float64),
            where=main_shards_per_domain > 0,
        )
        main_shards_per_domain[int(np.argmax(ratios))] += 1
    main_domains = []
    for domain_id, count in enumerate(main_shards_per_domain):
        main_domains.extend([domain_id] * int(count))
    main_domains = main_domains[:C]

    expected = []
    main_ratio = np.divide(1.0, np.maximum(main_shards_per_domain, 1), out=np.zeros_like(counts_per_domain, dtype=np.float64), where=main_shards_per_domain > 0)
    non_main_ratio = 1.0 / C
    for main_domain in main_domains:
        onehot = np.zeros(num_domains, dtype=np.float64)
        onehot[main_domain] = 1.0
        expected.append(counts_per_domain * (main_ratio * onehot * (1.0 - cfg.iid) + non_main_ratio * cfg.iid))
    expected = np.asarray(expected)
    final_counts = np.floor(expected).astype(np.int64)
    for domain_id, diff in enumerate((counts_per_domain - final_counts.sum(axis=0)).astype(np.int64)):
        for row in range(int(diff)):
            final_counts[row % C, domain_id] += 1

    domain_indices = []
    for domain_id in range(num_domains):
        idx = train_indices[np.where(domains == domain_id)[0]]
        domain_indices.append(rng.permutation(idx))
    pointers = np.zeros(num_domains, dtype=np.int64)
    client_indices = []
    for client_id in range(C):
        current = []
        for domain_id in range(num_domains):
            take = int(final_counts[client_id, domain_id])
            if take > 0:
                current.extend(domain_indices[domain_id][pointers[domain_id]:pointers[domain_id] + take].tolist())
                pointers[domain_id] += take
        client_indices.append(current)
    return [make_subset_like(train_subset, idxs, transform=transform) for idxs in client_indices]


def _paper_lambda_clients(train_subset, domain_field, transform, cfg: PartitionConfig):
    if not (
        (WILDSSubset and isinstance(train_subset, WILDSSubset))
        or (WILDSDataset and isinstance(train_subset, WILDSDataset))
    ):
        return _legacy_lambda_partition_generic(train_subset, domain_field, transform, cfg)
    try:
        return NonIIDSplitter(num_shards=cfg.num_clients, iid=cfg.iid, seed=cfg.seed).split(
            train_subset, domain_field, transform=transform
        )
    except Exception as exc:
        warnings.warn("Falling back to generic paper_lambda splitter: {}".format(exc))
        return _legacy_lambda_partition_generic(train_subset, domain_field, transform, cfg)


def _repair_empty_clients(client_indices, train_indices, min_samples):
    repair_count = 0
    if min_samples <= 0:
        return repair_count
    for client_id, indices in enumerate(client_indices):
        while len(indices) < min_samples:
            donors = sorted(range(len(client_indices)), key=lambda idx: len(client_indices[idx]), reverse=True)
            donor = next((idx for idx in donors if len(client_indices[idx]) > min_samples and idx != client_id), None)
            if donor is None:
                break
            moved = client_indices[donor].pop()
            client_indices[client_id].append(moved)
            repair_count += 1
    return repair_count


def _hds_client_indices(train_subset, cfg: PartitionConfig):
    rng = np.random.default_rng(cfg.seed)
    domains = extract_domain_ids(train_subset, cfg.domain_field)
    train_indices = get_subset_indices(train_subset)
    num_domains = int(domains.max()) + 1 if domains.size else 0
    station_groups = _station_groups_balanced(cfg.num_clients, cfg.num_stations)
    client_station = np.zeros(cfg.num_clients, dtype=np.int64)
    for station_id, clients in enumerate(station_groups):
        for client_id in clients:
            client_station[client_id] = station_id

    eps = 1e-6
    station_pi = rng.dirichlet(np.full(num_domains, max(cfg.alpha_station, eps)), size=cfg.num_stations)
    client_pi = np.zeros((cfg.num_clients, num_domains), dtype=np.float64)
    for client_id in range(cfg.num_clients):
        base = np.clip(station_pi[client_station[client_id]], eps, 1.0)
        base = base / base.sum()
        client_pi[client_id] = rng.dirichlet(np.maximum(cfg.alpha_client * base, eps))

    quantity_weights = _normal_sample_weights(rng, cfg)
    client_indices = [[] for _ in range(cfg.num_clients)]
    for domain_id in range(num_domains):
        domain_positions = np.where(domains == domain_id)[0]
        shuffled = rng.permutation(train_indices[domain_positions])
        if shuffled.size == 0:
            continue
        probs = quantity_weights * np.maximum(client_pi[:, domain_id], eps)
        probs = probs / probs.sum()
        counts = rng.multinomial(int(shuffled.size), probs)
        pointer = 0
        for client_id, count in enumerate(counts.tolist()):
            if count > 0:
                client_indices[client_id].extend(shuffled[pointer:pointer + count].tolist())
                pointer += count
        assert pointer == shuffled.size

    repair_count = _repair_empty_clients(client_indices, train_indices, cfg.min_client_samples)
    for idxs in client_indices:
        rng.shuffle(idxs)
    return client_indices, station_groups, repair_count


def partition_diagnostics(
    cfg: PartitionConfig,
    train_subset,
    client_datasets,
    station_client_indices,
    repair_count=0,
):
    train_indices = get_subset_indices(train_subset)
    assigned_indices = []
    for dataset in client_datasets:
        assigned_indices.extend(get_subset_indices(dataset).tolist())
    assigned = np.asarray(assigned_indices, dtype=np.int64)
    unique_assigned = np.unique(assigned) if assigned.size else np.asarray([], dtype=np.int64)
    train_set = set(map(int, train_indices.tolist()))
    assigned_set = set(map(int, assigned.tolist()))

    domains = extract_domain_ids(train_subset, cfg.domain_field)
    num_domains = int(domains.max()) + 1 if domains.size else 0
    train_domain_counts = np.bincount(domains, minlength=num_domains).astype(np.int64)
    client_counts = compute_client_domain_counts(client_datasets, cfg.domain_field, num_domains=num_domains)
    station_counts = compute_station_domain_counts(client_counts, station_client_indices)
    assigned_domain_counts = client_counts.sum(axis=0).astype(np.int64)
    client_sizes = client_counts.sum(axis=1)
    station_sizes = station_counts.sum(axis=1)
    client_entropies = np.asarray([entropy(row) for row in client_counts])
    station_entropies = np.asarray([entropy(row) for row in station_counts])
    intra_vals = []
    for station_clients in station_client_indices:
        rows = client_counts[np.asarray(station_clients, dtype=np.int64)] if station_clients else np.zeros((0, num_domains))
        intra_vals.extend(_pairwise_js(rows).tolist())
    inter_vals = _pairwise_js(station_counts)

    flat_station_clients = [idx for group in station_client_indices for idx in group]
    every_client_once = sorted(flat_station_clients) == list(range(len(client_datasets)))
    source_domain_coverage = (station_counts > 0).sum(axis=1).astype(int).tolist()
    all_source_domains_assigned = bool(np.array_equal(train_domain_counts, assigned_domain_counts))
    duplicated = int(assigned.size - unique_assigned.size)
    missing = int(len(train_set - assigned_set))
    empty_clients = int(np.sum(client_sizes == 0))
    diag = {
        "partition_method": cfg.method,
        "paper_name": PAPER_NAMES.get(cfg.method, cfg.method),
        "allocation_subset": "train",
        "seed": int(cfg.seed),
        "num_clients": int(cfg.num_clients),
        "num_stations": int(cfg.num_stations),
        "station_assignment": cfg.station_assignment,
        "iid": float(cfg.iid),
        "partition_eta": float(cfg.alpha_station),
        "partition_lambda": float(cfg.alpha_client),
        "alpha_station": float(cfg.alpha_station),
        "alpha_client": float(cfg.alpha_client),
        "quantity_mode": cfg.quantity_mode,
        "total_train_samples": int(len(train_indices)),
        "total_assigned_samples": int(assigned.size),
        "duplicated_sample_count": duplicated,
        "missing_sample_count": missing,
        "empty_client_count": empty_clients,
        "min_client_samples": int(client_sizes.min()) if client_sizes.size else 0,
        "median_client_samples": float(np.median(client_sizes)) if client_sizes.size else 0.0,
        "max_client_samples": int(client_sizes.max()) if client_sizes.size else 0,
        "min_station_samples": int(station_sizes.min()) if station_sizes.size else 0,
        "median_station_samples": float(np.median(station_sizes)) if station_sizes.size else 0.0,
        "max_station_samples": int(station_sizes.max()) if station_sizes.size else 0,
        "client_sample_cv": coefficient_of_variation(client_sizes),
        "station_sample_cv": coefficient_of_variation(station_sizes),
        "client_sample_gini": gini(client_sizes),
        "station_sample_gini": gini(station_sizes),
        "client_domain_entropy_mean": float(client_entropies.mean()) if client_entropies.size else 0.0,
        "client_domain_entropy_std": float(client_entropies.std()) if client_entropies.size else 0.0,
        "client_domain_entropy_min": float(client_entropies.min()) if client_entropies.size else 0.0,
        "client_domain_entropy_max": float(client_entropies.max()) if client_entropies.size else 0.0,
        "station_domain_entropy_mean": float(station_entropies.mean()) if station_entropies.size else 0.0,
        "station_domain_entropy_std": float(station_entropies.std()) if station_entropies.size else 0.0,
        "station_domain_entropy_min": float(station_entropies.min()) if station_entropies.size else 0.0,
        "station_domain_entropy_max": float(station_entropies.max()) if station_entropies.size else 0.0,
        "intra_station_js_mean": float(np.mean(intra_vals)) if intra_vals else 0.0,
        "intra_station_js_std": float(np.std(intra_vals)) if intra_vals else 0.0,
        "intra_station_js_min": float(np.min(intra_vals)) if intra_vals else 0.0,
        "intra_station_js_max": float(np.max(intra_vals)) if intra_vals else 0.0,
        "inter_station_js_mean": float(inter_vals.mean()) if inter_vals.size else 0.0,
        "inter_station_js_std": float(inter_vals.std()) if inter_vals.size else 0.0,
        "inter_station_js_min": float(inter_vals.min()) if inter_vals.size else 0.0,
        "inter_station_js_max": float(inter_vals.max()) if inter_vals.size else 0.0,
        "source_domain_coverage_per_station": source_domain_coverage,
        "train_domain_counts": train_domain_counts.tolist(),
        "assigned_domain_counts": assigned_domain_counts.tolist(),
        "all_source_domains_assigned": all_source_domains_assigned,
        "num_single_domain_clients": int(np.sum((client_counts > 0).sum(axis=1) <= 1)),
        "num_single_domain_stations": int(np.sum((station_counts > 0).sum(axis=1) <= 1)),
        "repair_count": int(repair_count),
        "every_client_once": bool(every_client_once),
        "every_station_has_client": bool(all(len(group) > 0 for group in station_client_indices)),
    }
    diag["valid_hard_filters_passed"] = bool(
        diag["missing_sample_count"] == 0
        and diag["duplicated_sample_count"] == 0
        and diag["empty_client_count"] == 0
        and diag["min_client_samples"] >= cfg.min_client_samples
        and diag["every_client_once"]
        and diag["every_station_has_client"]
        and diag["all_source_domains_assigned"]
    )
    diag["_client_domain_counts"] = client_counts.tolist()
    diag["_station_domain_counts"] = station_counts.tolist()
    return diag


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def save_partition_report_json(report_dir, diagnostics):
    path = Path(report_dir) / "partition_diagnostics.json"
    clean = {k: v for k, v in diagnostics.items() if not k.startswith("_")}
    path.write_text(json.dumps(clean, indent=2, default=_json_safe), encoding="utf-8")
    return path


def save_partition_metrics_csv(report_dir, diagnostics):
    path = Path(report_dir) / "partition_diagnostics.csv"
    clean = {
        key: value
        for key, value in diagnostics.items()
        if not key.startswith("_") and not isinstance(value, (list, dict))
    }
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(clean.keys()))
        writer.writeheader()
        writer.writerow(clean)
    return path


def save_partition_counts_csv(report_dir, diagnostics):
    report_dir = Path(report_dir)
    for name, key in [("client_domain_counts.csv", "_client_domain_counts"), ("station_domain_counts.csv", "_station_domain_counts")]:
        rows = diagnostics.get(key, [])
        path = report_dir / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            n_domains = len(rows[0]) if rows else 0
            writer.writerow(["id"] + ["domain_{}".format(i) for i in range(n_domains)] + ["total"])
            for idx, row in enumerate(rows):
                writer.writerow([idx] + list(row) + [int(sum(row))])


def save_partition_summary_md(report_dir, diagnostics):
    path = Path(report_dir) / "partition_summary.md"
    lines = [
        "# Partition Summary",
        "",
        "- Method: `{}` ({})".format(diagnostics["partition_method"], diagnostics["paper_name"]),
        "- Seed: `{}`".format(diagnostics["seed"]),
        "- Clients/stations: `{}/{}`".format(diagnostics["num_clients"], diagnostics["num_stations"]),
        "- Eta/lambda: `{:.4f}` / `{:.4f}`".format(
            diagnostics["partition_eta"], diagnostics["partition_lambda"]
        ),
        "- Total assigned/train: `{}/{}`".format(diagnostics["total_assigned_samples"], diagnostics["total_train_samples"]),
        "- Missing/duplicated/empty clients: `{}/{}/{}`".format(
            diagnostics["missing_sample_count"],
            diagnostics["duplicated_sample_count"],
            diagnostics["empty_client_count"],
        ),
        "- Inter-station JS mean: `{:.4f}`".format(diagnostics["inter_station_js_mean"]),
        "- Intra-station JS mean: `{:.4f}`".format(diagnostics["intra_station_js_mean"]),
        "- Client/station sample CV: `{:.4f}` / `{:.4f}`".format(
            diagnostics["client_sample_cv"], diagnostics["station_sample_cv"]
        ),
        "- Hard filters passed: `{}`".format(diagnostics["valid_hard_filters_passed"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_partition_plots(report_dir, diagnostics):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn("Skipping partition plots because matplotlib is unavailable: {}".format(exc))
        return
    report_dir = Path(report_dir)
    for name, key in [
        ("client_domain_stacked.png", "_client_domain_counts"),
        ("station_domain_stacked.png", "_station_domain_counts"),
    ]:
        counts = np.asarray(diagnostics.get(key, []), dtype=np.float64)
        if counts.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        bottom = np.zeros(counts.shape[0])
        for domain_id in range(counts.shape[1]):
            ax.bar(np.arange(counts.shape[0]), counts[:, domain_id], bottom=bottom, label="d{}".format(domain_id))
            bottom += counts[:, domain_id]
        ax.set_title(name.replace("_", " "))
        ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        fig.savefig(report_dir / name)
        plt.close(fig)
    counts = np.asarray(diagnostics.get("_station_domain_counts", []), dtype=np.float64)
    if counts.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(counts, aspect="auto")
        ax.set_title("station domain heatmap")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(report_dir / "station_domain_heatmap.png")
        plt.close(fig)


def _save_reports(cfg: PartitionConfig, hparam, diagnostics):
    if not cfg.save_report:
        return None
    report_dir = cfg.report_dir
    if report_dir is None:
        base = hparam.get("data_path", "./outputs/")
        report_dir = os.path.join(base, "partition_reports", hparam.get("id", "partition_run"))
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    save_partition_report_json(report_dir, diagnostics)
    save_partition_metrics_csv(report_dir, diagnostics)
    save_partition_counts_csv(report_dir, diagnostics)
    save_partition_summary_md(report_dir, diagnostics)
    if cfg.plot:
        save_partition_plots(report_dir, diagnostics)
    return report_dir


def _log_wandb(diagnostics, hparam):
    if not hparam.get("wandb", False):
        return
    try:
        import wandb
        wandb.log({
            "partition/inter_station_js_mean": diagnostics["inter_station_js_mean"],
            "partition/intra_station_js_mean": diagnostics["intra_station_js_mean"],
            "partition/client_sample_cv": diagnostics["client_sample_cv"],
            "partition/station_sample_cv": diagnostics["station_sample_cv"],
            "partition/empty_client_count": diagnostics["empty_client_count"],
        }, commit=False)
    except Exception as exc:
        warnings.warn("Failed to log partition diagnostics to wandb: {}".format(exc))


def build_hierarchical_partition(train_subset, domain_field, transform, hparam) -> PartitionResult:
    cfg = _partition_config_from_hparam(hparam, domain_field)
    if cfg.method in {"hds_label", "hds_full"}:
        raise NotImplementedError("{} is future work; label skew is not enabled in this phase.".format(cfg.method))

    repair_count = 0
    if cfg.method in {"paper_lambda", "paper_lambda_clustered", "paper_lambda_mixed"}:
        client_datasets = _paper_lambda_clients(train_subset, domain_field, transform, cfg)
        assignment = cfg.station_assignment
        if cfg.method == "paper_lambda_clustered":
            assignment = "domain_clustered"
        elif cfg.method == "paper_lambda_mixed":
            assignment = "domain_mixed"
        station_client_indices = assign_clients_to_stations(
            client_datasets,
            num_stations=cfg.num_stations,
            assignment=assignment,
            domain_field=domain_field,
            seed=cfg.seed,
        )
    elif cfg.method in {"hds_eta_lambda", "hds_dirichlet", "hds_inter", "hds_intra", "hds_severe", "hds_quantity", "hds_partial"}:
        attempts = cfg.max_resample_attempts if cfg.resample_until_nonempty else 1
        last = None
        for attempt in range(max(1, attempts)):
            attempt_cfg = PartitionConfig(**{**asdict(cfg), "seed": cfg.seed + attempt})
            client_indices, station_client_indices, repair_count = _hds_client_indices(train_subset, attempt_cfg)
            client_datasets = [make_subset_like(train_subset, idxs, transform=transform) for idxs in client_indices]
            tmp_diag = partition_diagnostics(cfg, train_subset, client_datasets, station_client_indices, repair_count=repair_count)
            last = (client_datasets, station_client_indices, tmp_diag)
            if tmp_diag["empty_client_count"] == 0:
                break
        client_datasets, station_client_indices, _ = last
    else:
        raise ValueError("Unknown partition_method '{}'.".format(cfg.method))

    diagnostics = partition_diagnostics(cfg, train_subset, client_datasets, station_client_indices, repair_count=repair_count)
    report_dir = _save_reports(cfg, hparam, diagnostics)
    if report_dir:
        diagnostics["report_dir"] = str(report_dir)
    _log_wandb(diagnostics, hparam)
    return PartitionResult(
        client_datasets=client_datasets,
        station_client_indices=station_client_indices,
        diagnostics=diagnostics,
    )
