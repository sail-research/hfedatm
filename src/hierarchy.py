from collections import Counter, OrderedDict
import copy

import numpy as np
import torch

from .fedrc import aggregate_gaussian_summaries
from .fediir import aggregate_gradient_sums, ema_gradient_mean
from .merging import StationPayload, clone_state_dict_to_cpu
from .sketches import aggregate_sketches


def _to_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    if hasattr(values, "cpu"):
        return values.cpu().numpy()
    return np.asarray(values)


def get_domain_field_index(dataset, domain_field):
    if isinstance(domain_field, (list, tuple)):
        domain_field = domain_field[0]
    if isinstance(domain_field, str):
        fields = getattr(dataset, "_metadata_fields", None)
        if fields is None and hasattr(dataset, "dataset"):
            fields = getattr(dataset.dataset, "_metadata_fields", None)
        if fields is None:
            fields = getattr(dataset, "metadata_fields", None)
        if fields is None and hasattr(dataset, "dataset"):
            fields = getattr(dataset.dataset, "metadata_fields", None)
        if fields is None:
            raise ValueError("Dataset does not expose metadata fields.")
        return fields.index(domain_field)
    return int(domain_field)


def get_domain_counts(dataset, domain_field, minlength=None):
    field_idx = get_domain_field_index(dataset, domain_field)
    if hasattr(dataset, "metadata_array"):
        metadata = dataset.metadata_array
    elif hasattr(dataset, "_metadata_array"):
        metadata = dataset._metadata_array
    elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        parent_metadata = dataset.dataset.metadata_array if hasattr(dataset.dataset, "metadata_array") else dataset.dataset._metadata_array
        metadata = _to_numpy(parent_metadata)[np.asarray(dataset.indices, dtype=np.int64)]
    else:
        raise ValueError("Dataset does not expose metadata_array.")
    domains = _to_numpy(metadata[:, field_idx]).astype(int)
    if minlength is None:
        minlength = int(domains.max()) + 1 if domains.size else 0
    return np.bincount(domains, minlength=minlength)


def summarize_client_domains(client_datasets, domain_field, minlength=None):
    if minlength is None:
        max_domain = 0
        for dataset in client_datasets:
            counts = get_domain_counts(dataset, domain_field)
            if counts.size:
                max_domain = max(max_domain, counts.size - 1)
        minlength = max_domain + 1
    return np.stack([
        get_domain_counts(dataset, domain_field, minlength=minlength)
        for dataset in client_datasets
    ])


def assign_clients_to_stations(
    client_datasets,
    num_stations,
    assignment="contiguous",
    domain_field=None,
    seed=0,
):
    """Assign flat client shards to station groups.

    The default "contiguous" mode intentionally preserves the client order
    emitted by NonIIDSplitter. Since that splitter orders clients by dominant
    domain, this yields domain-skewed stations when iid/lambda is small.
    """
    num_clients = len(client_datasets)
    if num_stations < 1:
        raise ValueError("num_stations must be at least 1")
    if num_clients < num_stations:
        raise ValueError("num_clients must be greater than or equal to num_stations")

    assignment = assignment.lower()
    aliases = {
        "clustered": "domain_clustered",
        "domain_skewed": "domain_clustered",
        "skewed": "contiguous",
        "mixed": "domain_mixed",
        "domain_balanced": "domain_mixed",
        "balanced_by_order": "round_robin",
        "random": "random_balanced",
    }
    assignment = aliases.get(assignment, assignment)
    client_ids = np.arange(num_clients)
    rng = np.random.default_rng(seed)

    if assignment == "contiguous":
        groups = np.array_split(client_ids, num_stations)
    elif assignment == "round_robin":
        groups = [client_ids[i::num_stations] for i in range(num_stations)]
    elif assignment == "random_balanced":
        groups = np.array_split(rng.permutation(client_ids), num_stations)
    elif assignment == "balanced":
        if domain_field is None:
            raise ValueError("domain_field is required for balanced station assignment")
        client_domain_counts = summarize_client_domains(client_datasets, domain_field)
        dominant_domains = np.argmax(client_domain_counts, axis=1)
        client_sizes = client_domain_counts.sum(axis=1)
        station_domain_counts = np.zeros((num_stations, client_domain_counts.shape[1]), dtype=np.int64)
        station_sizes = np.zeros(num_stations, dtype=np.int64)
        groups = [[] for _ in range(num_stations)]
        order = sorted(
            range(num_clients),
            key=lambda idx: (-client_sizes[idx], dominant_domains[idx], idx),
        )
        for client_id in order:
            dominant = dominant_domains[client_id]
            station_id = min(
                range(num_stations),
                key=lambda sid: (
                    station_domain_counts[sid, dominant],
                    station_sizes[sid],
                    len(groups[sid]),
                    sid,
                ),
            )
            groups[station_id].append(client_id)
            station_domain_counts[station_id] += client_domain_counts[client_id]
            station_sizes[station_id] += client_sizes[client_id]
    elif assignment == "domain_clustered":
        if domain_field is None:
            raise ValueError("domain_field is required for domain_clustered station assignment")
        client_domain_counts = summarize_client_domains(client_datasets, domain_field)
        client_sizes = client_domain_counts.sum(axis=1)
        proportions = np.divide(
            client_domain_counts.astype(float),
            np.maximum(client_sizes[:, None], 1),
        )
        dominant_domains = np.argmax(proportions, axis=1)
        purity = np.max(proportions, axis=1)
        order = sorted(
            range(num_clients),
            key=lambda idx: (dominant_domains[idx], -purity[idx], -client_sizes[idx], idx),
        )
        groups = np.array_split(np.asarray(order, dtype=int), num_stations)
    elif assignment == "domain_mixed":
        if domain_field is None:
            raise ValueError("domain_field is required for domain_mixed station assignment")
        client_domain_counts = summarize_client_domains(client_datasets, domain_field)
        client_sizes = client_domain_counts.sum(axis=1)
        proportions = np.divide(
            client_domain_counts.astype(float),
            np.maximum(client_sizes[:, None], 1),
        )
        global_prop = proportions.mean(axis=0)
        station_counts = np.zeros((num_stations, client_domain_counts.shape[1]), dtype=float)
        groups = [[] for _ in range(num_stations)]
        capacity = int(np.ceil(num_clients / num_stations))
        order = sorted(
            range(num_clients),
            key=lambda idx: (-np.max(proportions[idx]), int(np.argmax(proportions[idx])), idx),
        )
        for client_id in order:
            best_station = None
            best_score = None
            for station_id in range(num_stations):
                if len(groups[station_id]) >= capacity:
                    continue
                candidate = station_counts[station_id] + client_domain_counts[client_id]
                candidate_prop = candidate / max(candidate.sum(), 1.0)
                balance_penalty = abs((len(groups[station_id]) + 1) - (num_clients / num_stations))
                score = float(np.sum((candidate_prop - global_prop) ** 2)) + 0.001 * balance_penalty
                if best_score is None or score < best_score:
                    best_score = score
                    best_station = station_id
            if best_station is None:
                best_station = min(range(num_stations), key=lambda sid: len(groups[sid]))
            groups[best_station].append(int(client_id))
            station_counts[best_station] += client_domain_counts[client_id]
    else:
        raise ValueError(
            "Unknown station assignment '{}'. Use contiguous, round_robin, random_balanced, balanced, domain_clustered, or domain_mixed.".format(
                assignment
            )
        )

    assignments = [sorted([int(idx) for idx in group]) for group in groups]
    flat = sorted(idx for group in assignments for idx in group)
    if flat != list(range(num_clients)):
        raise ValueError("Station assignment must include every client exactly once")
    return assignments


class Station:
    def __init__(self, station_id, client_indices, clients):
        self.station_id = station_id
        self.client_indices = list(client_indices)
        self.clients = {idx: clients[idx] for idx in self.client_indices}
        self.fediir_grad_mean = None

    def __len__(self):
        return sum(len(client) for client in self.clients.values())

    def set_device(self, device):
        for client in self.clients.values():
            if hasattr(client, "set_device"):
                client.set_device(device)
            else:
                client.device = torch.device(device)

    def sample_clients(self, fraction, rng):
        num_clients = len(self.client_indices)
        num_sampled = max(int(fraction * num_clients), 1)
        num_sampled = min(num_sampled, num_clients)
        sampled = rng.choice(self.client_indices, size=num_sampled, replace=False)
        return sorted(int(idx) for idx in sampled.tolist())

    def update_clients(self, client_indices, model_state):
        for idx in client_indices:
            self.clients[idx].update_model(copy.deepcopy(model_state))

    def fit_clients(self, client_indices, server_round):
        total_size = 0
        for idx in client_indices:
            self.clients[idx].fit(server_round)
            total_size += len(self.clients[idx])
        return total_size

    def aggregate_clients(self, client_indices, coefficients):
        averaged_weights = OrderedDict()
        for pos, idx in enumerate(client_indices):
            local_weights = self.clients[idx].model.state_dict()
            for key in local_weights.keys():
                if pos == 0:
                    averaged_weights[key] = coefficients[pos] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[pos] * local_weights[key]
        return averaged_weights

    def prepare_fediir_clients(self, client_indices):
        fediir_clients = [
            self.clients[idx]
            for idx in client_indices
            if hasattr(self.clients[idx], "compute_fediir_classifier_grad_sum")
        ]
        if not fediir_clients:
            return
        if len(fediir_clients) != len(client_indices):
            raise RuntimeError("FedIIR clients cannot be mixed with non-FedIIR clients in one station update")
        gradient_sums = []
        batch_counts = []
        for client in fediir_clients:
            grad_sum, batch_count = client.compute_fediir_classifier_grad_sum(
                max_batches=client.hparam.get("fediir_mean_grad_max_batches", None)
            )
            gradient_sums.append(grad_sum)
            batch_counts.append(batch_count)
        current_mean = aggregate_gradient_sums(gradient_sums, batch_counts)
        if self.fediir_grad_mean is None:
            self.fediir_grad_mean = tuple(torch.zeros_like(grad) for grad in current_mean)
        ema = float(fediir_clients[0].hparam.get("fediir_ema", 0.95))
        self.fediir_grad_mean = ema_gradient_mean(self.fediir_grad_mean, current_mean, ema)
        for client in fediir_clients:
            client.set_fediir_grad_mean(self.fediir_grad_mean)

    def train_station_model(self, model_state, server_round, station_rounds, client_fraction, rng):
        station_state = copy.deepcopy(model_state)
        selected_total_size = 0
        for _ in range(station_rounds):
            sampled_client_indices = self.sample_clients(client_fraction, rng)
            self.update_clients(sampled_client_indices, station_state)
            self.prepare_fediir_clients(sampled_client_indices)
            selected_total_size = self.fit_clients(sampled_client_indices, server_round)
            coefficients = [
                len(self.clients[idx]) / selected_total_size
                for idx in sampled_client_indices
            ]
            station_state = self.aggregate_clients(sampled_client_indices, coefficients)
        return station_state, selected_total_size

    def _sample_weights(self, client_indices):
        sizes = [len(self.clients[idx]) for idx in client_indices]
        total = sum(sizes)
        if total <= 0:
            return [1.0 / len(client_indices) for _ in client_indices]
        return [size / total for size in sizes]

    def _aggregate_fisher(self, fisher_payloads, weights):
        out = {}
        for fisher, weight in zip(fisher_payloads, weights):
            for key, value in fisher.items():
                term = float(weight) * value.detach().cpu()
                out[key] = term if key not in out else out[key] + term
        return out

    def train_station_payload(
        self,
        model_state,
        server_round,
        station_rounds,
        client_fraction,
        rng,
        collect_sketches=False,
        sketch_config=None,
        collect_fisher=False,
        fisher_config=None,
        collect_gaussian=False,
        gaussian_config=None,
    ):
        station_state, selected_total_size = self.train_station_model(
            model_state,
            server_round=server_round,
            station_rounds=station_rounds,
            client_fraction=client_fraction,
            rng=rng,
        )
        stat_client_indices = list(self.client_indices)
        stat_weights = self._sample_weights(stat_client_indices)
        sketches = {}
        fisher = {}
        gaussian = {}

        if collect_sketches:
            sketch_payloads = []
            max_batches = None
            layer_names = None
            if sketch_config:
                max_batches = sketch_config.get("max_batches")
                layer_names = sketch_config.get("layer_names")
            for idx in stat_client_indices:
                sketch_payloads.append(
                    self.clients[idx].collect_activation_sketches(
                        station_state,
                        layer_names=layer_names,
                        max_batches=max_batches,
                        sketch_config=sketch_config,
                    )
                )
            sketches = aggregate_sketches(sketch_payloads, stat_weights)

        if collect_fisher:
            fisher_config = fisher_config or {}
            fisher_payloads = []
            for idx in stat_client_indices:
                fisher_payloads.append(
                    self.clients[idx].compute_diagonal_fisher(
                        station_state,
                        max_batches=fisher_config.get("max_batches", 1),
                        fisher_eps=fisher_config.get("fisher_eps", 1e-8),
                        label_mode=fisher_config.get("label_mode", "true_labels"),
                        fisher_clip=fisher_config.get("fisher_clip", 0.0),
                        fisher_normalize=fisher_config.get("fisher_normalize", False),
                    )
                )
            fisher = self._aggregate_fisher(fisher_payloads, stat_weights)

        if collect_gaussian:
            gaussian_config = gaussian_config or {}
            summaries = []
            for idx in stat_client_indices:
                summaries.append(
                    self.clients[idx].compute_gaussian_summary(
                        station_state,
                        stat_source=gaussian_config.get("stat_source", "rgb"),
                        max_batches=gaussian_config.get("max_batches", 1),
                    )
                )
            gaussian = aggregate_gaussian_summaries(summaries, stat_weights)

        return StationPayload(
            station_id=int(self.station_id),
            state_dict=clone_state_dict_to_cpu(station_state),
            num_samples=int(selected_total_size or len(self)),
            client_indices=[int(idx) for idx in self.client_indices],
            sketches=sketches,
            fisher=fisher,
            gaussian=gaussian,
            metrics={},
        )

    def domain_counts(self, domain_field, minlength=None):
        counts = []
        for client in self.clients.values():
            counts.append(get_domain_counts(client.dataset, domain_field, minlength=minlength))
        if not counts:
            return np.zeros(minlength or 0, dtype=np.int64)
        return np.stack(counts).sum(axis=0)

    @property
    def dominant_domain(self):
        client_sizes_by_domain = Counter()
        for client in self.clients.values():
            counts = get_domain_counts(client.dataset, "domain")
            for domain_id, count in enumerate(counts.tolist()):
                client_sizes_by_domain[domain_id] += count
        if not client_sizes_by_domain:
            return None
        return client_sizes_by_domain.most_common(1)[0][0]
