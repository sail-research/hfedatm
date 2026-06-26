import copy
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from multiprocessing import pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm
from collections import OrderedDict
import torch.distributions as dist

from .models import *
from .utils import *
from .client import *
try:
    from .dataset_bundle import *
except ImportError:
    pass
from .hierarchy import Station, assign_clients_to_stations
from .fedrc import aggregate_gaussian_summaries, fedrc_gaussian_weights
from .fediir import aggregate_gradient_sums, ema_gradient_mean
from .merging import (
    GraphConsistentAligner,
    HungarianAligner,
    StationPayload,
    UnitSignatureBuilder,
    clone_state_dict_to_cpu,
    estimate_tensor_dict_size_mb,
    fisher_weighted_average,
    is_floating_tensor,
    json_dumps_safe,
    normalize_coefficients,
    regmean_merge_parameter,
    shrink_gram,
    state_prefix_for_module_name,
    weighted_average_state_dicts,
)
from .mtgc import MTGCApproxState

try:
    import wandb
except ImportError:
    class _NoWandb:
        summary = {}

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _NoWandb()

class FedAvg(object):
    def __init__(self, device, ds_bundle, hparam):
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = []
        self.hparam = hparam
        self.num_rounds = hparam['num_rounds']
        self.fraction = hparam['fraction']
        self.num_clients = 0
        self.test_dataloader = {}
        self._round = 0
        self.featurizer = None
        self.classifier = None
    
    def setup_model(self, model_file=None, start_epoch=0):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self._tqdm(self.clients):
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
    
    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)

    def _tqdm(self, iterable, **kwargs):
        kwargs.setdefault("disable", bool(self.hparam.get("disable_tqdm", False)))
        return tqdm(iterable, **kwargs)
    
    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in self._tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
        else:
            # send the global model to selected clients
            for idx in self._tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())

    def sample_clients(self):
        """
        Description: Sample a subset of clients. 
        Could be overriden if some methods require specific ways of sampling.
        """
        # sample clients randommly
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    

    def update_clients(self, sampled_client_indices):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        def update_single_client(selected_index):
            self.clients[selected_index].fit(self._round)
            client_size = len(self.clients[selected_index])
            return client_size
        selected_total_size = 0
        for idx in self._tqdm(sampled_client_indices, leave=False):
            client_size = update_single_client(idx)
            selected_total_size += client_size
        return selected_total_size


    def evaluate_clients(self, sampled_client_indices):
        def evaluate_single_client(selected_index):
            self.clients[selected_index].client_evaluate()
            return True
        for idx in self._tqdm(sampled_client_indices):
            self.clients[idx].client_evaluate()
            

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in self._tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)
    

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)
    
    def build_wandb_eval_metrics(self, metric_dict):
        logged_metrics = {"comm_round": self._round}
        for split_name, split_metrics in metric_dict.items():
            for metric_name, metric_value in split_metrics.items():
                if isinstance(metric_value, torch.Tensor):
                    if metric_value.numel() != 1:
                        continue
                    metric_value = metric_value.item()
                elif isinstance(metric_value, np.generic):
                    metric_value = metric_value.item()
                elif not isinstance(metric_value, (int, float)):
                    continue
                logged_metrics[f"eval/{split_name}/{metric_name}"] = metric_value
        return logged_metrics

    def evaluate_global_model(self, dataloader):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = None
            y_true = None
            for batch in self._tqdm(dataloader):
                data, labels, meta_batch = batch[0], batch[1], batch[2]
                if isinstance(meta_batch, list):
                    meta_batch = meta_batch[0]
                data, labels = data.to(self.device), labels.to(self.device)
                if self._featurizer.probabilistic:
                    features_params = self.featurizer(data)
                    z_dim = int(features_params.shape[-1]/2)
                    if len(features_params.shape) == 2:
                        z_mu = features_params[:,:z_dim]
                        z_sigma = F.softplus(features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    elif len(features_params.shape) == 3:
                        flattened_features_params = features_params.view(-1, features_params.shape[-1])
                        z_mu = flattened_features_params[:,:z_dim]
                        z_sigma = F.softplus(flattened_features_params[:,z_dim:])
                        z_dist = dist.Independent(dist.normal.Normal(z_mu,z_sigma),1)
                    features = z_dist.rsample()
                    if len(features_params.shape) == 3:
                        features = features.view(data.shape[0], -1, z_dim)
                else:
                    features = self.featurizer(data)
                prediction = self.classifier(features)
                if self.ds_bundle.is_classification:
                    prediction = torch.argmax(prediction, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = labels
                    metadata = meta_batch
                else:
                    y_pred = torch.cat((y_pred, prediction))
                    y_true = torch.cat((y_true, labels))
                    metadata = torch.cat((metadata, meta_batch))
                # print("DEBUG: server.py:183")
                # break
            if y_pred is None:
                warnings.warn("Received an empty evaluation dataloader; returning no metrics.")
                self.model.to("cpu")
                return {}
            metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            print(metric)
            if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        return metric[0]

    def fit(self):
        """
        Description: Execute the whole process of the federated learning.
        """
        best_id_val_round = 0
        best_id_val_value = 0
        best_id_val_test_value = 0
        best_lodo_val_round = 0
        best_lodo_val_value = 0
        best_lodo_val_test_value = 0

        for r in range(self.num_rounds):
            print("num of rounds: {}".format(r))

            self.train_federated_model()
            metric_dict = {}
            id_flag = False
            lodo_flag = False
            id_t_val = 0
            t_val = 0
            for name, dataloader in self.test_dataloader.items():
                metric = self.evaluate_global_model(dataloader)
                if not metric:
                    continue
                metric_dict[name] = metric
                
                if name == 'val':
                    lodo_val = metric[self.ds_bundle.key_metric]
                    if lodo_val > best_lodo_val_value:
                        best_lodo_val_round = r
                        best_lodo_val_value = lodo_val
                        lodo_flag = True
                if name == 'id_val':
                    id_val = metric[self.ds_bundle.key_metric]
                    if id_val > best_id_val_value:
                        best_id_val_round = r
                        best_id_val_value = id_val
                        id_flag = True
                if name == 'test':
                    t_val = metric[self.ds_bundle.key_metric]
                if name == 'id_test':
                    id_t_val = metric[self.ds_bundle.key_metric]
            if lodo_flag:
                best_lodo_val_test_value = t_val
            if id_flag:
                best_id_val_test_value = id_t_val
            
            print(metric_dict)
            if self.hparam['wandb']:
                wandb.log(
                    self.build_wandb_eval_metrics(metric_dict),
                    step=self._round*self.hparam['local_epochs'],
                )
            self.save_model(r)
            self._round += 1
        if self.hparam['wandb']:
            if best_id_val_round != 0: 
                wandb.summary['best_id_round'] = best_id_val_round
                wandb.summary['best_id_val_acc'] = best_id_val_test_value
            if best_lodo_val_round != 0:
                wandb.summary['best_lodo_round'] = best_lodo_val_round
                wandb.summary['best_lodo_val_acc'] = best_lodo_val_test_value
        else:
            print(f"best_id_round: {best_id_val_round}")
            print(f"best_id_val_acc: {best_id_val_test_value}")
            print(f"best_lodo_round: {best_lodo_val_round}")
            print(f"best_lodo_val_acc: {best_lodo_val_test_value}")
        self.transmit_model()

    def save_model(self, num_epoch):
        if not self.hparam.get("save_checkpoints", False):
            return
        path = f"{self.hparam['data_path']}/models/{self.ds_bundle.name}_{self.clients[0].name}_{self.hparam['id']}_{num_epoch}.pth"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


class FedIIRServer(FedAvg):
    """Flat FedAvg server with exact FedIIR inter-client gradient alignment.

    This follows the official FedIIR implementation pattern: sample clients,
    broadcast the global model, compute the EMA-smoothed mean classifier
    gradient over sampled client batches, run local FedIIR updates, then
    FedAvg aggregate the updated models.
    """

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.fediir_grad_mean = None

    def prepare_fediir_clients(self, sampled_client_indices):
        selected_clients = [self.clients[idx] for idx in sampled_client_indices]
        if not all(hasattr(client, "compute_fediir_classifier_grad_sum") for client in selected_clients):
            raise RuntimeError("FedIIRServer requires client_method='FedIIR' for all sampled clients")
        gradient_sums = []
        batch_counts = []
        for client in selected_clients:
            grad_sum, batch_count = client.compute_fediir_classifier_grad_sum(
                max_batches=self.hparam.get("fediir_mean_grad_max_batches", None)
            )
            gradient_sums.append(grad_sum)
            batch_counts.append(batch_count)
        current_mean = aggregate_gradient_sums(gradient_sums, batch_counts)
        if self.fediir_grad_mean is None:
            self.fediir_grad_mean = tuple(torch.zeros_like(grad) for grad in current_mean)
        self.fediir_grad_mean = ema_gradient_mean(
            self.fediir_grad_mean,
            current_mean,
            float(self.hparam.get("fediir_ema", 0.95)),
        )
        for client in selected_clients:
            client.set_fediir_grad_mean(self.fediir_grad_mean)

    def train_federated_model(self):
        sampled_client_indices = self.sample_clients()
        self.transmit_model(sampled_client_indices)
        self.prepare_fediir_clients(sampled_client_indices)
        selected_total_size = self.update_clients(sampled_client_indices)
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)


class HierarchicalFedAvg(FedAvg):
    """Three-tier HFL baseline: clients -> stations -> server.

    This keeps the existing client-side algorithms untouched. The station layer
    only orchestrates repeated client aggregation before the server receives one
    model per station.
    """
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.num_stations = int(hparam.get("num_stations", 1))
        self.station_rounds = int(hparam.get("station_rounds", 1))
        self.station_fraction = float(hparam.get("station_fraction", 1.0))
        self.station_client_fraction = float(hparam.get("station_client_fraction", 1.0))
        self.station_assignment = hparam.get("station_assignment", "contiguous")
        self.rng = np.random.default_rng(int(hparam.get("seed", 0)))
        self.stations = []

    def register_stations(self, station_client_indices):
        if len(station_client_indices) != self.num_stations:
            raise ValueError(
                "Expected {} station groups, got {}".format(
                    self.num_stations, len(station_client_indices)
                )
            )
        self.stations = [
            Station(station_id, client_indices, self.clients)
            for station_id, client_indices in enumerate(station_client_indices)
        ]
        for station in self.stations:
            if len(station.client_indices) == 0:
                raise ValueError("Station {} has no clients".format(station.station_id))

    def _ensure_stations(self):
        if self.stations:
            return
        station_client_indices = assign_clients_to_stations(
            [client.dataset for client in self.clients],
            num_stations=self.num_stations,
            assignment=self.station_assignment,
            domain_field=self.ds_bundle.groupby_fields,
            seed=int(self.hparam.get("seed", 0)),
        )
        self.register_stations(station_client_indices)

    def sample_stations(self):
        num_sampled = max(int(self.station_fraction * self.num_stations), 1)
        num_sampled = min(num_sampled, self.num_stations)
        sampled = self.rng.choice(
            np.arange(self.num_stations),
            size=num_sampled,
            replace=False,
        )
        return sorted(int(idx) for idx in sampled.tolist())

    def station_parallel_devices(self):
        raw = self.hparam.get("station_parallel_gpus", "")
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)):
            parts = [str(item).strip() for item in raw]
        else:
            parts = [part.strip() for part in str(raw).split(",")]
        parts = [part for part in parts if part]
        if not parts:
            return []
        if not torch.cuda.is_available():
            raise RuntimeError("station_parallel_gpus requires CUDA.")
        devices = []
        visible_count = torch.cuda.device_count()
        for part in parts:
            idx = int(part)
            if idx < 0 or idx >= visible_count:
                raise ValueError(
                    "station_parallel_gpus uses CUDA index {}, but this process sees {} CUDA device(s). "
                    "Use indices relative to CUDA_VISIBLE_DEVICES.".format(idx, visible_count)
                )
            devices.append(torch.device(f"cuda:{idx}"))
        return devices

    def _station_rngs(self, count):
        return [
            np.random.default_rng(int(self.rng.integers(0, np.iinfo(np.uint32).max)))
            for _ in range(count)
        ]

    def _run_station_jobs(self, station_ids, job_fn):
        devices = self.station_parallel_devices()
        rngs = self._station_rngs(len(station_ids))
        results = [None for _ in station_ids]

        def run_one(position, station_id):
            if devices:
                self.stations[station_id].set_device(devices[position % len(devices)])
            return position, job_fn(station_id, rngs[position])

        if not devices or len(station_ids) <= 1:
            for position, station_id in enumerate(self._tqdm(station_ids, leave=False)):
                _, result = run_one(position, station_id)
                results[position] = result
            return results

        max_workers = min(len(devices), len(station_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_one, position, station_id)
                for position, station_id in enumerate(station_ids)
            ]
            for future in as_completed(futures):
                position, result = future.result()
                results[position] = result
        return results

    def aggregate_station_models(self, station_states, coefficients):
        averaged_weights = OrderedDict()
        for pos, station_state in enumerate(station_states):
            for key in self.model.state_dict().keys():
                if pos == 0:
                    averaged_weights[key] = coefficients[pos] * station_state[key]
                else:
                    averaged_weights[key] += coefficients[pos] * station_state[key]
        self.model.load_state_dict(averaged_weights)

    def _diagnostics_path(self):
        run_id = self.hparam.get("id", "default")
        method = self.hparam.get("server_method", self.__class__.__name__)
        diag_dir = os.path.join(self.hparam.get("data_path", "."), "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        return os.path.join(diag_dir, f"{run_id}_{method}.jsonl")

    def _log_aggregation_diagnostics(self, diagnostics):
        log_enabled = self.hparam.get("hfedatm_log_diagnostics", True)
        if log_enabled:
            with open(self._diagnostics_path(), "a", encoding="utf-8") as fh:
                fh.write(json_dumps_safe(diagnostics) + "\n")
        if self.hparam.get("wandb", False):
            wandb.log(
                {
                    "comm_round": self._round,
                    "agg/latency": diagnostics.get("aggregation_latency_sec", 0.0),
                    "agg/sketch_mb": diagnostics.get("sketch_mb", 0.0),
                    "agg/fisher_mb": diagnostics.get("fisher_mb", 0.0),
                    "agg/alignment_cost_mean": diagnostics.get("alignment_cost_mean", 0.0),
                    "agg/fallback_count": sum(diagnostics.get("fallback_counts", {}).values()),
                },
                step=self._round * self.hparam["local_epochs"],
            )

    def train_federated_model(self):
        self._ensure_stations()
        sampled_station_indices = self.sample_stations()
        global_state = clone_state_dict_to_cpu(self.model.state_dict())
        t0 = time.time()

        def station_job(station_id, station_rng):
            return self.stations[station_id].train_station_model(
                global_state,
                server_round=self._round,
                station_rounds=self.station_rounds,
                client_fraction=self.station_client_fraction,
                rng=station_rng,
            )

        station_results = self._run_station_jobs(sampled_station_indices, station_job)
        station_states = [state for state, _ in station_results]
        station_sizes = [size for _, size in station_results]

        total_size = sum(station_sizes)
        coefficients = [station_size / total_size for station_size in station_sizes]
        self.aggregate_station_models(station_states, coefficients)
        diagnostics = {
            "server_method": self.__class__.__name__,
            "round": int(self._round),
            "station_ids": sampled_station_indices,
            "station_sizes": [int(size) for size in station_sizes],
            "coefficients": coefficients,
            "aggregation_latency_sec": time.time() - t0,
            "station_parallel_gpus": [str(device) for device in self.station_parallel_devices()],
            "sketch_mb": 0.0,
            "fisher_mb": 0.0,
            "gaussian_summary_mb": 0.0,
            "conv_layers_aligned": 0,
            "conv_layers_merged": 0,
            "linear_layers_merged": 0,
            "norm_layers_handled": 0,
            "attention_projections_handled": 0,
            "alignment_cost_mean": 0.0,
            "alignment_cost_std": 0.0,
            "regmean_solve_failures": 0,
            "fallback_counts": {},
        }
        self._log_aggregation_diagnostics(diagnostics)


class HierarchicalMergeServer(HierarchicalFedAvg):
    """Base class for station-server payload aggregation methods."""

    collect_sketches = False
    collect_fisher = False
    collect_gaussian = False

    def _sketch_config(self):
        return {
            "mode": self.hparam.get("activation_sketch_mode", "diag"),
            "max_batches": int(self.hparam.get("activation_sketch_max_batches", 1)),
            "max_patches": int(self.hparam.get("activation_sketch_max_patches", 2048)),
            "max_full_dim": int(self.hparam.get("activation_sketch_max_full_dim", 4096)),
            "block_size": int(self.hparam.get("activation_sketch_block_size", 512)),
            "dtype": self.hparam.get("activation_sketch_dtype", "float32"),
            "device": self.hparam.get("activation_sketch_device", "cpu"),
            "shrinkage_alpha": float(self.hparam.get("activation_sketch_shrinkage_alpha", 0.75)),
            "dp_epsilon": float(self.hparam.get("activation_sketch_dp_epsilon", -1.0)),
            "dp_delta": float(self.hparam.get("activation_sketch_dp_delta", 1e-5)),
            "dp_clip": float(self.hparam.get("activation_sketch_dp_clip", 0.0)),
            "lowrank_rank": int(self.hparam.get("activation_sketch_lowrank_rank", 64)),
            "random_projection_dim": int(self.hparam.get("activation_sketch_random_projection_dim", 256)),
            "random_seed": int(self.hparam.get("activation_sketch_random_seed", 0)),
        }

    def _fisher_config(self):
        return {
            "max_batches": int(self.hparam.get("fisher_batches", 1)),
            "fisher_eps": float(self.hparam.get("fisher_eps", 1e-8)),
            "label_mode": self.hparam.get("fisher_label_mode", "true_labels"),
            "fisher_clip": float(self.hparam.get("fisher_clip", 0.0)),
            "fisher_normalize": bool(self.hparam.get("fisher_normalize", False)),
        }

    def _gaussian_config(self):
        return {
            "stat_source": self.hparam.get("fedrc_stat_source", "rgb"),
            "max_batches": int(self.hparam.get("fedrc_max_batches", 1)),
        }

    def station_coefficients(self, payloads):
        weighting = self.hparam.get("hfedatm_station_weighting", "num_samples")
        if weighting == "uniform":
            return [1.0 / len(payloads) for _ in payloads]
        return normalize_coefficients([payload.num_samples for payload in payloads])

    def aggregate_station_payloads(self, payloads, coefficients, diagnostics=None):
        fallback_counts = diagnostics.setdefault("fallback_counts", {}) if diagnostics is not None else None
        return weighted_average_state_dicts(
            payloads,
            coefficients,
            reference_state=clone_state_dict_to_cpu(self.model.state_dict()),
            fallback_counts=fallback_counts,
        )

    def _diagnostics_path(self):
        run_id = self.hparam.get("id", "default")
        method = self.hparam.get("server_method", self.__class__.__name__)
        diag_dir = os.path.join(self.hparam.get("data_path", "."), "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        return os.path.join(diag_dir, f"{run_id}_{method}.jsonl")

    def _write_diagnostics(self, diagnostics):
        log_enabled = self.hparam.get("hfedatm_log_diagnostics", True)
        if not log_enabled:
            return
        with open(self._diagnostics_path(), "a", encoding="utf-8") as fh:
            fh.write(json_dumps_safe(diagnostics) + "\n")

    def _log_aggregation_diagnostics(self, diagnostics):
        self._write_diagnostics(diagnostics)
        if self.hparam.get("wandb", False):
            costs = diagnostics.get("alignment_costs", [])
            fallback_count = sum(diagnostics.get("fallback_counts", {}).values())
            wandb.log(
                {
                    "comm_round": self._round,
                    "agg/latency": diagnostics.get("aggregation_latency_sec", 0.0),
                    "agg/sketch_mb": diagnostics.get("sketch_mb", 0.0),
                    "agg/fisher_mb": diagnostics.get("fisher_mb", 0.0),
                    "agg/alignment_cost_mean": float(np.mean(costs)) if costs else 0.0,
                    "agg/fallback_count": fallback_count,
                },
                step=self._round * self.hparam["local_epochs"],
            )

    def train_federated_model(self):
        self._ensure_stations()
        sampled_station_indices = self.sample_stations()
        global_state = clone_state_dict_to_cpu(self.model.state_dict())
        t0 = time.time()

        def station_job(station_id, station_rng):
            return self.stations[station_id].train_station_payload(
                global_state,
                server_round=self._round,
                station_rounds=self.station_rounds,
                client_fraction=self.station_client_fraction,
                rng=station_rng,
                collect_sketches=self.collect_sketches,
                sketch_config=self._sketch_config(),
                collect_fisher=self.collect_fisher,
                fisher_config=self._fisher_config(),
                collect_gaussian=self.collect_gaussian,
                gaussian_config=self._gaussian_config(),
            )

        payloads = self._run_station_jobs(sampled_station_indices, station_job)

        coefficients = self.station_coefficients(payloads)
        diagnostics = {
            "server_method": self.__class__.__name__,
            "round": int(self._round),
            "station_ids": [payload.station_id for payload in payloads],
            "station_sizes": [payload.num_samples for payload in payloads],
            "coefficients": coefficients,
            "station_parallel_gpus": [str(device) for device in self.station_parallel_devices()],
            "sketch_mb": estimate_tensor_dict_size_mb([payload.sketches for payload in payloads]),
            "fisher_mb": estimate_tensor_dict_size_mb([payload.fisher for payload in payloads]),
            "gaussian_summary_mb": estimate_tensor_dict_size_mb([payload.gaussian for payload in payloads]),
            "conv_layers_aligned": 0,
            "conv_layers_merged": 0,
            "linear_layers_merged": 0,
            "norm_layers_handled": 0,
            "attention_projections_handled": 0,
            "alignment_costs": [],
            "regmean_solve_failures": 0,
            "fallback_counts": {},
        }
        new_state = self.aggregate_station_payloads(payloads, coefficients, diagnostics=diagnostics)
        diagnostics["aggregation_latency_sec"] = time.time() - t0
        costs = diagnostics.get("alignment_costs", [])
        diagnostics["alignment_cost_mean"] = float(np.mean(costs)) if costs else 0.0
        diagnostics["alignment_cost_std"] = float(np.std(costs)) if costs else 0.0
        self.model.load_state_dict(new_state)
        self._log_aggregation_diagnostics(diagnostics)

    def _module_dict(self):
        root = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return dict(root.named_modules())

    def _key(self, module_name, suffix):
        return f"{state_prefix_for_module_name(module_name)}.{suffix}"

    def _reference_index(self, payloads, reference="first"):
        if reference == "largest_station":
            sizes = [payload.num_samples for payload in payloads]
            return int(np.argmax(sizes))
        if reference == "best_source_val":
            warnings.warn("best_source_val reference requested but no source-val station metric exists; using first station.")
        return 0

    def _scope_allows(self, kind, scope):
        scope = str(scope).lower()
        if scope == "all":
            return True
        if scope == "conv_linear":
            return kind in {"conv", "linear"}
        return kind == scope

    def _weighted_tensor(self, values, coefficients, dtype=None):
        merged = None
        for coeff, value in zip(coefficients, values):
            term = float(coeff) * value.detach().cpu()
            merged = term if merged is None else merged + term
        return merged.to(dtype=dtype or values[0].dtype)

    def _payload_tensors(self, payloads, key):
        values = [payload.state_dict.get(key) for payload in payloads]
        if any(value is None for value in values):
            return None
        if not all(is_floating_tensor(value) for value in values):
            return None
        if any(value.shape != values[0].shape for value in values):
            return None
        return values

    def _align_output_units(self, values, coefficients, aligner, diagnostics, allow_alignment=True):
        if not allow_alignment or len(values) <= 1:
            return list(values), [None for _ in values]
        ref = values[0]
        aligned = [ref.detach().cpu()]
        permutations = [torch.arange(ref.shape[0], dtype=torch.long)]
        try:
            ref_sig = UnitSignatureBuilder.output_signatures_from_weight(ref)
        except Exception:
            diagnostics["fallback_counts"]["signature_unavailable"] = diagnostics["fallback_counts"].get("signature_unavailable", 0) + 1
            return list(values), [None for _ in values]
        for value in values[1:]:
            try:
                tgt_sig = UnitSignatureBuilder.output_signatures_from_weight(value)
                perm, cost = aligner.align(ref_sig, tgt_sig)
                if perm.numel() != value.shape[0]:
                    raise ValueError("Permutation length does not match output dimension")
                aligned.append(value.detach().cpu().index_select(0, perm))
                permutations.append(perm)
                diagnostics["alignment_costs"].append(cost)
            except Exception as exc:
                warnings.warn(f"Alignment fallback: {exc}")
                diagnostics["fallback_counts"]["alignment_failed"] = diagnostics["fallback_counts"].get("alignment_failed", 0) + 1
                aligned.append(value.detach().cpu())
                permutations.append(None)
        return aligned, permutations

    def _merge_functional_layers(
        self,
        payloads,
        coefficients,
        diagnostics,
        use_regmean=True,
        regmean_kinds=None,
        align=False,
        scope="all",
        reference="first",
        align_solver=None,
    ):
        fallback_counts = diagnostics.setdefault("fallback_counts", {})
        new_state = weighted_average_state_dicts(
            payloads,
            coefficients,
            reference_state=clone_state_dict_to_cpu(self.model.state_dict()),
            fallback_counts=fallback_counts,
        )
        module_dict = self._module_dict()
        regmean_kinds = set(regmean_kinds or {"conv", "linear"})
        align_solver = align_solver or self.hparam.get("align_solver", "hungarian")
        aligner_kwargs = {
            "reg": float(self.hparam.get("ot_reg", 0.05)),
            "iters": int(self.hparam.get("ot_iters", 25)),
        }
        aligner = (
            GraphConsistentAligner(align_solver, **aligner_kwargs)
            if self.hparam.get("graph_consistency", True)
            else HungarianAligner(align_solver, **aligner_kwargs)
        )
        ref_idx = self._reference_index(payloads, reference=reference)
        if ref_idx != 0:
            payloads = [payloads[ref_idx]] + [payload for i, payload in enumerate(payloads) if i != ref_idx]
            coefficients = [coefficients[ref_idx]] + [coeff for i, coeff in enumerate(coefficients) if i != ref_idx]
            coefficients = normalize_coefficients(coefficients)

        for module_name, module in module_dict.items():
            if isinstance(module, nn.Conv2d):
                kind = "conv"
            elif isinstance(module, nn.Linear):
                kind = "linear"
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                diagnostics["norm_layers_handled"] += 1
                continue
            else:
                continue
            if not self._scope_allows(kind, scope):
                continue
            weight_key = self._key(module_name, "weight")
            bias_key = self._key(module_name, "bias")
            weights = self._payload_tensors(payloads, weight_key)
            if weights is None:
                fallback_counts["missing_or_mismatched_weight"] = fallback_counts.get("missing_or_mismatched_weight", 0) + 1
                continue

            is_classifier = isinstance(module, nn.Linear) and module_name == "1"
            allow_alignment = align and not is_classifier
            aligned_weights, permutations = self._align_output_units(
                weights,
                coefficients,
                aligner,
                diagnostics,
                allow_alignment=allow_alignment,
            )
            if allow_alignment:
                diagnostics["conv_layers_aligned"] += int(kind == "conv")

            merged_weight = self._weighted_tensor(aligned_weights, coefficients, dtype=new_state[weight_key].dtype)
            if use_regmean and kind in regmean_kinds:
                grams = []
                for payload in payloads:
                    sketch = payload.sketches.get(module_name, {})
                    gram = sketch.get("G")
                    if gram is not None and gram.dim() == 2:
                        gram = shrink_gram(gram, self.hparam.get("activation_sketch_shrinkage_alpha", 0.75))
                    grams.append(gram)
                merged_weight, ok = regmean_merge_parameter(
                    aligned_weights,
                    grams,
                    coefficients,
                )
                if not ok:
                    diagnostics["regmean_solve_failures"] += 1
                    fallback_counts["missing_sketch_or_regmean_failed"] = fallback_counts.get("missing_sketch_or_regmean_failed", 0) + 1
                    merged_weight = self._weighted_tensor(aligned_weights, coefficients, dtype=new_state[weight_key].dtype)
            new_state[weight_key] = merged_weight.to(dtype=new_state[weight_key].dtype)

            biases = self._payload_tensors(payloads, bias_key) if bias_key in new_state else None
            if biases is not None:
                aligned_biases = []
                for bias, perm in zip(biases, permutations):
                    if perm is not None and allow_alignment:
                        aligned_biases.append(bias.detach().cpu().index_select(0, perm))
                    else:
                        aligned_biases.append(bias.detach().cpu())
                new_state[bias_key] = self._weighted_tensor(aligned_biases, coefficients, dtype=new_state[bias_key].dtype)

            diagnostics["conv_layers_merged"] += int(kind == "conv")
            diagnostics["linear_layers_merged"] += int(kind == "linear")
            lower_name = module_name.lower()
            if any(token in lower_name for token in ["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value", "dense"]):
                diagnostics["attention_projections_handled"] += 1

        if self.hparam.get("residual_consistency", True):
            fallback_counts["residual_graph_tracing_conservative"] = fallback_counts.get("residual_graph_tracing_conservative", 0) + 1
        return new_state


class HFedATM(HierarchicalMergeServer):
    """Original HFedATM: FOT-style Conv2d matching plus Linear RegMean."""

    collect_sketches = True

    def _sketch_config(self):
        config = super()._sketch_config()
        config["layer_names"] = [
            name
            for name, module in self._module_dict().items()
            if isinstance(module, nn.Linear)
        ]
        return config

    def aggregate_station_payloads(self, payloads, coefficients, diagnostics=None):
        return self._merge_functional_layers(
            payloads,
            coefficients,
            diagnostics or {},
            use_regmean=True,
            regmean_kinds={"linear"},
            align=True,
            scope="all",
            reference=self.hparam.get("ot_reference", "first"),
            align_solver=self.hparam.get("align_solver", "hungarian"),
        )



class RegMeanAll(HierarchicalMergeServer):
    """RegMean-style functional merge for Conv2d/Linear without graph alignment."""

    collect_sketches = True

    def aggregate_station_payloads(self, payloads, coefficients, diagnostics=None):
        return self._merge_functional_layers(
            payloads,
            coefficients,
            diagnostics or {},
            use_regmean=True,
            regmean_kinds={"conv", "linear"},
            align=False,
            scope=self.hparam.get("regmean_all_scope", "all"),
            reference="first",
        )


class FisherMerging(HierarchicalMergeServer):
    """Fisher-weighted averaging baseline."""

    collect_fisher = True

    def aggregate_station_payloads(self, payloads, coefficients, diagnostics=None):
        fallback_counts = diagnostics.setdefault("fallback_counts", {}) if diagnostics is not None else None
        return fisher_weighted_average(
            payloads,
            coefficients,
            fisher_eps=float(self.hparam.get("fisher_eps", 1e-8)),
            reference_state=clone_state_dict_to_cpu(self.model.state_dict()),
            fallback_counts=fallback_counts,
        )


class OTFusion(HierarchicalMergeServer):
    """Layer-wise OT/Hungarian alignment plus averaging, without RegMean."""

    def aggregate_station_payloads(self, payloads, coefficients, diagnostics=None):
        return self._merge_functional_layers(
            payloads,
            coefficients,
            diagnostics or {},
            use_regmean=False,
            regmean_kinds=set(),
            align=True,
            scope=self.hparam.get("ot_scope", "all"),
            reference=self.hparam.get("ot_reference", "first"),
            align_solver=self.hparam.get("ot_solver", self.hparam.get("align_solver", "hungarian")),
        )


class FedMAStyle(HierarchicalMergeServer):
    """Practical station-server adaptation of FedMA-style matched averaging.

    This is a practical station-server adaptation of FedMA-style matched
    averaging, not necessarily the exact original multi-round FedMA protocol.
    """

    def aggregate_station_payloads(self, payloads, coefficients, diagnostics=None):
        return self._merge_functional_layers(
            payloads,
            coefficients,
            diagnostics or {},
            use_regmean=False,
            regmean_kinds=set(),
            align=True,
            scope=self.hparam.get("fedma_scope", "all"),
            reference="first",
            align_solver=self.hparam.get("fedma_matching", "hungarian"),
        )


class ModelSoup(HierarchicalMergeServer):
    """Station model soup baseline. Greedy soup falls back to uniform without source validation."""

    def aggregate_station_payloads(self, payloads, coefficients, diagnostics=None):
        if self.hparam.get("model_soup_type", "uniform") == "greedy":
            warnings.warn("Greedy model soup needs source-validation station metrics; falling back to uniform soup.")
            if diagnostics is not None:
                diagnostics.setdefault("fallback_counts", {})["greedy_soup_no_source_val"] = 1
        return weighted_average_state_dicts(
            payloads,
            coefficients,
            reference_state=clone_state_dict_to_cpu(self.model.state_dict()),
            fallback_counts=diagnostics.setdefault("fallback_counts", {}) if diagnostics is not None else None,
        )


class FedRCHFLGaussian(HierarchicalMergeServer):
    """IROS 2024 FedRC HFL Gaussian-weighted aggregation.

    This implementation uses Gaussian summaries for both edge aggregation
    (vehicle/client -> station) and cloud aggregation (station -> server).
    It is distinct from the ICML 2024 robust-clustering FedRC.
    """

    collect_gaussian = True

    def _fedrc_weights(self, summaries, sample_counts):
        return fedrc_gaussian_weights(
            summaries,
            sample_counts,
            tau=float(self.hparam.get("fedrc_tau", 1.0)),
            distance=self.hparam.get("fedrc_distance", "bhattacharyya"),
            use_num_samples=bool(self.hparam.get("fedrc_use_num_samples", True)),
        )

    def _train_station_fedrc_payload(self, station, global_state):
        station_state = copy.deepcopy(global_state)
        selected_total_size = 0
        last_summaries = []
        last_weights = []
        stat_source = self.hparam.get("fedrc_stat_source", "rgb")
        max_batches = int(self.hparam.get("fedrc_max_batches", 1))
        for _ in range(self.station_rounds):
            sampled_client_indices = station.sample_clients(self.station_client_fraction, self.rng)
            station.update_clients(sampled_client_indices, station_state)
            summaries = []
            sample_counts = []
            for idx in sampled_client_indices:
                client = station.clients[idx]
                summaries.append(
                    client.compute_gaussian_summary(
                        station_state,
                        stat_source=stat_source,
                        max_batches=max_batches,
                    )
                )
                sample_counts.append(len(client))
            client_weights = self._fedrc_weights(summaries, sample_counts)
            selected_total_size = station.fit_clients(sampled_client_indices, self._round)
            station_state = station.aggregate_clients(sampled_client_indices, client_weights)
            last_summaries = summaries
            last_weights = client_weights

        station_gaussian = aggregate_gaussian_summaries(last_summaries, last_weights) if last_summaries else {}
        return StationPayload(
            station_id=int(station.station_id),
            state_dict=clone_state_dict_to_cpu(station_state),
            num_samples=int(selected_total_size or len(station)),
            client_indices=[int(idx) for idx in station.client_indices],
            gaussian=station_gaussian,
        )

    def station_coefficients(self, payloads):
        return self._fedrc_weights(
            [payload.gaussian for payload in payloads],
            [payload.num_samples for payload in payloads],
        )

    def train_federated_model(self):
        self._ensure_stations()
        sampled_station_indices = self.sample_stations()
        global_state = self.model.state_dict()
        t0 = time.time()
        payloads = [
            self._train_station_fedrc_payload(self.stations[station_id], global_state)
            for station_id in self._tqdm(sampled_station_indices, leave=False)
        ]
        coefficients = self.station_coefficients(payloads)
        diagnostics = {
            "server_method": self.__class__.__name__,
            "round": int(self._round),
            "station_ids": [payload.station_id for payload in payloads],
            "station_sizes": [payload.num_samples for payload in payloads],
            "coefficients": coefficients,
            "sketch_mb": 0.0,
            "fisher_mb": 0.0,
            "gaussian_summary_mb": estimate_tensor_dict_size_mb([payload.gaussian for payload in payloads]),
            "conv_layers_aligned": 0,
            "conv_layers_merged": 0,
            "linear_layers_merged": 0,
            "norm_layers_handled": 0,
            "attention_projections_handled": 0,
            "alignment_costs": [],
            "regmean_solve_failures": 0,
            "fallback_counts": {},
        }
        new_state = weighted_average_state_dicts(
            payloads,
            coefficients,
            reference_state=clone_state_dict_to_cpu(self.model.state_dict()),
            fallback_counts=diagnostics["fallback_counts"],
        )
        diagnostics["aggregation_latency_sec"] = time.time() - t0
        self.model.load_state_dict(new_state)
        self._log_aggregation_diagnostics(diagnostics)


class FedRCRobustClustering(HierarchicalFedAvg):
    """ICML 2024 robust-clustering FedRC.

    This is not the IROS HFL Gaussian FedRC. The robust clustering protocol is
    not implemented in this benchmark layer yet.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "FedRCRobustClustering (ICML 2024 robust clustering) is distinct from "
            "FedRCHFLGaussian and is not implemented yet."
        )


class MTGCApprox(HierarchicalFedAvg):
    """Documented approximation that tracks HFL drift but does not inject exact local corrections."""

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.mtgc_state = MTGCApproxState.empty()
        self._mtgc_warned = False

    def train_federated_model(self):
        if not self._mtgc_warned:
            warnings.warn(
                "MTGCApprox tracks group/global drift after HFL aggregation but does not "
                "inject exact MTGC control variables into every client optimizer step."
            )
            self._mtgc_warned = True
        before = clone_state_dict_to_cpu(self.model.state_dict())
        super().train_federated_model()
        after = clone_state_dict_to_cpu(self.model.state_dict())
        self.mtgc_state.group_controls[-1] = {
            key: after[key] - before[key]
            for key in before.keys()
            if is_floating_tensor(before[key]) and before[key].shape == after[key].shape
        }


class MTGC(HierarchicalFedAvg):
    """Exact MTGC port following the official NeurIPS 2024 implementation.

    Use with `client_method='MTGCClient'`. The implementation follows the
    official parameter-vector update structure: client correction `Z_i` is reset
    every global round and updated after each edge aggregation; group-global
    correction `Y_j` persists across global rounds and is updated after global
    aggregation.
    """

    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.mtgc_group_controls = {}

    def register_clients(self, clients):
        super().register_clients(clients)
        if not all(hasattr(client, "set_mtgc_correction_vector") for client in self.clients):
            raise RuntimeError("MTGC requires client_method='MTGCClient'")

    def _param_keys(self):
        return [name for name, _ in self.model.named_parameters()]

    def _state_param_vector(self, state_dict):
        return torch.cat([
            state_dict[key].detach().cpu().reshape(-1).float()
            for key in self._param_keys()
        ])

    def _zero_control(self):
        return torch.zeros_like(self._state_param_vector(self.model.state_dict()))

    def _client_n_minibatch(self, client):
        try:
            batches = len(client.dataloader)
        except TypeError:
            batches = int(np.ceil(len(client) / max(int(self.hparam.get("batch_size", 1)), 1)))
        return max(1, int(batches) * int(self.hparam.get("local_epochs", 1)))

    def _learning_rate(self):
        if "optimizer_config" in self.hparam and "lr" in self.hparam["optimizer_config"]:
            return float(self.hparam["optimizer_config"]["lr"])
        return float(self.hparam.get("lr", 1.0))

    def _average_client_states(self, client_indices, coefficients=None):
        if coefficients is None:
            coefficients = [1.0 / len(client_indices) for _ in client_indices]
        payloads = [
            StationPayload(
                station_id=-1,
                state_dict=clone_state_dict_to_cpu(self.clients[idx].model.state_dict()),
                num_samples=len(self.clients[idx]),
                client_indices=[int(idx)],
            )
            for idx in client_indices
        ]
        return weighted_average_state_dicts(
            payloads,
            coefficients,
            reference_state=clone_state_dict_to_cpu(self.model.state_dict()),
        )

    def _average_payload_states(self, payloads, coefficients=None):
        if coefficients is None:
            coefficients = [1.0 / len(payloads) for _ in payloads]
        return weighted_average_state_dicts(
            payloads,
            coefficients,
            reference_state=clone_state_dict_to_cpu(self.model.state_dict()),
        )

    def train_federated_model(self):
        if self.station_fraction != 1.0 or self.station_client_fraction != 1.0:
            raise RuntimeError(
                "Exact MTGC implementation follows the official full-participation "
                "HFL protocol. Set station_fraction=1.0 and station_client_fraction=1.0."
            )
        self._ensure_stations()
        lr = self._learning_rate()
        if lr <= 0:
            raise ValueError("MTGC requires a positive learning rate")

        param_dim = self._zero_control().numel()
        for station in self.stations:
            self.mtgc_group_controls.setdefault(
                station.station_id,
                torch.zeros(param_dim, dtype=torch.float32),
            )
        client_controls = {
            idx: torch.zeros(param_dim, dtype=torch.float32)
            for idx in range(self.num_clients)
        }
        global_state = clone_state_dict_to_cpu(self.model.state_dict())
        final_client_vectors = {}
        final_client_payloads = []

        for station in self._tqdm(self.stations, leave=False):
            station_state = copy.deepcopy(global_state)
            client_indices = list(station.client_indices)
            if not client_indices:
                continue
            for _ in range(self.station_rounds):
                round_vectors = {}
                for idx in client_indices:
                    client = self.clients[idx]
                    client.update_model(copy.deepcopy(station_state))
                    correction = client_controls[idx] + self.mtgc_group_controls[station.station_id]
                    client.set_mtgc_correction_vector(correction)
                    client.fit(self._round)
                    client_state = clone_state_dict_to_cpu(client.model.state_dict())
                    round_vectors[idx] = self._state_param_vector(client_state)

                coeffs = [1.0 / len(client_indices) for _ in client_indices]
                station_state = self._average_client_states(client_indices, coeffs)
                edge_vector = self._state_param_vector(station_state)
                for idx in client_indices:
                    scale = 1.0 / (self._client_n_minibatch(self.clients[idx]) * lr)
                    client_controls[idx] = client_controls[idx] + scale * (round_vectors[idx] - edge_vector)

            for idx in client_indices:
                client_state = clone_state_dict_to_cpu(self.clients[idx].model.state_dict())
                final_client_vectors[idx] = self._state_param_vector(client_state)
                final_client_payloads.append(
                    StationPayload(
                        station_id=station.station_id,
                        state_dict=client_state,
                        num_samples=len(self.clients[idx]),
                        client_indices=[int(idx)],
                    )
                )

        if not final_client_payloads:
            raise RuntimeError("MTGC did not train any clients")
        global_coeffs = [1.0 / len(final_client_payloads) for _ in final_client_payloads]
        new_state = self._average_payload_states(final_client_payloads, global_coeffs)
        global_vector = self._state_param_vector(new_state)

        for station in self.stations:
            client_indices = list(station.client_indices)
            if not client_indices:
                continue
            edge_vector = torch.stack([final_client_vectors[idx] for idx in client_indices], dim=0).mean(dim=0)
            n_minibatch = np.mean([self._client_n_minibatch(self.clients[idx]) for idx in client_indices])
            scale = 1.0 / (max(float(n_minibatch), 1.0) * lr * max(int(self.station_rounds), 1))
            self.mtgc_group_controls[station.station_id] = (
                self.mtgc_group_controls[station.station_id] + scale * (edge_vector - global_vector)
            )

        self.model.load_state_dict(new_state)


OTFedMAStyle = FedMAStyle
FedRC = FedRCHFLGaussian


class FedDG(FedAvg):
    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier))
            client.set_amploader(self.amploader)
        super().register_clients(clients)
            
    def set_amploader(self, amp_dataset):
        self.amploader = amp_dataset


class FedADGServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.gen_input_size = int(hparam['hparam5'])

    def setup_model(self, model_file, start_epoch):
        """
        The model setup depends on the datasets. 
        """
        assert self._round == 0
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self._generator = GeneDistrNet(num_labels=self.ds_bundle.n_classes, input_size=self.gen_input_size, hidden_size=self._featurizer.n_outputs)
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self.generator = nn.DataParallel(self._generator)
        self.model = nn.DataParallel(nn.Sequential(self._featurizer, self._classifier))
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
            self._round = int(start_epoch)

    def register_clients(self, clients):
        # assert self._round == 0
        self.clients = clients
        self.num_clients = len(self.clients)
        for client in self.clients:
            client.setup_model(copy.deepcopy(self._featurizer), copy.deepcopy(self._classifier), copy.deepcopy(self._generator))

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in self._tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict(), self._generator.state_dict())

            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            logging.debug(message)
            del message
        else:
            # send the global model to selected clients
            for idx in self._tqdm(sampled_client_indices, leave=False):
                self.clients[idx].update_model(self.model.state_dict(), self._generator.state_dict())
            message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            logging.debug(message)
            del message

    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        logging.debug(message)
        del message

        averaged_weights = OrderedDict()
        averaged_generator_weights = OrderedDict()
        for it, idx in self._tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            local_generator_weights = self.clients[idx].generator.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]                 
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]         
            for key in self.generator.state_dict().keys():
                if it == 0:
                    averaged_generator_weights[key] = coefficients[it] * local_generator_weights[key]
                    
                else:
                    averaged_generator_weights[key] += coefficients[it] * local_generator_weights[key]
        self.model.load_state_dict(averaged_weights)
        self.generator.load_state_dict(averaged_generator_weights)


class FedGMA(FedAvg):
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        num_sampled_clients = len(sampled_client_indices)
        delta = []
        sign_delta = ParamDict()
        self.model.to('cpu')
        last_weights = ParamDict(self.model.state_dict())
        for it, idx in self._tqdm(enumerate(sampled_client_indices), leave=False):
            self.clients[idx].model.to('cpu')
            local_weights = ParamDict(self.clients[idx].model.state_dict())
            delta.append(coefficients[it] * (local_weights - last_weights))
            if it == 0:
                sum_delta = delta[it]
                sign_delta = delta[it].sign()
            else:
                sum_delta += delta[it]
                sign_delta += delta[it].sign()
                # if it == 0:
                #     averaged_weights[key] = coefficients[it] * local_weights[key]
                # else:
                #     averaged_weights[key] += coefficients[it] * local_weights[key]
        sign_delta /= num_sampled_clients
        abs_sign_delta = sign_delta.abs()
        # print(sign_delta[key])
        mask = abs_sign_delta.ge(self.hparam['hparam1'])
        # print("--mid--")
        # print(mask)
        # print("-------")
        final_mask = mask + (0-mask) * abs_sign_delta
        averaged_weights = last_weights + self.hparam['hparam1'] * final_mask * sum_delta 
        self.model.load_state_dict(averaged_weights)



class ScaffoldServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.c = None

    def transmit_model(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in self._tqdm(self.clients, leave=False):
            # for client in self.clients:
                client.update_model(self.model.state_dict())
                client.c_global = copy.deepcopy(self.c)
        else:
            # send the global model to selected clients
            for idx in self._tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_model(self.model.state_dict())
                self.clients[idx].c_global = copy.deepcopy(self.c)
    
    def aggregate(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in self._tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            if it == 0:
                c_local = self.clients[idx].c_local
            else:
                c_local += self.clients[idx].c_local
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
    
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.c = c_local / len(sampled_client_indices)
        self.model.load_state_dict(averaged_weights)


class AFLServer(FedAvg):
    def __init__(self, device, ds_bundle, hparam):
        super().__init__(device, ds_bundle, hparam)
        self.group_weights = torch.zeros(self.ds_bundle.grouper.n_groups)
        train_set = self.ds_bundle.dataset.get_subset('train', transform=self.ds_bundle.train_transform)
        train_g = self.ds_bundle.grouper.metadata_to_group(train_set.metadata_array)
        unique_groups, unique_counts = torch.unique(train_g, sorted=False, return_counts=True)
        counts = torch.zeros(self.ds_bundle.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.is_group_in_train = is_group_in_train
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
       

    def transmit_lambda(self, sampled_client_indices=None):
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            for client in self._tqdm(self.clients, leave=False):
            # for client in self.clients:
            
                client.update_vector(self.group_weights)
        else:
            # send the global model to selected clients
            for idx in self._tqdm(sampled_client_indices, leave=False):
            # for idx in sampled_client_indices:
                self.clients[idx].update_vector(self.group_weights)

    def aggregate(self,sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        for it, idx in self._tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)

    def update_lambda(self, sampled_client_indices):
        self.transmit_model(sampled_client_indices)
        total_loss_per_domain = torch.zeros_like(self.group_weights)
        total_samples_per_domain = torch.zeros_like(self.group_weights)
        # for client in tqdm(self.clients, leave=False):
        # # for client in self.clients:
        #     loss_per_domain, samples_per_domain = client.gradient_lambda()
        #     total_loss_per_domain += loss_per_domain
        #     total_samples_per_domain += samples_per_domain

        # send the global model to selected clients
        for idx in self._tqdm(sampled_client_indices, leave=False):
        # for idx in sampled_client_indices:
            loss_per_domain, samples_per_domain = self.clients[idx].gradient_lambda()
            total_loss_per_domain += loss_per_domain
            total_samples_per_domain += samples_per_domain
        self.group_weights += torch.nan_to_num(self.hparam['hparam1'] * total_loss_per_domain / total_samples_per_domain, nan=0.0)

        print(self.group_weights)

        self.group_weights = euclidean_proj_simplex(self.group_weights)

                
        print("after proj")
        print(self.group_weights)
        # print(self.group_weights)
        wandb.log({"l0_lmda": torch.count_nonzero(self.group_weights[self.group_weights>0.001])} ,step=self._round*self.hparam['local_epochs'])

    def train_federated_model(self):
        """Do federated training."""
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        self.transmit_lambda(sampled_client_indices)

        # updated selected clients with local dataset
        selected_total_size = self.update_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        # self.evaluate_clients(sampled_client_indices)

        # average each updated model parameters of the selected clients and update the global model
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        self.aggregate(sampled_client_indices, mixing_coefficients)

        self.update_lambda(sampled_client_indices)
