import copy
import json
import math
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class StationPayload:
    station_id: int
    state_dict: OrderedDict
    num_samples: int
    client_indices: List[int]
    sketches: Dict[str, Any] = field(default_factory=dict)
    fisher: Dict[str, torch.Tensor] = field(default_factory=dict)
    gaussian: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


def is_floating_tensor(t: Any) -> bool:
    return torch.is_tensor(t) and torch.is_floating_point(t)


def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    return OrderedDict(
        (key, value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value))
        for key, value in state_dict.items()
    )


def strip_module_prefix(key: str) -> str:
    return key[len("module."):] if key.startswith("module.") else key


def add_module_prefix(key: str) -> str:
    return key if key.startswith("module.") else "module." + key


def state_prefix_for_module_name(module_name: str) -> str:
    return "module" if module_name == "" else "module." + module_name


def safe_average_buffer(values: Sequence[torch.Tensor], coefficients: Sequence[float], key: str, reference=None):
    ref = values[0] if reference is None else reference
    if key.endswith("num_batches_tracked"):
        stacked = torch.stack([v.detach().cpu().to(torch.long) for v in values])
        return torch.max(stacked, dim=0).values.to(dtype=ref.dtype)
    return ref.detach().cpu().clone() if torch.is_tensor(ref) else copy.deepcopy(ref)


def _same_shape(values: Sequence[torch.Tensor]) -> bool:
    return all(torch.is_tensor(v) and v.shape == values[0].shape for v in values)


def weighted_average_state_dicts(
    payloads: Sequence[StationPayload],
    coefficients: Sequence[float],
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    fallback_counts: Optional[Dict[str, int]] = None,
) -> OrderedDict:
    if not payloads:
        raise ValueError("weighted_average_state_dicts requires at least one payload")
    if len(payloads) != len(coefficients):
        raise ValueError("payloads and coefficients must have the same length")
    reference_state = reference_state or payloads[0].state_dict
    out = OrderedDict()
    keys = list(reference_state.keys())
    for key in keys:
        values = [p.state_dict.get(key) for p in payloads]
        if any(v is None for v in values):
            if fallback_counts is not None:
                fallback_counts["missing_key"] = fallback_counts.get("missing_key", 0) + 1
            out[key] = reference_state[key].detach().cpu().clone()
            continue
        if all(is_floating_tensor(v) for v in values) and _same_shape(values):
            merged = None
            for coeff, value in zip(coefficients, values):
                term = float(coeff) * value.detach().cpu()
                merged = term if merged is None else merged + term
            out[key] = merged.to(dtype=reference_state[key].dtype)
        elif all(torch.is_tensor(v) for v in values) and _same_shape(values):
            out[key] = safe_average_buffer(values, coefficients, key, reference_state[key])
        else:
            if fallback_counts is not None:
                fallback_counts["shape_or_type_mismatch"] = fallback_counts.get("shape_or_type_mismatch", 0) + 1
            out[key] = reference_state[key].detach().cpu().clone() if torch.is_tensor(reference_state[key]) else copy.deepcopy(reference_state[key])
    return out


def estimate_tensor_dict_size_mb(obj: Any) -> float:
    total = 0
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size() / (1024 ** 2)
    if isinstance(obj, dict):
        for value in obj.values():
            total += estimate_tensor_dict_size_mb(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            total += estimate_tensor_dict_size_mb(value)
    return float(total)


def normalize_coefficients(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        return [1.0 / len(weights) for _ in weights]
    return [float(w) / total for w in weights]


def shrink_gram(G: torch.Tensor, alpha: float) -> torch.Tensor:
    if G.dim() == 1:
        return G
    diag = torch.diag(torch.diag(G))
    return float(alpha) * G + (1.0 - float(alpha)) * diag


def regmean_solve(
    weights: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
    coefficients: Sequence[float],
    ridge: float = 1e-4,
    fallback: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, bool]:
    """Closed-form RegMean for Linear weights or flattened Conv2d weights.

    `weights` must be [out_dim, in_dim]. `grams` may be full [in_dim, in_dim]
    or diagonal [in_dim].
    """
    if not weights:
        raise ValueError("regmean_solve requires weights")
    ref = weights[0].detach().cpu().float()
    if fallback is None:
        fallback = sum(float(c) * w.detach().cpu().float() for c, w in zip(coefficients, weights))
    if any(w.shape != ref.shape for w in weights):
        return fallback.to(dtype=weights[0].dtype), False
    in_dim = ref.shape[1]
    if any(g is None for g in grams):
        return fallback.to(dtype=weights[0].dtype), False

    grams_cpu = [g.detach().cpu().float() for g in grams]
    if all(g.dim() == 1 for g in grams_cpu):
        denom = torch.zeros(in_dim, dtype=torch.float32)
        numer = torch.zeros_like(ref)
        for coeff, W, Gd in zip(coefficients, weights, grams_cpu):
            if Gd.numel() != in_dim:
                return fallback.to(dtype=weights[0].dtype), False
            d = float(coeff) * Gd
            numer += W.detach().cpu().float() * d.unsqueeze(0)
            denom += d
        merged = numer / (denom + float(ridge)).clamp_min(float(ridge))
        return merged.to(dtype=weights[0].dtype), True

    if any(g.dim() != 2 or g.shape[0] != in_dim or g.shape[1] != in_dim for g in grams_cpu):
        return fallback.to(dtype=weights[0].dtype), False

    G_sum = torch.zeros((in_dim, in_dim), dtype=torch.float32)
    B_sum = torch.zeros_like(ref)
    for coeff, W, G in zip(coefficients, weights, grams_cpu):
        c = float(coeff)
        G_sum += c * G
        B_sum += c * W.detach().cpu().float().matmul(G)
    eye = torch.eye(in_dim, dtype=torch.float32)
    reg = max(float(ridge), 0.0)
    for multiplier in [1.0, 10.0, 100.0, 1000.0]:
        try:
            sol = torch.linalg.solve((G_sum + reg * multiplier * eye).T, B_sum.T).T
            if torch.isfinite(sol).all():
                return sol.to(dtype=weights[0].dtype), True
        except RuntimeError:
            continue
    try:
        sol = B_sum.matmul(torch.linalg.pinv(G_sum + max(reg, 1e-8) * eye))
        if torch.isfinite(sol).all():
            warnings.warn("RegMean solve used pseudo-inverse fallback", RuntimeWarning)
            return sol.to(dtype=weights[0].dtype), True
    except RuntimeError:
        pass
    return fallback.to(dtype=weights[0].dtype), False


def regmean_merge_parameter(
    weights: Sequence[torch.Tensor],
    grams: Sequence[torch.Tensor],
    coefficients: Sequence[float],
    ridge: float,
) -> Tuple[torch.Tensor, bool]:
    ref = weights[0]
    if ref.dim() == 2:
        return regmean_solve(weights, grams, coefficients, ridge=ridge)
    if ref.dim() == 4:
        flat_weights = [w.detach().cpu().reshape(w.shape[0], -1) for w in weights]
        merged, ok = regmean_solve(flat_weights, grams, coefficients, ridge=ridge)
        return merged.reshape_as(ref).to(dtype=ref.dtype), ok
    return sum(float(c) * w.detach().cpu() for c, w in zip(coefficients, weights)).to(dtype=ref.dtype), False


def fisher_weighted_average(
    payloads: Sequence[StationPayload],
    coefficients: Sequence[float],
    fisher_eps: float = 1e-8,
    reference_state: Optional[Dict[str, torch.Tensor]] = None,
    fallback_counts: Optional[Dict[str, int]] = None,
) -> OrderedDict:
    reference_state = reference_state or payloads[0].state_dict
    avg = weighted_average_state_dicts(payloads, coefficients, reference_state, fallback_counts)
    out = OrderedDict()
    for key, ref in reference_state.items():
        values = [p.state_dict.get(key) for p in payloads]
        fishers = [p.fisher.get(key) for p in payloads]
        if (
            all(is_floating_tensor(v) for v in values)
            and all(is_floating_tensor(f) for f in fishers)
            and _same_shape(values)
            and _same_shape(fishers)
            and values[0].shape == fishers[0].shape
        ):
            numer = torch.zeros_like(values[0].detach().cpu().float())
            denom = torch.zeros_like(numer)
            for coeff, theta, fish in zip(coefficients, values, fishers):
                scale = float(coeff) * fish.detach().cpu().float()
                numer += scale * theta.detach().cpu().float()
                denom += scale
            out[key] = (numer / (denom + float(fisher_eps))).to(dtype=ref.dtype)
        else:
            if fallback_counts is not None and is_floating_tensor(ref):
                fallback_counts["missing_fisher"] = fallback_counts.get("missing_fisher", 0) + 1
            out[key] = avg[key]
    return out


class UnitSignatureBuilder:
    @staticmethod
    def output_signatures_from_weight(weight: torch.Tensor) -> torch.Tensor:
        W = weight.detach().cpu().float()
        if W.dim() == 4:
            sig = W.reshape(W.shape[0], -1)
        elif W.dim() == 2:
            sig = W
        else:
            raise ValueError("Only Conv2d/Linear weights have output unit signatures")
        return torch.nn.functional.normalize(sig, p=2, dim=1, eps=1e-12)


class HungarianAligner:
    def __init__(self, solver: str = "hungarian", reg: float = 0.05, iters: int = 25):
        self.solver = solver
        self.reg = float(reg)
        self.iters = int(iters)
        self.last_transport = None

    @staticmethod
    def sinkhorn_transport(cost: torch.Tensor, reg: float = 0.05, iters: int = 25) -> torch.Tensor:
        cost = cost.detach().cpu().float()
        n, m = cost.shape
        if n == 0 or m == 0:
            return torch.zeros_like(cost)
        a = torch.full((n,), 1.0 / n, dtype=cost.dtype)
        b = torch.full((m,), 1.0 / m, dtype=cost.dtype)
        reg = max(float(reg), 1e-8)
        K = torch.exp(-cost / reg).clamp_min(1e-30)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(max(int(iters), 1)):
            u = a / (K.matmul(v).clamp_min(1e-30))
            v = b / (K.T.matmul(u).clamp_min(1e-30))
        plan = u[:, None] * K * v[None, :]
        return plan / plan.sum().clamp_min(1e-30)

    def align(self, reference_signatures: torch.Tensor, target_signatures: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if self.solver == "none":
            n = min(reference_signatures.shape[0], target_signatures.shape[0])
            return torch.arange(n), 0.0
        ref = reference_signatures.detach().cpu().float()
        tgt = target_signatures.detach().cpu().float()
        n = min(ref.shape[0], tgt.shape[0])
        ref = ref[:n]
        tgt = tgt[:n]
        cost_tensor = torch.cdist(ref, tgt, p=2)
        cost = cost_tensor.numpy()
        if self.solver == "sinkhorn":
            plan = self.sinkhorn_transport(cost_tensor, reg=self.reg, iters=self.iters)
            self.last_transport = plan
            score = (-plan).numpy()
            try:
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(score)
                order = np.argsort(rows)
                perm = cols[order].tolist()
            except Exception:
                perm = []
                unused = set(range(n))
                plan_np = plan.numpy()
                for i in range(n):
                    j = max(unused, key=lambda col: plan_np[i, col])
                    perm.append(j)
                    unused.remove(j)
        elif self.solver == "greedy":
            perm = []
            unused = set(range(n))
            for i in range(n):
                j = min(unused, key=lambda col: cost[i, col])
                perm.append(j)
                unused.remove(j)
        else:
            try:
                from scipy.optimize import linear_sum_assignment

                rows, cols = linear_sum_assignment(cost)
                order = np.argsort(rows)
                perm = cols[order].tolist()
            except Exception:
                perm = []
                unused = set(range(n))
                for i in range(n):
                    j = min(unused, key=lambda col: cost[i, col])
                    perm.append(j)
                    unused.remove(j)
        mean_cost = float(cost[np.arange(n), np.asarray(perm)].mean()) if n else 0.0
        return torch.tensor(perm, dtype=torch.long), mean_cost


class GraphConsistentAligner(HungarianAligner):
    """Practical graph-consistent aligner.

    This implementation computes stable layer-wise permutations and exposes the
    same interface as HungarianAligner. Full residual/attention graph tracing is
    intentionally conservative and handled by safe server fallbacks.
    """

    def align_module_weight(self, reference_weight: torch.Tensor, target_weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
        return self.align(
            UnitSignatureBuilder.output_signatures_from_weight(reference_weight),
            UnitSignatureBuilder.output_signatures_from_weight(target_weight),
        )


def apply_output_permutation_to_weight_bias(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    permutation: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    perm = permutation.to(dtype=torch.long)
    out_weight = weight.detach().cpu().index_select(0, perm)
    out_bias = bias.detach().cpu().index_select(0, perm) if bias is not None else None
    return out_weight, out_bias


def json_dumps_safe(obj: Dict[str, Any]) -> str:
    def default(value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    return json.dumps(obj, default=default, sort_keys=True)
