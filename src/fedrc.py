from typing import Any, Dict, List, Sequence

import torch


def aggregate_gaussian_summaries(summaries: Sequence[Dict[str, Any]], weights: Sequence[float]) -> Dict[str, Any]:
    valid = [
        (summary, float(weight))
        for summary, weight in zip(summaries, weights)
        if summary and int(summary.get("count", 0)) > 0
    ]
    if not valid:
        return {}
    total_weight = sum(weight for _, weight in valid)
    if total_weight <= 0:
        total_weight = float(len(valid))
        valid = [(summary, 1.0) for summary, _ in valid]

    mean = None
    second = None
    count = 0
    for summary, weight in valid:
        mu = summary["mean"].detach().cpu().float()
        var = summary["var"].detach().cpu().float().clamp_min(0)
        coeff = weight / total_weight
        mean = coeff * mu if mean is None else mean + coeff * mu
        second_term = coeff * (var + mu * mu)
        second = second_term if second is None else second + second_term
        count += int(summary.get("count", 0))
    var = (second - mean * mean).clamp_min(0)
    return {
        "count": count,
        "mean": mean,
        "var": var,
        "source": valid[0][0].get("source", "unknown"),
    }


def gaussian_distance(a: Dict[str, Any], b: Dict[str, Any], metric: str = "diag_w2") -> float:
    if not a or not b:
        return 0.0
    mu_a = a["mean"].detach().cpu().float()
    mu_b = b["mean"].detach().cpu().float()
    var_a = a["var"].detach().cpu().float().clamp_min(1e-12)
    var_b = b["var"].detach().cpu().float().clamp_min(1e-12)
    metric = metric.lower()
    if metric == "diag_w2":
        d = (mu_a - mu_b).pow(2).sum() + (var_a.sqrt() - var_b.sqrt()).pow(2).sum()
    elif metric == "euclidean_mean":
        d = (mu_a - mu_b).pow(2).sum()
    elif metric == "kl_sym":
        kl_ab = 0.5 * ((var_a / var_b) + (mu_b - mu_a).pow(2) / var_b - 1 + (var_b.log() - var_a.log())).sum()
        kl_ba = 0.5 * ((var_b / var_a) + (mu_a - mu_b).pow(2) / var_a - 1 + (var_a.log() - var_b.log())).sum()
        d = 0.5 * (kl_ab + kl_ba)
    elif metric == "bhattacharyya":
        var_mid = 0.5 * (var_a + var_b)
        d = 0.125 * ((mu_a - mu_b).pow(2) / var_mid).sum() + 0.5 * (var_mid.log() - 0.5 * (var_a.log() + var_b.log())).sum()
    else:
        raise ValueError(f"Unsupported FedRC Gaussian distance '{metric}'")
    return float(torch.nan_to_num(d, nan=0.0, posinf=1e12, neginf=0.0).item())


def fedrc_gaussian_weights(
    summaries: Sequence[Dict[str, Any]],
    sample_counts: Sequence[int],
    tau: float = 1.0,
    distance: str = "diag_w2",
    use_num_samples: bool = True,
) -> List[float]:
    if not summaries:
        return []
    base_counts = [float(max(int(count), 0)) for count in sample_counts]
    if sum(base_counts) <= 0:
        base_counts = [1.0 for _ in summaries]
    barycenter_weights = [count / sum(base_counts) for count in base_counts]
    barycenter = aggregate_gaussian_summaries(summaries, barycenter_weights)
    tau = max(float(tau), 1e-12)
    raw = []
    for summary, count in zip(summaries, base_counts):
        d = gaussian_distance(summary, barycenter, metric=distance)
        prior = count if use_num_samples else 1.0
        raw.append(prior * torch.exp(torch.tensor(-d / tau)).item())
    total = float(sum(raw))
    if total <= 0:
        return [1.0 / len(raw) for _ in raw]
    return [float(weight) / total for weight in raw]
