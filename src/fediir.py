from typing import Iterable, Sequence, Tuple

import torch


GradientTuple = Tuple[torch.Tensor, ...]


def zeros_like_parameters(parameters: Iterable[torch.nn.Parameter], device="cpu") -> GradientTuple:
    return tuple(torch.zeros_like(param.detach(), device=device) for param in parameters)


def add_gradient_tuples(left: GradientTuple, right: GradientTuple) -> GradientTuple:
    if len(left) != len(right):
        raise ValueError("Gradient tuples must have the same length")
    return tuple(a + b for a, b in zip(left, right))


def scale_gradient_tuple(grads: GradientTuple, scale: float) -> GradientTuple:
    return tuple(float(scale) * grad for grad in grads)


def aggregate_gradient_sums(
    gradient_sums: Sequence[GradientTuple],
    batch_counts: Sequence[int],
) -> GradientTuple:
    if not gradient_sums:
        raise ValueError("FedIIR requires at least one client gradient sum")
    if len(gradient_sums) != len(batch_counts):
        raise ValueError("gradient_sums and batch_counts must have the same length")
    total_batches = int(sum(batch_counts))
    if total_batches <= 0:
        raise ValueError("FedIIR mean gradient requires at least one batch")
    out = tuple(torch.zeros_like(grad.detach().cpu()) for grad in gradient_sums[0])
    for grad_tuple in gradient_sums:
        out = add_gradient_tuples(out, tuple(grad.detach().cpu() for grad in grad_tuple))
    return tuple(grad / total_batches for grad in out)


def ema_gradient_mean(previous: GradientTuple, current: GradientTuple, ema: float) -> GradientTuple:
    if len(previous) != len(current):
        raise ValueError("FedIIR EMA tuples must have the same length")
    ema = float(ema)
    return tuple(ema * old.detach().cpu() + (1.0 - ema) * new.detach().cpu() for old, new in zip(previous, current))


def fediir_penalty(classifier_grads: GradientTuple, mean_grads: GradientTuple) -> torch.Tensor:
    if len(classifier_grads) != len(mean_grads):
        raise ValueError("FedIIR gradient tuples must have the same length")
    penalty = None
    for grad_client, grad_mean in zip(classifier_grads, mean_grads):
        term = (grad_client - grad_mean.to(grad_client.device, dtype=grad_client.dtype)).pow(2).sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        raise ValueError("FedIIR penalty requires at least one classifier gradient")
    return penalty


def default_fediir_penalty(dataset_name: str) -> float:
    name = str(dataset_name).lower()
    if name == "rotatedmnist":
        return 1e-2
    if name == "vlcs":
        return 5e-3
    if name == "pacs":
        return 1e-3
    if name == "officehome":
        return 5e-4
    return 1e-3
