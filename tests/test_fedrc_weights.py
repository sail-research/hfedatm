import torch

from src.fedrc import fedrc_gaussian_weights


def test_fedrc_gaussian_weights_sum_to_one():
    summaries = [
        {"count": 10, "mean": torch.tensor([0.0, 0.0]), "var": torch.tensor([1.0, 1.0])},
        {"count": 20, "mean": torch.tensor([1.0, 0.0]), "var": torch.tensor([1.5, 1.0])},
        {"count": 30, "mean": torch.tensor([2.0, 0.0]), "var": torch.tensor([2.0, 1.0])},
    ]
    weights = fedrc_gaussian_weights(summaries, [10, 20, 30], tau=1.0)
    assert len(weights) == 3
    assert all(weight >= 0 for weight in weights)
    assert abs(sum(weights) - 1.0) < 1e-6
