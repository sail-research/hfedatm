import torch

from src.fediir import aggregate_gradient_sums, ema_gradient_mean, fediir_penalty


def test_fediir_mean_gradient_and_ema_match_official_formula():
    g1 = (torch.tensor([2.0, 4.0]), torch.tensor([[1.0]]))
    g2 = (torch.tensor([6.0, 8.0]), torch.tensor([[3.0]]))
    current = aggregate_gradient_sums([g1, g2], [1, 1])
    assert torch.allclose(current[0], torch.tensor([4.0, 6.0]))
    assert torch.allclose(current[1], torch.tensor([[2.0]]))

    previous = (torch.zeros(2), torch.zeros(1, 1))
    smoothed = ema_gradient_mean(previous, current, ema=0.95)
    assert torch.allclose(smoothed[0], torch.tensor([0.2, 0.3]))
    assert torch.allclose(smoothed[1], torch.tensor([[0.1]]))


def test_fediir_penalty_is_squared_distance_to_mean_gradient():
    client_grads = (torch.tensor([1.0, 3.0]), torch.tensor([[2.0]]))
    mean_grads = (torch.tensor([0.0, 1.0]), torch.tensor([[5.0]]))
    penalty = fediir_penalty(client_grads, mean_grads)
    assert torch.allclose(penalty, torch.tensor(14.0))
