import torch

from src.merging import regmean_solve


def test_regmean_matches_closed_form_full():
    W1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    W2 = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
    X1 = torch.tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    X2 = torch.tensor([[0.5, 1.0], [1.0, 1.5]])
    G1 = X1.T @ X1
    G2 = X2.T @ X2
    coeffs = [0.4, 0.6]
    ridge = 1e-4

    merged, ok = regmean_solve([W1, W2], [G1, G2], coeffs, ridge=ridge)
    G_sum = coeffs[0] * G1 + coeffs[1] * G2 + ridge * torch.eye(2)
    B_sum = coeffs[0] * W1 @ G1 + coeffs[1] * W2 @ G2
    expected = torch.linalg.solve(G_sum.T, B_sum.T).T

    assert ok
    assert torch.allclose(merged, expected, atol=1e-5)
    assert torch.isfinite(merged).all()


def test_regmean_diag_mode_shape_and_value():
    W1 = torch.tensor([[1.0, 3.0]])
    W2 = torch.tensor([[5.0, 7.0]])
    G1 = torch.tensor([2.0, 4.0])
    G2 = torch.tensor([6.0, 8.0])
    merged, ok = regmean_solve([W1, W2], [G1, G2], [0.5, 0.5], ridge=0.0)
    expected = torch.tensor([[(0.5 * 2 * 1 + 0.5 * 6 * 5) / 4, (0.5 * 4 * 3 + 0.5 * 8 * 7) / 6]])
    assert ok
    assert merged.shape == W1.shape
    assert torch.allclose(merged, expected)
