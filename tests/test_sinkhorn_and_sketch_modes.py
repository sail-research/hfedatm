import torch
import torch.nn as nn

from src.merging import HungarianAligner
from src.sketches import ActivationSketcher, SketchConfig


def test_sinkhorn_alignment_builds_valid_transport_and_permutation():
    ref = torch.eye(3)
    target = ref[[2, 0, 1]]
    aligner = HungarianAligner("sinkhorn", reg=0.05, iters=50)
    perm, cost = aligner.align(ref, target)
    plan = aligner.last_transport
    assert sorted(perm.tolist()) == [0, 1, 2]
    assert cost >= 0
    assert plan.shape == (3, 3)
    assert torch.isclose(plan.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(plan >= 0)


def test_lowrank_activation_sketch_materializes_gram_without_raw_data():
    model = nn.Sequential(nn.Linear(5, 3))
    cfg = SketchConfig(mode="lowrank", lowrank_rank=3, max_full_dim=16, random_seed=123)
    with ActivationSketcher(model, cfg) as sketcher:
        _ = model(torch.randn(7, 5))
    sketch = sketcher.payload()["0"]
    assert sketch["mode"] == "lowrank"
    assert sketch["C"].shape == (5, 3)
    assert sketch["W"].shape == (3, 3)
    assert sketch["G"].shape == (5, 5)
    assert "activations" not in sketch and "X" not in sketch and "batch" not in sketch


def test_random_projection_activation_sketch_materializes_gram_without_raw_data():
    model = nn.Sequential(nn.Linear(6, 2))
    cfg = SketchConfig(mode="random_projection", random_projection_dim=4, max_full_dim=16, random_seed=123)
    with ActivationSketcher(model, cfg) as sketcher:
        _ = model(torch.randn(8, 6))
    sketch = sketcher.payload()["0"]
    assert sketch["mode"] == "random_projection"
    assert sketch["G_projected"].shape == (4, 4)
    assert sketch["projection"].shape == (6, 4)
    assert sketch["G"].shape == (6, 6)
    assert "raw" not in sketch and "X" not in sketch and "batch" not in sketch
