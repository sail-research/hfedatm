import torch
import torch.nn as nn

from src.sketches import ActivationSketcher, SketchConfig


def test_activation_sketch_payload_has_no_raw_activations():
    model = nn.Sequential(nn.Linear(4, 2))
    cfg = SketchConfig(mode="diag", max_batches=1)
    with ActivationSketcher(model, cfg) as sketcher:
        _ = model(torch.randn(3, 4))
    payload = sketcher.payload()
    forbidden = {"x", "X", "activation", "activations", "raw", "batch"}
    for sketch in payload.values():
        assert set(sketch.keys()).isdisjoint(forbidden)
        assert {"G", "count", "mode", "kind", "dim"}.issubset(set(sketch.keys()))
