import torch
import torch.nn as nn

from src.sketches import ActivationSketcher, SketchConfig


def test_conv2d_sketch_uses_unfold_shape():
    model = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False))
    x = torch.randn(4, 2, 5, 5)
    cfg = SketchConfig(mode="full", max_batches=1, max_patches=1000)
    with ActivationSketcher(model, cfg) as sketcher:
        _ = model(x)
    payload = sketcher.payload()
    assert "0" in payload
    assert payload["0"]["G"].shape == (2 * 3 * 3, 2 * 3 * 3)
    assert payload["0"]["count"] == 4 * 5 * 5
