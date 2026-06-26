from collections import OrderedDict

import torch

from src.merging import StationPayload, fisher_weighted_average


def test_fisher_weighted_average_formula():
    p1 = StationPayload(
        station_id=0,
        state_dict=OrderedDict(w=torch.tensor([1.0, 3.0])),
        num_samples=1,
        client_indices=[0],
        fisher={"w": torch.tensor([2.0, 1.0])},
    )
    p2 = StationPayload(
        station_id=1,
        state_dict=OrderedDict(w=torch.tensor([5.0, 7.0])),
        num_samples=1,
        client_indices=[1],
        fisher={"w": torch.tensor([1.0, 3.0])},
    )
    out = fisher_weighted_average([p1, p2], [0.5, 0.5], fisher_eps=0.0)
    expected = torch.tensor([
        (0.5 * 2 * 1 + 0.5 * 1 * 5) / (0.5 * 2 + 0.5 * 1),
        (0.5 * 1 * 3 + 0.5 * 3 * 7) / (0.5 * 1 + 0.5 * 3),
    ])
    assert torch.allclose(out["w"], expected)
