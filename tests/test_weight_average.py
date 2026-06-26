from collections import OrderedDict

import torch

from src.merging import StationPayload, weighted_average_state_dicts


def test_weighted_average_state_dicts_float_and_buffer():
    p1 = StationPayload(
        station_id=0,
        state_dict=OrderedDict(
            weight=torch.tensor([1.0, 3.0]),
            num_batches_tracked=torch.tensor(2, dtype=torch.long),
        ),
        num_samples=1,
        client_indices=[0],
    )
    p2 = StationPayload(
        station_id=1,
        state_dict=OrderedDict(
            weight=torch.tensor([3.0, 5.0]),
            num_batches_tracked=torch.tensor(5, dtype=torch.long),
        ),
        num_samples=3,
        client_indices=[1],
    )
    out = weighted_average_state_dicts([p1, p2], [0.25, 0.75])
    assert torch.allclose(out["weight"], torch.tensor([2.5, 4.5]))
    assert out["num_batches_tracked"].item() == 5
