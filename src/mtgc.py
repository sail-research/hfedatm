from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class MTGCApproxState:
    """Small state holder for the documented MTGC approximation.

    Exact MTGC needs client-to-group and group-to-global correction terms inside
    every local optimizer step. The current benchmark clients expose only
    `fit(server_round)`, so the exact algorithm is intentionally not hidden
    behind this helper.
    """

    client_controls: Dict[int, Dict[str, torch.Tensor]]
    group_controls: Dict[int, Dict[str, torch.Tensor]]

    @classmethod
    def empty(cls):
        return cls(client_controls={}, group_controls={})
