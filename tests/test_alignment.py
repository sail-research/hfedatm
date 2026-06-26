import torch

from src.merging import HungarianAligner, UnitSignatureBuilder


def test_hungarian_aligner_reduces_cost_for_permuted_linear_units():
    ref_weight = torch.eye(3)
    target_weight = ref_weight[[2, 0, 1]]
    ref_sig = UnitSignatureBuilder.output_signatures_from_weight(ref_weight)
    tgt_sig = UnitSignatureBuilder.output_signatures_from_weight(target_weight)
    identity_cost = torch.cdist(ref_sig, tgt_sig).diag().mean().item()
    perm, aligned_cost = HungarianAligner("hungarian").align(ref_sig, tgt_sig)
    assert perm.tolist() == [1, 2, 0]
    assert aligned_cost < identity_cost
