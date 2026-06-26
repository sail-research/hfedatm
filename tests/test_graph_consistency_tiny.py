import torch

from src.merging import GraphConsistentAligner


def test_graph_consistent_aligner_keeps_tensor_shapes():
    ref = torch.randn(4, 3, 1, 1)
    target = ref[[2, 0, 3, 1]].clone()
    perm, _ = GraphConsistentAligner("hungarian").align_module_weight(ref, target)
    aligned = target.index_select(0, perm)
    assert aligned.shape == ref.shape
    assert perm.shape == (4,)
