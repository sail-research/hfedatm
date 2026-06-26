# Configs

This directory intentionally contains only minimal public smoke configs.

- `hfedavg_smoke.json`: small HierarchicalFedAvg sanity run.
- `hfedatm_smoke.json`: small HFedATM sanity run.

Before running, update `dataset_path` to your local dataset root or keep the default `./data/datasets/` layout. Generated experiment grids, machine-specific paths, W&B run metadata, and GPU-specific configs are intentionally excluded from the public release.
