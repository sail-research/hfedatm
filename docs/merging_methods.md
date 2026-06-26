# Hierarchical Merging Methods

This codebase still uses the original dispatch style:

```bash
```

The JSON key `server_method` must be a class imported by `from src.server import *`.
The JSON key `client_method` must be a class imported by `from src.client import *`.

## Added Server Methods

- `HierarchicalFedAvg`: existing clients -> stations -> server weighted averaging baseline.
- `HFedATM`: original HFedATM-style station-server merge. Conv2d layers use FOT-style Hungarian output-filter matching plus averaging. Linear layers use RegMean from activation Gram sketches. Other parameters and buffers use safe weighted averaging.
- `RegMeanAll`: applies RegMean-style merging to Conv2d and Linear layers without graph-consistent alignment.
- `FisherMerging`: Fisher-weighted station model averaging using diagonal Fisher estimates from source data.
- `OTFusion`: layer-wise OT/Hungarian matching plus averaging, without the RegMean solver.
- `FedMAStyle`: practical station-server adaptation of FedMA-style matched averaging. This is not guaranteed to be the exact original multi-round FedMA protocol.
- `ModelSoup`: uniform station model soup. Greedy soup is deliberately source-validation only; if no source-validation station metric exists, it falls back to uniform and logs a warning.
- `FedRCHFLGaussian`: IROS 2024 HFL Gaussian FedRC. It computes RGB or feature diagonal Gaussian summaries and uses Gaussian-aware weights at both client-to-station and station-to-server aggregation. The default paper-aligned distance is Bhattacharyya.
- `FedRC`: alias for `FedRCHFLGaussian`.
- `FedRCRobustClustering`: placeholder for the ICML 2024 robust clustering FedRC. It raises `NotImplementedError` and is intentionally not confused with the IROS HFL Gaussian method.
- `MTGCApprox`: documented approximation that tracks group/global drift but does not inject exact MTGC correction terms into every client optimizer step.
- `MTGC`: exact MTGC port based on the official NeurIPS 2024 code structure. Use with `client_method="MTGCClient"` and full station/client participation.

## Added Client Methods

- `FedIIR`: exact FedIIR client objective from the official ICML 2023 code. A `FedIIRServer` or station hook must provide the EMA-smoothed mean classifier gradient before local training.
- `MTGCClient`: exact MTGC local objective with linear correction term `theta dot (Z_i + Y_j)`.

## FedIIR Paper Reproduction Notes

FedIIR's original paper runner is one-level FL, not HFL. Its official code treats each domain as a test domain in turn, splits source domains into clients, samples clients each communication round, computes the classifier mean gradient, then applies the FedIIR penalty during local updates.

In this repo:

- Use `server_method="FedIIRServer"` and `client_method="FedIIR"` for the flat FedIIR objective.
- Use `client_method="FedIIR"` with HFL servers only when you intentionally want FedIIR local training inside stations.
- The current HFedATM/FedDG runner uses its own PACS metadata split, so exact FedIIR paper numbers require the official FedIIR leave-one-domain-out runner or a matching LODO split wrapper.

## Privacy And Target Leakage

Only compact sketches, diagonal Fisher tensors, and Gaussian summaries leave clients/stations. Raw images, raw batches, and raw activation matrices are not serialized.

Greedy model soup and any future hyperparameter selection must use source validation only. Target-domain validation/statistics should not be used for tuning or station weighting.

## Known Limitations

- Most merging methods assume homogeneous model architectures.
- `sinkhorn`, low-rank sketches, and random-projection sketches are implemented. Low-rank/random-projection sketches materialize an approximate Gram only when dimensions are within `activation_sketch_max_full_dim`; otherwise factors are retained and RegMean falls back explicitly instead of pretending a full Gram exists.
- `FedRCRobustClustering` is not implemented in this HFL benchmark layer. Use the official ICML 2024 repository for exact robust-clustering FedRC experiments.

## Diagnostics

Hierarchical merge methods write JSONL diagnostics to:

```text
{data_path}/diagnostics/{run_id}_{server_method}.jsonl
```

Tracked fields include round, station ids/sizes, aggregation coefficients, sketch/Fisher/Gaussian payload size, Conv2d/Linear merge counts, alignment costs, RegMean failures, and fallback counts. When WandB is enabled, selected aggregation metrics are logged under `agg/*`.

## Smoke Configs

The `configs/` folder contains small JSON configs:

- `hfedavg_smoke.json`
- `hfedatm_smoke.json`
- `regmean_all_smoke.json`
- `fisher_merging_smoke.json`
- `otfusion_smoke.json`
- `fedma_style_smoke.json`
- `model_soup_smoke.json`
- `fedrc_hfl_gaussian_smoke.json`
- `fediir_pacs_smoke.json`
- `mtgc_smoke.json`
- `mtgc_approx_smoke.json`

Set `data_path` and `dataset_path` to your local/server dataset location before real training.
