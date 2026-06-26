# HFedATM

Official code for **HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation**, accepted to **CVPR 2026**.

Paper: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/papers/Nguyen_HFedATM_Hierarchical_Federated_Domain_Generalization_via_Optimal_Transport_and_Regularized_CVPR_2026_paper.pdf)

This repository builds on the FedDG benchmarking codebase and extends it to the **Hierarchical Federated Domain Generalization (HFedDG)** setting, where clients are grouped under intermediate stations before server aggregation. HFedATM is a data-free hierarchical aggregation method that combines:

- **Filter-wise Optimal Transport (FOT)** alignment for convolutional filters.
- **Shrinkage-aware Regularized Mean (RegMean)** aggregation for linear layers.
- Compatibility with existing FedDG client-side baselines without changing local training.

## Repository Scope

This public branch is the **conference release** for HFedATM. It contains the HFedDG benchmark implementation, hierarchical station support, HFedATM aggregation, partitioning utilities, and experiment configs/scripts used to reproduce the paper-style runs.

It intentionally does **not** include datasets, downloaded tensors, W&B logs, checkpoints, or experiment outputs.

## Methods

Server-side methods include:

- `FedAvg`
- `HierarchicalFedAvg`
- `HFedATM`
- `RegMeanAll`
- `FisherMerging`
- `OTFusion`
- `FedMAStyle`
- `ModelSoup`
- `FedRCHFLGaussian` / `FedRC`
- `MTGC` / `MTGCApprox`

Client-side methods include:

- `ERM` / FedAvg-style local training
- `FedProx`
- `FedSR`
- `FedIIR`
- `IRM`
- `VREx`
- `Fish`
- `MMD`
- `Coral`
- `GroupDRO`
- `Mixup`
- `Scaffold`
- `AFL`
- `FedDG`
- `FedADG`
- `FedGMA`

See [docs/merging_methods.md](docs/merging_methods.md) for implementation notes on hierarchical aggregation methods and diagnostics.

## Installation

```bash
conda create --name hfedatm --file requirements.txt
conda activate hfedatm
```

Alternatively, create your own Python environment and install the packages listed in `requirements.txt`.

## Datasets

The benchmark supports datasets derived from WILDS and additional FedDG datasets used by the codebase, including PACS and FEMNIST metadata support.

Expected metadata layout:

```text
resources/pacs_v1.0/
resources/femnist_v1.0/
```

Large datasets are not stored in this repository. Set `dataset_path` in the JSON config files to the local directory where datasets are available.

Some FedDG methods use Fourier-transformed data. Preprocessing scripts are provided under `scripts/`; update each script's dataset root before running it.

## Running Experiments

All experiments are launched from JSON configs:

```bash
python main.py --config_file <path_to_config.json>
```

Example smoke runs:

```bash
python main.py --config_file configs/hfedatm_smoke.json --no_wandb
python main.py --config_file configs/hfedavg_smoke.json --no_wandb
```

PACS/HFedDG configs and grid scripts are under:

```text
configs/
scripts/
```

## W&B Logging

Weights & Biases logging is optional. To disable it:

```bash
python main.py --config_file <path_to_config.json> --no_wandb
```

To enable W&B, configure `wandb_env.py` or provide the corresponding project/entity settings in your environment.

## Code Structure

```text
main.py                 Entry point and config loading
src/server.py           Server-side FL/HFL/HFedATM methods
src/client.py           Client-side FedDG/local training methods
src/hierarchy.py        Station abstraction and client-station topology
src/partitioning.py     HFedDG/PACS partitioning utilities
src/merging.py          Model aggregation and RegMean/FOT helpers
src/sketches.py         Activation sketching utilities
src/datasets.py         Dataset wrappers
src/dataset_bundle.py   Dataset-specific model/loss/config bundles
configs/                JSON experiment configs
scripts/                Training, preprocessing, and analysis scripts
tests/                  Unit tests
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{nguyen2026hfedatm,
  title     = {HFedATM: Hierarchical Federated Domain Generalization via Optimal Transport and Regularized Mean Aggregation},
  author    = {Nguyen, Thinh and Phan, Trung and Nguyen, Binh T. and Doan, Khoa D. and Wong, Kok-Seng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

## Acknowledgement

This implementation is built on top of the FedDG benchmarking codebase:

- [Benchmarking Algorithms for Domain Generalization in Federated Learning](https://openreview.net/forum?id=wprSv7ichW)

We thank the original benchmark authors for releasing their codebase.
