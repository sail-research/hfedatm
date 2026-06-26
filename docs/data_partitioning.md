# Hierarchical Data Partitioning


## Scope

- Current implemented experiment scope: PACS.
- Other datasets can be added later by reusing `src.partitioning.build_hierarchical_partition(...)`.
- Label-skew methods are future work. `hds_label` and `hds_full` intentionally raise `NotImplementedError`.

## Partition Names

| Config name | Paper name | Purpose |
|---|---|---|
| `paper_lambda` | Original-lambda | Backward-compatible lambda split using `NonIIDSplitter`. |
| `paper_lambda_clustered` | Original-lambda-C | Original lambda clients, grouped into similar-domain stations. |
| `paper_lambda_mixed` | Original-lambda-M | Original lambda clients, mixed into similar station domain proportions. |
| `hds_dirichlet` | HDS | Station-first hierarchical Dirichlet domain-skew split. |
| `hds_inter` | HDS-Inter | High inter-station shift, mild intra-station shift. |
| `hds_intra` | HDS-Intra | Mild inter-station shift, high intra-station shift. |
| `hds_severe` | HDS-Severe | High inter-station and high intra-station shift. |
| `hds_quantity` | HDS-Q | HDS plus station/client quantity skew. |
| `hds_partial` | HDS-PP | HDS-Inter allocation plus partial station/client participation configs. |
| `hds_label`, `hds_full` | future | Label skew/full skew extensions, not enabled in this phase. |

## Original Paper Anchor

`paper_lambda` is the continuity split. It preserves the existing lambda behavior:

```text
NonIIDSplitter(num_shards=num_clients, iid=iid, seed=seed)
```

Use lambda/iid values `{1.0, 0.1, 0.0}` for continuity with the existing HFedDG setup.

## HDS Theory

HDS separates two heterogeneity levels:

- `alpha_station` controls station-level domain distributions and therefore inter-station divergence.
- `alpha_client` controls client distributions inside each station and therefore intra-station divergence.


Default presets:

- `hds_inter`: `alpha_station=0.2`, `alpha_client=10.0`
- `hds_intra`: `alpha_station=10.0`, `alpha_client=0.2`
- `hds_severe`: `alpha_station=0.2`, `alpha_client=0.2`
- `hds_quantity`: `alpha_station=0.3`, `alpha_client=0.5`, lognormal quantity skew
- `hds_partial`: HDS-Inter allocation plus `station_fraction < 1` and `station_client_fraction < 1`

## Recommended Usage

- Main candidate: `hds_inter`.
- Fallback main: `hds_dirichlet` with moderate alpha values.
- Continuity: `paper_lambda` with lambda in `{1.0, 0.1, 0.0}`.
- Low inter-station control: `paper_lambda_mixed`.
- Intra-station ablation: `hds_intra`.
- Stress split: `hds_severe` or `hds_quantity`.


## Leakage Rule

Partitioning uses only the train subset passed by `main.py`. Validation, test, target, and unseen-domain data must not influence split construction or diagnostics.

## Diagnostics

Every partition writes reports under:

```text
{data_path}/partition_reports/{run_id}/
```

Files:

- `partition_diagnostics.json`
- `client_domain_counts.csv`
- `station_domain_counts.csv`
- `partition_summary.md`
- optional plots when `partition_plot=true`

Key metrics include sample conservation, duplicated/missing counts, empty clients, sample CV/Gini, domain entropy, intra-station JS, and inter-station JS.

## Commands

Metadata-only diagnostics:

```bash
python scripts/simulate_pacs_partitions.py \
  --output_dir outputs/pacs_partition_sim \
  --num_clients 100 \
  --num_stations 10 \
  --methods paper_lambda paper_lambda_clustered paper_lambda_mixed hds_dirichlet hds_inter hds_intra hds_severe hds_quantity hds_partial \
  --seeds 0 1 2 \
  --iid_values 1.0 0.1 0.0
```

Analyze diagnostics:

```bash
python scripts/analyze_pacs_partitions.py outputs/pacs_partition_sim \
  --output_dir outputs/pacs_partition_selection
```

Convenience wrapper:

```bash
bash scripts/run_pacs_partition_diagnostics.sh
```

Smoke training grid, if PACS image data/cache is available:

```bash
bash scripts/run_pacs_partition_smoke_grid.sh
```

Full-grid command preparation:

```bash
bash scripts/run_pacs_partition_full_grid.sh
```

