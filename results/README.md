# Results

| Directory | Content |
|---|---|
| [`main_table/`](main_table/) | SAG output behind Tables 2 and 3 of the paper (all eight benchmark datasets) |
| [`dynamic/`](dynamic/) | dynamic graph streams (Table 4) and their Neo4j deployment |
| [`seed_variance/`](seed_variance/), [`per_seed_alpha_sweeps/`](per_seed_alpha_sweeps/), [`neo4j_per_seed/`](neo4j_per_seed/), [`default_config/`](default_config/) | per-seed records showing that SAG scores are deterministic |

## SAG on the benchmark datasets (Tables 2 and 3)

Values in % from [`main_table/sag_all_datasets.json`](main_table/sag_all_datasets.json) (run `all_20260427_162846`, `n_runs = 3`). Macro-F1 and AUPRC are the balanced variants `macro_f1_bal` and `auprc_bal`; AUROC is computed on all nodes.

| Dataset | Type | Macro-F1 | AUROC | AUPRC | raw (`macro_f1_bal` / `auroc` / `auprc_bal`) |
|---|---|:-:|:-:|:-:|---|
| YelpReviews | ground-truth | 74.1 ± 0.0 | 79.1 ± 0.0 | 72.0 ± 0.0 | 74.11 / 79.13 / 71.98 |
| Amazon | ground-truth | 81.7 ± 0.0 | 88.6 ± 0.0 | 86.7 ± 0.0 | 81.67 / 88.56 / 86.65 |
| Enron | ground-truth | 78.8 ± 0.0 | 79.6 ± 0.0 | 79.0 ± 0.0 | 78.81 / 79.61 / 79.00 |
| CTU-13 | ground-truth | 94.3 ± 0.0 | 93.6 ± 0.0 | 84.9 ± 0.0 | 94.30 / 93.60 / 84.86 |
| BlogCatalog | injected | 72.3 ± 0.0 | 78.6 ± 0.0 | 81.4 ± 0.0 | 72.25 / 78.58 / 81.44 |
| Cora | injected | 90.0 ± 0.0 | 94.5 ± 0.0 | 93.3 ± 0.0 | 90.03 / 94.52 / 93.29 |
| Pubmed | injected | 91.3 ± 0.0 | 95.7 ± 0.0 | 93.8 ± 0.0 | 91.29 / 95.68 / 93.76 |
| Citeseer | injected | 91.7 ± 0.0 | 96.2 ± 0.0 | 95.4 ± 0.0 | 91.73 / 96.20 / 95.37 |

These runs use anchored evidence selection (`fusion_consensus=False`); the dynamic graph streams use consensus selection (`fusion_consensus=True`, the default). Both rules are described in Section 4.6 of the paper.

## Why the standard deviation is 0.0

SAG's streaming inference is a deterministic function of the input edge sequence: given the same edge order, it produces identical node scores for every random seed. The ground-truth datasets fix the edge order by construction (Amazon: relation-by-relation enumeration; CTU-13: chronological order of connection timestamps; Enron and YelpReviews: the underlying email and review timelines), and the remaining components (kernel sketch, evidence transforms) are fixed at construction time.

The per-seed records make this directly checkable:

```
Enron        per_seed_alpha_sweeps/Enron.json        alpha = 1.0, seeds 42 / 100 / 200
             AUROC = 79.6112   AUPRC_bal = 78.9988   F1_bal = 78.8125   (all seeds)

YelpReviews  seed_variance/sag_seed_variance_raw.csv  axis = stream_order, seeds 0..9
             AUROC = 79.1339   AUPRC_bal = 71.9832   F1_bal = 74.1100   (all seeds)

Amazon       neo4j_per_seed/neo4j_benchmark_per_seed.csv   method = SAG, seeds 0..2
             AUROC = 88.5629                                (all seeds)
             default_config/Amazon.json:  AUPRC_bal = 86.6453   F1_bal = 81.6693
```

For the JSON sweep files, the per-seed list is at `params.alpha.values[?value=1.0].per_seed`; for the CSV files, filter rows by `dataset` and `axis` (or `method`). The drivers are `experiments/run_param_seed_sensitivity.py`, `experiments/run_sag_seed_variance.py`, and `experiments/neo4j_benchmark.py`.
