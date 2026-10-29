# Dynamic graph streams: raw results

| File | Content |
|---|---|
| `all_metrics_dynamic.csv` | one row per (dataset, method): AUROC, AUPRC, AUPRC_bal, Macro-F1, Macro-F1_bal (mean, std), per-edge latency, peak memory |
| `all_metrics_neo4j.csv` | Neo4j deployment: same metrics plus end-to-end query latency |
| `runs/<dataset>_dyn_<timestamp>/dynamic_comparison.json` | per-run output of `experiments/run_dynamic_comparison.py` (dataset statistics, per-seed values) |
| `neo4j/neo4j_dyn_<dataset>_<timestamp>.json` | per-run output of `experiments/run_neo4j_dynamic.py` |

The table in the top-level README reports `macro_f1_bal`, `auroc`, and `auprc_bal` from `all_metrics_dynamic.csv`. SAG entries come from `AMLSim_dyn_20260725_160605`, `Bitcoin-Alpha_dyn_20260725_160552`, and `Bitcoin-OTC_dyn_20260725_160557` (default configuration, `fusion_consensus=True`); baseline entries come from the other runs of the same dataset.

Regenerate the figures with:

```bash
python scripts/figures/plot_dynamic_streams.py
```
