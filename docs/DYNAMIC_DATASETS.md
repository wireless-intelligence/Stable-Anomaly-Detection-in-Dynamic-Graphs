# Real Dynamic-Graph Datasets

These datasets carry **genuine per-edge timestamps** and **native node-level
anomaly labels** (fraud / illicit / SAR), so SAG is streamed strictly once in
true time order — no shuffled multi-epoch replay. They evaluate anomaly detection on genuinely time-evolving graphs rather than
on replayed static attributed graphs.

| Dataset          | Nodes | Edges | Time | Labels | Size | Where |
|------------------|-------|-------|------|--------|------|-------|
| **DGraphFin**    | 3.7M  | 4.3M  | 1–821 (days) | fraud user (class 1) | ~1 GB | cloud |
| **Elliptic-tx**  | 203k  | 234k  | 1–49 steps | illicit tx | ~400 MB | local ok |
| **Elliptic-wallet** | 822k | 1.27M | timestep | illicit wallet | ~1 GB | cloud |
| **AMLSim**       | config | config | tx timestamp | SAR account | config | local/cloud |
| Bitcoin-Alpha/OTC, UCI, Email-DNC, ... | small | small | raw | synthetic* | small | local |

\* Generic temporal edge lists ship no node labels; SAG's loader injects
**synthetic temporal anomalies over real structure** (`--no_inject` to disable).
Always report these as synthetic, never as native labels.

Runners:
```bash
# in-memory single-pass temporal comparison
python experiments/run_dynamic_comparison.py --dataset DGraphFin \
    --methods SAG,TGN,JODIE,DyRep,TGAT,EvolveGCN,MIDAS,SedanSpot,DOMINANT \
    --n_runs 3 --max_edges 2000000

# Neo4j system-level streaming (real per-edge timestamps stored on :INTERACT)
export NEO4J_URI=bolt://127.0.0.1:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=changeme
python experiments/run_neo4j_dynamic.py --dataset AMLSim \
    --methods SAG,TGN,JODIE,MIDAS --edge_limit 200000
```

## Dynamic baselines (official)

Registered in `src/baselines/dynamic_official.py`, using **official** code:

| Name | Source | Notes |
|------|--------|-------|
| TGN, JODIE, DyRep, TGAT | official DyGLib (`vendor/DyGLib`, via `setup_vendor_baselines.sh`) | subprocess imports DyGLib's `MemoryModel`/`TGAT` + samplers; self-supervised temporal link prediction → per-node surprise |
| EvolveGCN | official `torch-geometric-temporal` (`EvolveGCNO`) | `pip install torch-geometric-temporal` |

Clone-free fallback (torch/PyG only, no vendor clone) under `*-reimpl` names —
faithful reimplementations on PyG's TGN framework, e.g.
`--methods SAG,TGN-reimpl,JODIE-reimpl,DyRep-reimpl,TGAT-reimpl,EvolveGCN-reimpl`.
Also available: native streaming detectors MIDAS / SedanSpot / NetWalk, and any
static GAD in `BASELINES` (run once on the accumulated graph, flagged `static`).
The runner groups results by paradigm: sag / dynamic / streaming / static.

---

## DGraphFin  (primary real dataset)

License-gated: register and download `dgraphfin.npz` from
<https://dgraph.xinye.com/dataset>. Place it at:

```
data/DGraphFin/dgraphfin.npz
```

(The PyG cache layout `data/DGraphFin/raw/dgraphfin.npz` is also accepted.)
Node classes: 0 = normal, 1 = fraud (positive), 2/3 = background (excluded from
evaluation via `eval_mask`).

Optional PyG helper (still needs the manual file):
```python
from torch_geometric.datasets import DGraphFin
DGraphFin(root="data/DGraphFin")   # verifies the npz is in place
```

## Elliptic++  (Bitcoin transaction + wallet graphs)

Repo: <https://github.com/git-disl/EllipticPlusPlus>
Original Elliptic (tx only): Kaggle `ellipticco/elliptic-data-set`.

Transaction graph — place any of these name variants under `data/Elliptic++/`:
```
elliptic_txs_edgelist.csv   (or txs_edgelist.csv)
elliptic_txs_features.csv   (or txs_features.csv)   # col0=txId, col1=timestep, rest=features
elliptic_txs_classes.csv    (or txs_classes.csv)    # txId,class  (1=illicit,2=licit,unknown)
```
Wallet graph (`--dataset Elliptic-wallet`):
```
AddrAddr_edgelist.csv
wallets_features.csv
wallets_classes.csv
```

## AMLSim  (synthetic AML transaction stream)

Repo: <https://github.com/IBM/AMLSim>. Generate, then place the CSV outputs
under `data/AMLSim/`:
```
transactions.csv     # sender/receiver/amount/timestamp/is_fraud (columns auto-detected)
accounts.csv         # is_sar / label per account   (optional)
sar_accounts.csv     # SAR account ids               (optional; preferred label source)
```
Column names vary across AMLSim versions — the loader matches them fuzzily.

## Generic temporal edge lists (BAG / SNAP family)

Bitcoin-Alpha/OTC, UCI, Email-DNC, Digg, MOOC, Wikipedia, Reddit, ...
Place a `(source, target, [weight], timestamp)` CSV/TXT under
`data/<Name>/<name>.csv`. Optional native labels: `data/<Name>/<name>_labels.csv`
with `node_id,label` rows (1 = anomaly). Without a label file the loader injects
synthetic temporal anomalies.

BAG benchmark (AAAI 2026) unified temporal datasets + baselines:
<https://github.com/> (search "BAG dynamic graph anomaly benchmark").
