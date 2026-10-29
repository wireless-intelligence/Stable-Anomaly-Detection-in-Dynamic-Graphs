"""CONAD grid search across paper datasets (Figure 7 fill-in).

Sweeps the standard CONAD hyperparameters and records the best AUROC /
AUPRC_bal / Macro-F1_bal per dataset. Pygod's official CONAD wrapper
(`pygod.detector.CONAD`) is used end-to-end (gradient training).

Usage:
  python experiments/run_conad_grid.py \
      --datasets Cora,Citeseer,BlogCatalog,Pubmed,ACM,Flickr,Enron,CTU-13,Amazon \
      --output_dir outputs/conad_grid \
      --max_combos 24
"""

import argparse, itertools, json, os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pygod.detector import CONAD

from src.data import DataLoader
from src.evaluation import compute_all_metrics
from src.utils import set_seed


# Standard CONAD grid (full = 144 combos; we randomly sample by default).
# Names match pygod 1.1.x — `weight` is attribute-vs-structure (paper's alpha).
GRID = {
    "hid_dim":      [32, 64, 128],
    "num_layers":   [2, 3],
    "weight":       [0.3, 0.5, 0.7],
    "lr":           [1e-3, 5e-3],
    "epoch":        [100, 200],
    "dropout":      [0.0, 0.3],
    "weight_decay": [5e-4],
    "eta":          [0.5, 0.7],
    "m":            [50],
}


def to_pyg_data(loader, device):
    from torch_geometric.data import Data
    import scipy.sparse as sp
    A = sp.csr_matrix(loader.adj).tocoo()
    edge_index = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    x = torch.tensor(np.asarray(loader.features.todense() if sp.issparse(loader.features)
                                else loader.features), dtype=torch.float)
    y = torch.tensor(loader.labels, dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, y=y)
    return d.to(device)


def run_combo(data, params, gpu):
    set_seed(42); torch.manual_seed(42)
    det = CONAD(**params, gpu=gpu, verbose=0)
    t0 = time.time()
    det.fit(data)
    elapsed = time.time() - t0
    return det.decision_score_, elapsed


def grid_for_dataset(name, output_dir, max_combos):
    print(f"\n=== {name} ===", flush=True)
    set_seed(42)
    ld = DataLoader("data", name); ld.load()
    n = ld.stats["num_nodes"]
    n_anom = int(ld.labels.sum())
    print(f"  N={n:,}  F={ld.features.shape[1]}  anomalies={n_anom}", flush=True)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    gpu = 0 if use_cuda else -1
    print(f"  device={device}", flush=True)
    data = to_pyg_data(ld, device)

    keys = list(GRID.keys())
    grids = list(itertools.product(*[GRID[k] for k in keys]))
    if max_combos and max_combos < len(grids):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(grids), size=max_combos, replace=False)
        grids = [grids[i] for i in idx]
    print(f"  grid size: {len(grids)} combos", flush=True)

    results, best = [], None
    for ci, vals in enumerate(grids):
        params = dict(zip(keys, vals))
        try:
            scores, t = run_combo(data, params, gpu)
            mt = compute_all_metrics(ld.labels, scores)
            row = {
                "params": params,
                "auroc": float(mt["auroc"]),
                "auprc_bal": float(mt["auprc_bal"]),
                "macro_f1_bal": float(mt["macro_f1_bal"]),
                "elapsed_s": t,
            }
            results.append(row)
            if best is None or mt["auroc"] > best["auroc"]:
                best = row
                print(f"  [{ci+1}/{len(grids)}] *NEW BEST* AUROC={mt['auroc']*100:.2f}  "
                      f"params={params}  ({t:.1f}s)", flush=True)
            elif (ci + 1) % 5 == 0:
                print(f"  [{ci+1}/{len(grids)}] AUROC={mt['auroc']*100:.2f} "
                      f"(best={best['auroc']*100:.2f})", flush=True)
        except Exception as e:
            print(f"  [{ci+1}/{len(grids)}] FAILED: {e}", flush=True)

    out = {
        "dataset": name,
        "stats": {"num_nodes": n, "num_features": int(ld.features.shape[1]),
                  "num_anomalies": n_anom},
        "best": best, "all": results,
    }
    out_path = os.path.join(output_dir, f"conad_grid_{name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    if best:
        print(f"  >>> {out_path}", flush=True)
        print(f"  BEST: AUROC={best['auroc']*100:.2f}  "
              f"AUPRC_bal={best['auprc_bal']*100:.2f}  "
              f"F1_bal={best['macro_f1_bal']*100:.2f}", flush=True)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets",
                   default="Cora,Citeseer,BlogCatalog,Pubmed,ACM,Flickr,Enron,CTU-13,Amazon")
    p.add_argument("--output_dir", default="./outputs/conad_grid")
    p.add_argument("--max_combos", type=int, default=24)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summary = {}
    for ds in datasets:
        try:
            best = grid_for_dataset(ds, args.output_dir, args.max_combos)
            summary[ds] = best
        except Exception as e:
            print(f"!!! {ds} FAILED: {e}", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sum_path = os.path.join(args.output_dir, f"summary_{ts}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> summary: {sum_path}\n")
    print(f"{'Dataset':<14} {'AUROC':>7} {'AUPRC_bal':>10} {'F1_bal':>8}")
    for ds, b in summary.items():
        if b:
            print(f"{ds:<14} {b['auroc']*100:>7.2f} {b['auprc_bal']*100:>10.2f} "
                  f"{b['macro_f1_bal']*100:>8.2f}")


if __name__ == "__main__":
    main()
