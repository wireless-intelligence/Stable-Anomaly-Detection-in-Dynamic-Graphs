"""SAG memory overhead breakdown profiler (revision experiment 11).

After running SAG end-to-end on a target dataset, walks the model and reports
the per-component byte cost.  Components covered:

  state.s                  : per-node anomaly state (R columns x N rows)
  state.exposure           : exposure counters lambda_v (length N)
  state.last_time          : last update time per node (length N)
  scorer.edge_prev/curr    : exact edge count maps (dict)
  scorer.node_prev/curr    : per-node window counters (length N)
  kernel.sketch / weights  : multi-scale sketch state if present
  attr.attribute_evidence  : node attribute predicate (length N)
  attr.iforest_evidence    : iForest predicate if computed (length N)
  attr.pca_mahal_evidence  : PCA-Mahalanobis predicate if computed (length N)
  struct.structural_score  : structural-statistics predicate (length N)
  trained_ae_score         : cached GCN-AE attribute predicate if used
  process_rss_gb           : Python process RSS via psutil (fallback baseline)

For each component we compute an explicit byte count via numpy .nbytes when
the field is an ndarray, by scaling by sizeof(np.float64) for length-N
vectors when the field type is wrong, and via a small-object-size estimator
for python dicts (sys.getsizeof(d) + sum of per-entry pointer overhead).

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision/memory_breakdown.csv
  outputs/results/revision/memory_breakdown_report.md
"""

import argparse
import copy
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


COLS = [
    "dataset", "component", "kind", "n_entries", "bytes",
    "bytes_per_node", "bytes_per_edge_processed",
    "notes",
]


def _is_torch_tensor(x):
    return type(x).__module__.startswith("torch") and hasattr(x, "numel")


def _numel(arr):
    if arr is None:
        return 0
    if _is_torch_tensor(arr):
        return int(arr.numel())
    if hasattr(arr, "size") and not callable(arr.size):
        return int(arr.size)
    if hasattr(arr, "shape"):
        n = 1
        for d in tuple(arr.shape):
            n *= int(d)
        return int(n)
    return 0


def _np_bytes(arr):
    if arr is None:
        return 0, "ndarray (None)"
    if _is_torch_tensor(arr):
        try:
            element_size = int(arr.element_size())
        except Exception:
            element_size = 4
        nbytes = _numel(arr) * element_size
        try:
            dev = str(arr.device)
        except Exception:
            dev = "?"
        return int(nbytes), f"tensor dtype={arr.dtype} shape={tuple(arr.shape)} device={dev}"
    if hasattr(arr, "nbytes") and not callable(getattr(arr, "nbytes", None)):
        return int(arr.nbytes), f"ndarray dtype={arr.dtype} shape={tuple(arr.shape)}"
    return 0, "n/a"


def _dict_bytes(d):
    """Estimate dict + entries cost (cap on small int keys / float values)."""
    if d is None:
        return 0, "dict (None)"
    n = len(d)
    if n == 0:
        return sys.getsizeof(d), f"empty dict (sizeof={sys.getsizeof(d)})"
    # one sample to estimate per-entry overhead
    sample = next(iter(d.items()))
    key_sz = sys.getsizeof(sample[0])
    if isinstance(sample[0], tuple):
        key_sz += sum(sys.getsizeof(x) for x in sample[0])
    val_sz = sys.getsizeof(sample[1])
    per_entry = key_sz + val_sz + 32  # conservative slot overhead
    total = sys.getsizeof(d) + n * per_entry
    return int(total), f"dict (n={n}, ~{per_entry}B/entry)"


def _scipy_bytes(A):
    if A is None:
        return 0, "sparse (None)"
    if not sp.issparse(A):
        return 0, "not sparse"
    A = A.tocsr() if not isinstance(A, sp.csr_matrix) else A
    b = int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)
    return b, f"csr nnz={A.nnz}"


def profile_model(model, loader, num_edges):
    s = loader.get_stats()
    N = s["num_nodes"]
    rows = []

    def add(component, kind, n_entries, b, notes=""):
        rows.append({
            "component": component, "kind": kind,
            "n_entries": int(n_entries) if n_entries is not None else 0,
            "bytes": int(b),
            "bytes_per_node": float(b) / max(1, N),
            "bytes_per_edge_processed": float(b) / max(1, num_edges),
            "notes": notes,
        })

    def _is_array_like(x):
        return x is not None and (
            isinstance(x, np.ndarray) or _is_torch_tensor(x)
        )

    s_arr = getattr(model.state, "s", None)
    if _is_array_like(s_arr):
        b, notes = _np_bytes(s_arr)
        add("state.s", "ndarray", _numel(s_arr), b, notes)

    expo = getattr(model.state, "exposure", None)
    if _is_array_like(expo):
        b, notes = _np_bytes(expo)
        add("state.exposure", "ndarray", _numel(expo), b, notes)
    else:
        add("state.exposure", "ndarray", 0, 0,
            "exposure counters not exposed on model.state")

    lt = getattr(model.state, "last_time", None)
    if _is_array_like(lt):
        b, notes = _np_bytes(lt)
        add("state.last_time", "ndarray", _numel(lt), b, notes)
    else:
        add("state.last_time", "ndarray", 0, 0, "n/a")

    sc = model.scorer
    for fname in ("edge_prev", "edge_curr"):
        d = getattr(sc, fname, None)
        b, notes = _dict_bytes(d) if isinstance(d, dict) else (0, "n/a")
        add(f"scorer.{fname}", "dict",
            len(d) if isinstance(d, dict) else 0, b, notes)

    for fname in ("node_prev", "node_curr"):
        v = getattr(sc, fname, None)
        if _is_array_like(v):
            b, notes = _np_bytes(v)
            add(f"scorer.{fname}", "ndarray", _numel(v), b, notes)
        else:
            add(f"scorer.{fname}", "ndarray", 0, 0, "n/a")

    k = getattr(model, "kernel", None)
    if k is not None:
        for fname in ("sketch", "weights", "S", "z"):
            attr_v = getattr(k, fname, None)
            if _is_array_like(attr_v):
                b, notes = _np_bytes(attr_v)
                add(f"kernel.{fname}", "ndarray", _numel(attr_v), b, notes)
    else:
        add("kernel", "ndarray", 0, 0, "kernel sketch not in use")

    a = getattr(model, "attr", None)
    if a is not None:
        for fname in ("attribute_evidence", "iforest_evidence",
                      "pca_mahal_evidence"):
            v = getattr(a, fname, None)
            if _is_array_like(v):
                b, notes = _np_bytes(v)
                add(f"attr.{fname}", "ndarray", _numel(v), b, notes)
    tae = getattr(model, "trained_ae_score", None)
    if _is_array_like(tae):
        b, notes = _np_bytes(tae)
        add("trained_ae_score", "ndarray", _numel(tae), b, notes)

    # Structural predicate
    st = getattr(model, "struct", None)
    ss = getattr(st, "structural_score", None) if st is not None else None
    if _is_array_like(ss):
        b, notes = _np_bytes(ss)
        add("struct.structural_score", "ndarray", _numel(ss), b, notes)

    # Process RSS (informational baseline)
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        rss_gb = rss / (1024 ** 3)
        add("process_rss", "informational", 0, int(rss),
            f"psutil.Process().memory_info().rss = {rss_gb:.3f}GB")
    except Exception:
        pass

    return rows


def write_csv(dataset, num_edges, rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            r2 = {"dataset": dataset, **r}
            w.writerow({k: r2.get(k, "") for k in COLS})


def write_report(dataset, stats, rows, path):
    rss = next((r for r in rows if r["component"] == "process_rss"), None)
    rss_gb = (rss["bytes"] / (1024 ** 3)) if rss else float("nan")
    total_explicit = sum(r["bytes"] for r in rows
                         if r["component"] != "process_rss")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# SAG memory breakdown — {dataset}\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"- num_nodes: **{stats['num_nodes']}**\n")
        f.write(f"- num_edges: **{stats['num_edges']}**\n")
        f.write(f"- num_anomalies: {stats['num_anomalies']}\n\n")
        f.write(f"- explicit-component total: **{total_explicit/1024/1024:.2f} MB**\n")
        f.write(f"- python process RSS:        **{rss_gb:.3f} GB** (informational; "
                "includes interpreter, numpy, scipy, torch, all imports)\n\n")
        f.write("## Per-component breakdown\n\n")
        f.write("| component | kind | entries | bytes | MB | per-node (B) | per-edge (B) | notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        sorted_rows = sorted([r for r in rows if r["component"] != "process_rss"],
                             key=lambda r: -r["bytes"])
        for r in sorted_rows:
            f.write("| `{c}` | {k} | {n} | {b} | {mb:.3f} | {bn:.1f} | {be:.4f} | {nt} |\n".format(
                c=r["component"], k=r["kind"], n=r["n_entries"],
                b=r["bytes"], mb=r["bytes"] / 1024 / 1024,
                bn=r["bytes_per_node"], be=r["bytes_per_edge_processed"],
                nt=r["notes"]))
        f.write("\n")
        f.write("## Notes on accounting\n\n")
        f.write(
            "- ndarray sizes are exact via `.nbytes`.\n"
            "- dict sizes use `sys.getsizeof(d)` plus a per-entry estimate "
            "(`sizeof(key) + sizeof(value) + 32 B slot overhead`). This is a "
            "rough lower-bound -- CPython's hash table over-allocates by ~33%, "
            "so the true memory occupancy is slightly higher.\n"
            "- `process_rss` is the OS-reported resident set size of the "
            "Python interpreter at the end of the run; it is NOT the sum of "
            "the explicit component sizes (it includes all imports, the SAG "
            "model, its inputs, and torch caches).\n"
            "- The exact edge-count maps (`scorer.edge_prev`/`edge_curr`) "
            "scale with the number of *unique* edge keys in the current "
            "window. Substituting a count-min sketch (e.g. MIDAS-style) "
            "would replace these dicts with two fixed-size O(num_hash * "
            "buckets) arrays. The runner reports the dict size measured here "
            "so the trade-off can be quantified directly against the MIDAS "
            "row of the scalability experiment, which uses 32768 buckets x 2 "
            "hashes = 65536 doubles = ~512 KB regardless of stream length.\n")


# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc, seed):
    cfg = CfgShim()
    for k in dir(Config):
        if k.startswith("_") or callable(getattr(Config, k, None)):
            continue
        try:
            setattr(cfg, k, copy.deepcopy(getattr(Config, k)))
        except Exception:
            pass
    for k, v in dc.items():
        try:
            setattr(cfg, k, copy.deepcopy(v))
        except Exception:
            pass
    cfg.has_ground_truth = bool(dc.get("has_ground_truth", False))
    cfg.use_continuous_dev = False
    cfg.use_real_time_decay = False
    cfg.seed = int(seed)
    return cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Enron")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    log_dir = os.path.join(
        args.output_dir, "logs",
        f"memory_breakdown_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(canon_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger("SAG_MEM_PROF", log_dir)

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    try:
        dc = get_dataset_config(args.dataset)
    except ValueError as e:
        logger.error(f"unknown dataset {args.dataset}: {e}")
        return
    loader = DataLoader(
        data_dir="./data", dataset_name=args.dataset,
        clique_size=dc.get("clique_size", 15),
        num_cliques=dc.get("num_cliques", 10),
        seed=args.seed,
    )
    loader.load()
    s = loader.get_stats()
    logger.info(f"stats: {s}")

    set_seed(args.seed)
    cfg = build_cfg(dc, seed=args.seed)
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(args.seed)
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    _ = model.get_scores()

    rows = profile_model(model, loader, num_edges=s["num_edges"])

    csv_path = os.path.join(canon_dir, "memory_breakdown.csv")
    md_path = os.path.join(canon_dir, "memory_breakdown_report.md")

    if os.path.exists(csv_path) and not args.overwrite:
        logger.warning(f"refusing to overwrite {csv_path} (pass --overwrite)")
    else:
        write_csv(args.dataset, s["num_edges"], rows, csv_path)
        logger.info(f"wrote {csv_path}")
    if os.path.exists(md_path) and not args.overwrite:
        logger.warning(f"refusing to overwrite {md_path} (pass --overwrite)")
    else:
        write_report(args.dataset, s, rows, md_path)
        logger.info(f"wrote {md_path}")

    save_json({"dataset": args.dataset, "stats": s, "rows": rows,
               "argv": sys.argv,
               "timestamp": datetime.now().isoformat(timespec="seconds")},
              os.path.join(canon_dir, "memory_breakdown.json"))


if __name__ == "__main__":
    main()
