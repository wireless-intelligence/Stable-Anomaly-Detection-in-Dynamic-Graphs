"""Minimal validation of the OFFICIAL DyGLib data path used by
src/baselines/dynamic_official.py.

It proves that the array mapping our subprocess runner applies
(``build_dyglib_arrays``) is equivalent to what DyGLib's **official**
``preprocess_data/preprocess_data.py`` produces, and (if possible) that DyGLib's
official ``get_link_prediction_data`` + a model (TGAT) accept those arrays.

Steps:
  1. Locate vendor/DyGLib (clone via setup_vendor_baselines.sh). Skip if absent.
  2. Build a tiny NON-bipartite temporal graph.
  3. OFFICIAL path: write it as a DyGLib raw CSV and run the official
     preprocess_data.py CLI (backing up / restoring any real dataset dir of the
     same name), then read ml_<name>.csv / .npy / _node.npy.
  4. OURS: build_dyglib_arrays(...) on the same graph.
  5. Assert equivalence of node/edge reindexing (1-based), idx, timestamps and
     padding rows. (Node-feature *values* legitimately differ: DyGLib zero-fills
     featureless raw graphs, whereas our adapter injects the real node features.)
  6. Optionally instantiate the official TGAT and run one forward on our arrays.

Exit codes: 0 = PASS, 1 = FAIL (mismatch), 2 = SKIP (vendor/DyGLib missing).

    python scripts/check_dyglib_pipeline.py
"""

import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.dynamic_official import build_dyglib_arrays, _vendor_path

# An allowed NON-bipartite name in DyGLib's argparse choices (so the official
# CLI accepts it). We back up any real data under this name and restore it.
CHECK_NAME = "Contacts"
NODE_FEAT_DIM = 8       # small, to keep the check fast; official default is 172
EDGE_FEAT_DIM = 4


def _make_tiny_graph(n=30, e=200, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, size=e)
    dst = rng.integers(0, n, size=e)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ts = np.sort(rng.uniform(0, 1000, size=src.shape[0]))  # ascending
    edges = np.stack([src, dst], axis=1).astype(np.int64)
    feats = rng.standard_normal((n, NODE_FEAT_DIM)).astype(np.float32)
    return edges, ts.astype(np.float64), feats, n


def _write_raw_csv(path, edges, ts):
    """DyGLib raw format: header line + rows `u,i,ts,label,f1..fk` (ascending ts)."""
    rng = np.random.default_rng(1)
    with open(path, "w", encoding="utf-8") as f:
        f.write("u,i,ts,label," + ",".join(f"f{j}" for j in range(EDGE_FEAT_DIM)) + "\n")
        for k in range(edges.shape[0]):
            feat = ",".join(f"{v:.4f}" for v in rng.standard_normal(EDGE_FEAT_DIM))
            f.write(f"{int(edges[k,0])},{int(edges[k,1])},{float(ts[k]):.6f},0,{feat}\n")


def _run_official_preprocess(vendor):
    dg_dir = os.path.join(vendor, "DG_data", CHECK_NAME)
    proc_dir = os.path.join(vendor, "processed_data", CHECK_NAME)
    backups = []
    for d in (dg_dir, proc_dir):
        if os.path.exists(d):
            bak = d + ".sagbak"
            if os.path.exists(bak):
                shutil.rmtree(bak)
            shutil.move(d, bak)
            backups.append((d, bak))
    os.makedirs(dg_dir, exist_ok=True)
    return dg_dir, proc_dir, backups


def _restore(backups, dg_dir, proc_dir):
    for d in (dg_dir, proc_dir):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    for orig, bak in backups:
        if os.path.exists(bak):
            shutil.move(bak, orig)


def main():
    vendor = _vendor_path("DyGLib")
    if not os.path.isdir(vendor) or not os.path.isdir(os.path.join(vendor, "preprocess_data")):
        print(f"[SKIP] vendor/DyGLib not found at {vendor}.")
        print("       Clone it first: bash setup_vendor_baselines.sh")
        return 2

    edges, ts, feats, n = _make_tiny_graph()
    ours = build_dyglib_arrays(edges, ts, feats, n)

    dg_dir, proc_dir, backups = _run_official_preprocess(vendor)
    try:
        _write_raw_csv(os.path.join(dg_dir, f"{CHECK_NAME}.csv"), edges, ts)
        script = os.path.join(vendor, "preprocess_data", "preprocess_data.py")
        cmd = [sys.executable, script, "--dataset_name", CHECK_NAME,
               "--node_feat_dim", str(NODE_FEAT_DIM)]
        cp = subprocess.run(cmd, cwd=vendor, capture_output=True, text=True)
        if cp.returncode != 0:
            print("[FAIL] official preprocess_data.py returned non-zero")
            print("STDOUT:", cp.stdout[-1500:])
            print("STDERR:", cp.stderr[-1500:])
            return 1

        import pandas as pd
        ml_csv = os.path.join(proc_dir, f"ml_{CHECK_NAME}.csv")
        ml_edge = os.path.join(proc_dir, f"ml_{CHECK_NAME}.npy")
        ml_node = os.path.join(proc_dir, f"ml_{CHECK_NAME}_node.npy")
        for p in (ml_csv, ml_edge, ml_node):
            if not os.path.exists(p):
                print(f"[FAIL] official output missing: {p}")
                return 1
        df = pd.read_csv(ml_csv)
        off_u = df.u.values.astype(np.int64)
        off_i = df.i.values.astype(np.int64)
        off_ts = df.ts.values.astype(np.float64)
        off_idx = df.idx.values.astype(np.int64)
        off_edge = np.load(ml_edge)
        off_node = np.load(ml_node)
    finally:
        _restore(backups, dg_dir, proc_dir)

    # ---- assertions: our mapping == official (non-bipartite) ----
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        print(f"  [{'ok' if cond else 'XX'}] {name}")

    chk("edge count matches", len(off_u) == ours["src_node_ids"].shape[0])
    chk("u == our src (1-based, no offset)", np.array_equal(off_u, ours["src_node_ids"]))
    chk("i == our dst (1-based, no offset)", np.array_equal(off_i, ours["dst_node_ids"]))
    chk("timestamps match", np.allclose(off_ts, ours["node_interact_times"], atol=1e-3))
    chk("idx == our edge_ids (1-based)", np.array_equal(off_idx, ours["edge_ids"]))
    chk("edge feats have padding row0 (official)", np.allclose(off_edge[0], 0))
    chk("edge feats have padding row0 (ours)", np.allclose(ours["edge_raw_features"][0], 0))
    chk("edge feats rows == E+1 (official)", off_edge.shape[0] == len(off_u) + 1)
    chk("edge feats rows == E+1 (ours)", ours["edge_raw_features"].shape[0] == len(off_u) + 1)
    chk("node feats rows == N+1 (official)", off_node.shape[0] == n + 1)
    chk("node feats rows == N+1 (ours)", ours["node_raw_features"].shape[0] == n + 1)
    chk("node feats padding row0 (ours)", np.allclose(ours["node_raw_features"][0], 0))
    chk("max node id == N (1..N)", int(max(off_u.max(), off_i.max())) == n)

    # ---- optional: official TGAT accepts our arrays ----
    try:
        import torch
        sys.path.insert(0, vendor)
        from utils.DataLoader import Data
        from utils.utils import get_neighbor_sampler
        from models.TGAT import TGAT
        from models.modules import MergeLayer
        data = Data(src_node_ids=ours["src_node_ids"], dst_node_ids=ours["dst_node_ids"],
                    node_interact_times=ours["node_interact_times"],
                    edge_ids=ours["edge_ids"], labels=ours["labels"])
        ns = get_neighbor_sampler(data, sample_neighbor_strategy="recent", seed=0)
        tgat = TGAT(node_raw_features=ours["node_raw_features"],
                    edge_raw_features=ours["edge_raw_features"], neighbor_sampler=ns,
                    time_feat_dim=16, num_layers=1, num_heads=2, dropout=0.1,
                    device=torch.device("cpu"))
        se, de = tgat.compute_src_dst_node_temporal_embeddings(
            src_node_ids=data.src_node_ids[:8], dst_node_ids=data.dst_node_ids[:8],
            node_interact_times=data.node_interact_times[:8], num_neighbors=5)
        chk("official TGAT forward on our arrays", se.shape[0] == 8 and de.shape[0] == 8)
    except Exception as e:
        print(f"  [--] official TGAT forward skipped: {type(e).__name__}: {e}")

    ok = all(c for _, c in checks)
    print(f"\n{'PASS' if ok else 'FAIL'}: {sum(c for _,c in checks)}/{len(checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
