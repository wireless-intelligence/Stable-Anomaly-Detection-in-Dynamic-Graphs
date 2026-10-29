import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


SCENARIOS = {
    1: "1/capture20110810.binetflow",
    2: "2/capture20110811.binetflow",
    3: "3/capture20110812.binetflow",
    4: "4/capture20110815.binetflow",
    5: "5/capture20110815-2.binetflow",
    6: "6/capture20110816.binetflow",
    7: "7/capture20110816-2.binetflow",
    8: "8/capture20110816-3.binetflow",
    9: "9/capture20110817.binetflow",
    10: "10/capture20110818.binetflow",
    11: "11/capture20110818-2.binetflow",
    12: "12/capture20110819.binetflow",
    13: "13/capture20110815-3.binetflow",
}


def _safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _safe_port(s):
    try:
        return float(int(s, 0))
    except (ValueError, TypeError):
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0


def _parse_starttime(s: str) -> float:
    if not s:
        return 0.0
    import datetime as _dt
    try:
        for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
            try:
                return _dt.datetime.strptime(s.strip(), fmt).timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    return 0.0


def load_binetflow(path: str):
    feats = []
    labels = []
    times = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = _safe_float(row.get("Dur", 0))
            pkts = _safe_float(row.get("TotPkts", 0))
            tot_b = _safe_float(row.get("TotBytes", 0))
            src_b = _safe_float(row.get("SrcBytes", 0))
            sport = _safe_port(row.get("Sport", 0))
            dport = _safe_port(row.get("Dport", 0))
            stos = _safe_float(row.get("sTos", 0))
            dtos = _safe_float(row.get("dTos", 0))
            feats.append([dur, pkts, tot_b, src_b, sport, dport, stos, dtos])
            lab = (row.get("Label", "") or "").lower()
            is_bot = int("from-botnet" in lab or ("botnet" in lab and "background" not in lab))
            labels.append(is_bot)
            times.append(_parse_starttime(row.get("StartTime", "")))
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    t = np.asarray(times, dtype=np.float64)
    return X, y, t


def standardize(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sigma


def knn_edges(X, k: int = 3, batch: int = 1024):
    n = X.shape[0]
    Xs = standardize(X).astype(np.float32)
    norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-12
    Xn = Xs / norms
    rows, cols = [], []
    for s in range(0, n, batch):
        e = min(s + batch, n)
        sims = Xn[s:e] @ Xn.T
        for i in range(e - s):
            sims[i, s + i] = -np.inf
            top = np.argpartition(-sims[i], k)[:k]
            rows.extend([s + i] * k)
            cols.extend(top.tolist())
        if (s // batch) % 50 == 0:
            print(f"  knn batch {s}/{n}", flush=True)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    A.data = np.ones_like(A.data)
    return A


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, required=True,
                   help="Path to CTU-13-Dataset (containing folders 1..13)")
    p.add_argument("--scenario", type=int, default=12,
                   help="Single scenario to use (default: 12, which has 325,471 flows)")
    p.add_argument("--out_mat", type=str, required=True)
    p.add_argument("--k", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    rel = SCENARIOS.get(args.scenario)
    if rel is None:
        raise ValueError(f"unknown scenario {args.scenario}")
    path = os.path.join(args.in_dir, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"Loading {path} ...", flush=True)
    X, y, t = load_binetflow(path)
    print(f"  flows={X.shape[0]} botnet={int(y.sum())} ratio={y.mean()*100:.3f}%", flush=True)
    print(f"Building K-NN graph (k={args.k}) ...", flush=True)
    A = knn_edges(X, k=args.k)
    print(f"  edges (undirected)={A.nnz // 2}", flush=True)
    os.makedirs(os.path.dirname(args.out_mat), exist_ok=True)
    sio.savemat(
        args.out_mat,
        {
            "A": A,
            "X": X,
            "y": y.reshape(-1, 1),
            "edge_u": A.tocoo().row.astype(np.int64),
            "edge_v": A.tocoo().col.astype(np.int64),
            "starttime": t,
        },
    )
    print(f"Saved to {args.out_mat}", flush=True)


if __name__ == "__main__":
    main()
