#!/usr/bin/env python3
"""
Flow-based CTU-13 graph builder - nodes are network flows.
Edges are built from 4 K-NN sources by sharing key (SrcAddr / DstAddr / Proto / time).
Target: ~325K nodes, ~2M edges (K=3 K-NN graph) matching paper Table 1.
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.io import savemat

INFECTED_IPS = {"147.32.84.165", "147.32.84.191", "147.32.84.192"}


def pick_best_text_flow(in_dir: str) -> str:
    parent = os.path.dirname(in_dir)
    for fname in ["capture20110819.binetflow.2format", "capture20110819.binetflow"]:
        for root in [parent, in_dir]:
            p = os.path.join(root, fname)
            if os.path.exists(p):
                return p

    cand = []
    for root in [in_dir, parent]:
        if not os.path.isdir(root):
            continue
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if not os.path.isfile(p):
                continue
            if any(k in fn.lower() for k in ["binetflow", "netflow", "flow"]):
                if "biargus" in fn.lower():
                    continue
                try:
                    sz = os.path.getsize(p)
                    if sz > 1024 * 1024:
                        cand.append((sz, p))
                except OSError:
                    pass

    if not cand:
        raise RuntimeError(f"No flow file found near {in_dir}")
    cand.sort(reverse=True)
    return cand[0][1]


def robust_read_text(path: str, max_rows=None) -> pd.DataFrame:
    seps = [",", "\t", r"\s+"]
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(
                    path, sep=sep, engine="python", comment="#",
                    nrows=max_rows, on_bad_lines="skip",
                    encoding=enc, encoding_errors="replace",
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
    raise RuntimeError(f"Failed to parse {path}")


def pick_col(df: pd.DataFrame, keys):
    cols = {c.lower(): c for c in df.columns}
    for k in keys:
        if k.lower() in cols:
            return cols[k.lower()]
    for c in df.columns:
        for k in keys:
            if k.lower() in c.lower():
                return c
    return None


def build_labels(df, label_col, saddr_col):
    N = len(df)
    if label_col:
        lab = df[label_col].astype(str).fillna("").str.strip().str.lower()
        y = lab.str.contains("from-botnet", regex=False).astype(np.uint8).to_numpy()
        if y.sum() > 0:
            return y
    if saddr_col:
        src = df[saddr_col].astype(str).fillna("")
        y = src.isin(INFECTED_IPS).astype(np.uint8).to_numpy()
        return y
    return np.zeros(N, dtype=np.uint8)


def extract_features(df, exclude_cols):
    feat_cols = []
    priority = ["dur", "totpkts", "totbytes", "srcbytes", "sport", "dport"]
    for pc in priority:
        for c in df.columns:
            if c in exclude_cols or c in feat_cols:
                continue
            if pc in c.lower():
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().mean() > 0.3:
                    feat_cols.append(c)
                    break
    for c in df.columns:
        if c in exclude_cols or c in feat_cols:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.5:
            feat_cols.append(c)
    if not feat_cols:
        raise RuntimeError("No numeric features found!")
    X = np.stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
         for c in feat_cols],
        axis=1,
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, feat_cols


def build_knn_edges(df, saddr_col, daddr_col, proto_col, time_col, K):
    N = len(df)
    idx = np.arange(N, dtype=np.int32)

    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if t.notna().mean() >= 0.3:
            win = (t.astype("int64") // 10**9).fillna(0).to_numpy(dtype=np.int64)
        else:
            tnum = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)
            win = np.floor(tnum).astype(np.int64)
    else:
        win = np.arange(N, dtype=np.int64)

    def group_knn(col):
        if col is None:
            return [], []
        key = df[col].astype(str).fillna("").to_numpy()
        order = np.lexsort((win, key))
        keys_sorted = key[order]
        u_all, v_all = [], []
        for k in range(1, K + 1):
            a = order[:-k]
            b = order[k:]
            same = (keys_sorted[:-k] == keys_sorted[k:])
            aa, bb = a[same], b[same]
            if len(aa) > 0:
                u_all.append(aa)
                v_all.append(bb)
        if not u_all:
            return [], []
        return np.concatenate(u_all), np.concatenate(v_all)

    e1u, e1v = group_knn(saddr_col)
    e2u, e2v = group_knn(daddr_col)
    e3u, e3v = group_knn(proto_col)

    key_win = win.astype(str)
    order_w = np.lexsort((idx, key_win))
    keys_w = key_win[order_w]
    u4, v4 = [], []
    for k in range(1, K + 1):
        a = order_w[:-k]
        b = order_w[k:]
        same = (keys_w[:-k] == keys_w[k:])
        aa, bb = a[same], b[same]
        if len(aa) > 0:
            u4.append(aa)
            v4.append(bb)
    e4u = np.concatenate(u4) if u4 else np.array([], dtype=np.int32)
    e4v = np.concatenate(v4) if v4 else np.array([], dtype=np.int32)

    parts = [(e1u, e1v, "src"), (e2u, e2v, "dst"), (e3u, e3v, "proto"), (e4u, e4v, "time")]

    all_u, all_v = [], []
    for eu, ev, name in parts:
        if len(eu) > 0:
            all_u.append(eu)
            all_v.append(ev)
            print(f"[Edges] {name:8s}: {len(eu):>10,} directed edges", flush=True)

    if not all_u:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    u = np.concatenate(all_u).astype(np.int32)
    v = np.concatenate(all_v).astype(np.int32)
    uv = np.unique(np.stack([u, v], axis=1), axis=0)
    u_final = uv[:, 0].astype(np.int32)
    v_final = uv[:, 1].astype(np.int32)
    print(f"[Edges] Total before dedup: {len(u):>10,}", flush=True)
    print(f"[Edges] Total after dedup:  {len(u_final):>10,}", flush=True)
    return u_final, v_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_mat", type=str, required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    max_rows = None if args.max_rows <= 0 else int(args.max_rows)
    K = int(args.K)

    flow_path = pick_best_text_flow(args.in_dir)
    print(f"[CTU13] Using: {flow_path} ({os.path.getsize(flow_path)/1024/1024:.1f} MB)", flush=True)

    df = robust_read_text(flow_path, max_rows=max_rows)
    N = len(df)
    print(f"[CTU13] Loaded {N:,} flows", flush=True)

    label_col = pick_col(df, ["Label"])
    saddr_col = pick_col(df, ["SrcAddr", "srcaddr", "saddr"])
    daddr_col = pick_col(df, ["DstAddr", "dstaddr", "daddr"])
    proto_col = pick_col(df, ["Proto", "protocol"])
    time_col = pick_col(df, ["StartTime", "starttime", "time"])

    y = build_labels(df, label_col, saddr_col)
    print(f"[CTU13] Anomalies: {int(y.sum()):,} ({y.mean():.2%})", flush=True)

    exclude = {c for c in [label_col, saddr_col, daddr_col, proto_col, time_col] if c}
    X, feat_cols = extract_features(df, exclude)
    print(f"[CTU13] Features: {X.shape[1]} ({', '.join(feat_cols[:5])}...)", flush=True)

    u, v = build_knn_edges(df, saddr_col, daddr_col, proto_col, time_col, K)

    if time_col:
        t_dt = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if t_dt.notna().mean() >= 0.3:
            starttime = (t_dt.astype("int64") // 10**9).fillna(0).to_numpy(dtype=np.float64)
        else:
            starttime = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        starttime = np.arange(N, dtype=np.float64)

    os.makedirs(os.path.dirname(args.out_mat) or ".", exist_ok=True)
    savemat(args.out_mat, {
        "edge_u": u.reshape(-1, 1),
        "edge_v": v.reshape(-1, 1),
        "X": X,
        "y": y.reshape(-1, 1),
        "N_nodes": np.array([[N]], dtype=np.int64),
        "E_instances": np.array([[len(u)]], dtype=np.int64),
        "anom_ratio": np.array([[float(y.mean())]], dtype=np.float64),
        "starttime": starttime,
    })

    print(f"\n[CTU13] Saved: {args.out_mat}", flush=True)
    print(f"  Nodes: {N:,}", flush=True)
    print(f"  Edges: {len(u):,}", flush=True)
    print(f"  Anomalies: {int(y.sum()):,} ({y.mean():.4%})", flush=True)


if __name__ == "__main__":
    main()
