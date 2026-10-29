#!/usr/bin/env python3
"""
CTU-13 v3 builder: keeps the K-NN graph (K=3, 4 sources) from v2, but ENRICHES
node features with per-source aggregate statistics.

Rationale: 8 raw flow features (Dur/TotPkts/TotBytes/SrcBytes/Sport/Dport/sTos/dTos)
only weakly discriminate botnet vs background. The 3 botnet IPs send 98.8% of
anomalies. Per-source aggregate features (burst rate, dst-port diversity, etc.)
are unsupervised features that strongly correlate with botnet behavior.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.io import savemat


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_ctu13_groundtruth import (
    pick_best_text_flow,
    robust_read_text,
    pick_col,
    build_labels,
    build_knn_edges,
)


def compute_per_source_features(df, saddr_col, daddr_col, dport_col, time_col, dur_col, srcbytes_col):
    """Compute per-source aggregate features and broadcast to each flow."""
    n = len(df)

    src = df[saddr_col].astype(str).fillna("")
    counts = src.value_counts()
    src_count = src.map(counts).to_numpy(dtype=np.float32)

    if daddr_col is not None:
        dst = df[daddr_col].astype(str).fillna("")
        dst_uniq = dst.groupby(src).transform("nunique").to_numpy(dtype=np.float32)
    else:
        dst_uniq = np.zeros(n, dtype=np.float32)

    if dport_col is not None:
        dport_str = df[dport_col].astype(str).fillna("")
        dport_uniq = dport_str.groupby(src).transform("nunique").to_numpy(dtype=np.float32)
    else:
        dport_uniq = np.zeros(n, dtype=np.float32)

    if dur_col is not None:
        dur = pd.to_numeric(df[dur_col], errors="coerce").fillna(0.0)
        src_dur_mean = dur.groupby(src).transform("mean").to_numpy(dtype=np.float32)
        src_dur_std = dur.groupby(src).transform("std").fillna(0).to_numpy(dtype=np.float32)
    else:
        src_dur_mean = np.zeros(n, dtype=np.float32)
        src_dur_std = np.zeros(n, dtype=np.float32)

    if srcbytes_col is not None:
        sb = pd.to_numeric(df[srcbytes_col], errors="coerce").fillna(0.0)
        src_sb_mean = sb.groupby(src).transform("mean").to_numpy(dtype=np.float32)
        src_sb_std = sb.groupby(src).transform("std").fillna(0).to_numpy(dtype=np.float32)
    else:
        src_sb_mean = np.zeros(n, dtype=np.float32)
        src_sb_std = np.zeros(n, dtype=np.float32)

    if time_col is not None:
        t_dt = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if t_dt.notna().mean() >= 0.3:
            t_sec = (t_dt.astype("int64") // 10**9).fillna(0)
        else:
            t_sec = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
        t_sec_arr = t_sec.to_numpy(dtype=np.float64)
        src_t_min = t_sec.groupby(src).transform("min").to_numpy(dtype=np.float64)
        src_t_max = t_sec.groupby(src).transform("max").to_numpy(dtype=np.float64)
        span = np.maximum(src_t_max - src_t_min, 1.0)
        src_burst_rate = (src_count / span.astype(np.float32)).astype(np.float32)
    else:
        src_burst_rate = np.zeros(n, dtype=np.float32)

    src_dst_ratio = (src_count / np.maximum(dst_uniq, 1.0)).astype(np.float32)
    src_port_ratio = (src_count / np.maximum(dport_uniq, 1.0)).astype(np.float32)

    extra = np.stack([
        np.log1p(src_count),
        np.log1p(dst_uniq),
        np.log1p(dport_uniq),
        np.log1p(np.maximum(src_dur_mean, 0)),
        np.log1p(np.maximum(src_dur_std, 0)),
        np.log1p(np.maximum(src_sb_mean, 0)),
        np.log1p(np.maximum(src_sb_std, 0)),
        np.log1p(src_burst_rate),
        np.log1p(src_dst_ratio),
        np.log1p(src_port_ratio),
    ], axis=1).astype(np.float32)
    return extra, ["log_src_count", "log_dst_uniq", "log_dport_uniq",
                   "log_dur_mean", "log_dur_std",
                   "log_sb_mean", "log_sb_std",
                   "log_burst_rate", "log_src_dst_ratio", "log_src_port_ratio"]


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
    print(f"[CTU13-v3] Using: {flow_path} ({os.path.getsize(flow_path)/1024/1024:.1f} MB)", flush=True)
    df = robust_read_text(flow_path, max_rows=max_rows)
    n = len(df)
    print(f"[CTU13-v3] Loaded {n:,} flows", flush=True)

    label_col = pick_col(df, ["Label"])
    saddr_col = pick_col(df, ["SrcAddr", "srcaddr", "saddr"])
    daddr_col = pick_col(df, ["DstAddr", "dstaddr", "daddr"])
    proto_col = pick_col(df, ["Proto", "protocol"])
    time_col = pick_col(df, ["StartTime", "starttime", "time"])
    dport_col = pick_col(df, ["Dport", "dport"])
    dur_col = pick_col(df, ["Dur", "dur"])
    srcbytes_col = pick_col(df, ["SrcBytes", "srcbytes"])

    y = build_labels(df, label_col, saddr_col)
    print(f"[CTU13-v3] Anomalies: {int(y.sum()):,} ({y.mean():.2%})", flush=True)

    # Behavioral source features that distinguish scanning from normal traffic
    src = df[saddr_col].astype(str).fillna("")
    src_count = src.map(src.value_counts()).to_numpy(dtype=np.float32)
    if daddr_col is not None:
        dst_uniq = df[daddr_col].astype(str).fillna("").groupby(src).transform("nunique").to_numpy(dtype=np.float32)
    else:
        dst_uniq = np.ones(n, dtype=np.float32)
    if dport_col is not None:
        dport_uniq = df[dport_col].astype(str).fillna("").groupby(src).transform("nunique").to_numpy(dtype=np.float32)
    else:
        dport_uniq = np.ones(n, dtype=np.float32)
    dst_per_n = dst_uniq / np.maximum(src_count, 1)
    dport_per_n = dport_uniq / np.maximum(src_count, 1)
    log_n = np.log1p(src_count).astype(np.float32)
    X = np.stack([dst_per_n, dport_per_n, log_n], axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[CTU13-v3] Features: dst/n, dport/n, log(src_count). Shape={X.shape}", flush=True)

    # Build edges (same K-NN as v2)
    u, v = build_knn_edges(df, saddr_col, daddr_col, proto_col, time_col, K)

    if time_col:
        t_dt = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if t_dt.notna().mean() >= 0.3:
            starttime = (t_dt.astype("int64") // 10**9).fillna(0).to_numpy(dtype=np.float64)
        else:
            starttime = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    else:
        starttime = np.arange(n, dtype=np.float64)

    os.makedirs(os.path.dirname(args.out_mat) or ".", exist_ok=True)
    savemat(args.out_mat, {
        "edge_u": u.reshape(-1, 1),
        "edge_v": v.reshape(-1, 1),
        "X": X,
        "y": y.reshape(-1, 1),
        "N_nodes": np.array([[n]], dtype=np.int64),
        "E_instances": np.array([[len(u)]], dtype=np.int64),
        "anom_ratio": np.array([[float(y.mean())]], dtype=np.float64),
        "starttime": starttime,
    })
    print(f"[CTU13-v3] Saved: {args.out_mat}", flush=True)
    print(f"  Nodes: {n:,}  Edges: {len(u):,}  Features: {X.shape[1]}  Anomalies: {int(y.sum()):,} ({y.mean():.4%})", flush=True)


if __name__ == "__main__":
    main()
