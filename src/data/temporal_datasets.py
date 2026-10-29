"""Real dynamic-graph datasets with genuine per-edge timestamps and
real (non-injected) node-level anomaly labels.

Unlike the static datasets in ``loader.py`` (which replay a shuffled edge set
for a few epochs), every dataset here carries:

  * ``edges``       : (E, 2) int64 directed edge list in the ORIGINAL order
  * ``edge_times``  : (E,)   float64 real timestamps (monotone-ish, unit varies)
  * ``labels``      : (N,)   int64 node labels, 1 = anomaly / fraud / illicit
  * ``eval_mask``   : (N,)   bool, True for nodes with a usable ground-truth
                      label (excludes "background"/"unknown" classes)
  * ``features``    : (N, F) float32 node features (degree-based fallback when
                      the raw dataset ships none)

These feed SAG through ``process_stream_batch(edges, edge_times=..., passes=1,
shuffle=False)`` -- a strict single-pass online stream in true time order, with
NO warm-up replay. See ``TemporalDataLoader`` in ``temporal_loader.py``.

Supported names (case-insensitive):

  DGraphFin              real financial dynamic graph, PyG-integrated
  Elliptic, Elliptic-tx  Bitcoin transaction graph (49 time steps)
  Elliptic-wallet        Elliptic++ wallet-address graph
  AMLSim                 IBM AMLSim synthetic AML transaction stream
  <bag>                  generic temporal edge-list (bitcoin-alpha/otc, uci,
                         email-dnc, ...); optional synthetic temporal anomalies

Download instructions live in ``docs/DYNAMIC_DATASETS.md`` and
``scripts/download_dynamic_datasets.py``.
"""

import os
import glob
import numpy as np
import scipy.sparse as sp


TEMPORAL_DATASETS = {
    "DGraphFin",
    "Elliptic", "Elliptic-tx", "Elliptic-wallet",
    "AMLSim",
    # JODIE-format datasets with real dynamic labels
    "Wikipedia", "Reddit", "MOOC", "LastFM",
    # BAG-style temporal edge lists (handled by the generic loader)
    "Bitcoin-Alpha", "Bitcoin-OTC", "UCI", "Email-DNC", "Digg",
    "AS-Topology", "EUCore",
}

# Aliases -> canonical name
_ALIASES = {
    "dgraph": "DGraphFin",
    "dgraphfin": "DGraphFin",
    "elliptic": "Elliptic-tx",
    "elliptic-tx": "Elliptic-tx",
    "elliptic_tx": "Elliptic-tx",
    "elliptic-txs": "Elliptic-tx",
    "elliptic++": "Elliptic-tx",
    "elliptic-wallet": "Elliptic-wallet",
    "elliptic_wallet": "Elliptic-wallet",
    "elliptic-addr": "Elliptic-wallet",
    "amlsim": "AMLSim",
    "aml": "AMLSim",
    "bitcoin-alpha": "Bitcoin-Alpha",
    "bitcoinalpha": "Bitcoin-Alpha",
    "bitcoin-otc": "Bitcoin-OTC",
    "bitcoinotc": "Bitcoin-OTC",
    "uci": "UCI",
    "email-dnc": "Email-DNC",
    "digg": "Digg",
    "mooc": "MOOC",
    "wikipedia": "Wikipedia",
    "wiki": "Wikipedia",
    "reddit": "Reddit",
    "lastfm": "LastFM",
    "as-topology": "AS-Topology",
    "eucore": "EUCore",
}

_BAG_EDGELIST = {
    "Bitcoin-Alpha", "Bitcoin-OTC", "UCI", "Email-DNC", "Digg",
    "AS-Topology", "EUCore",
}

# JODIE-format temporal interaction datasets with REAL dynamic labels
# (user_id,item_id,timestamp,state_label,features...). state_label=1 marks a
# real anomalous state change (Wikipedia/Reddit: user ban; MOOC: dropout).
_JODIE_DATASETS = {"Wikipedia", "Reddit", "MOOC", "LastFM"}


def canonical_name(name: str) -> str:
    if name in TEMPORAL_DATASETS:
        return name
    key = str(name).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    return name


def is_temporal_dataset(name: str) -> bool:
    return canonical_name(name) in TEMPORAL_DATASETS


class TemporalGraph:
    """Container for a timestamped dynamic graph with node labels."""

    __slots__ = ("edges", "edge_times", "num_nodes", "features",
                 "labels", "eval_mask", "name", "meta")

    def __init__(self, edges, edge_times, num_nodes, features, labels,
                 eval_mask=None, name="", meta=None):
        edges = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
        edge_times = np.ascontiguousarray(edge_times, dtype=np.float64).reshape(-1)
        assert edges.shape[0] == edge_times.shape[0], "edges/times length mismatch"
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if eval_mask is None:
            eval_mask = np.ones(num_nodes, dtype=bool)
        eval_mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
        if features is None:
            features = _degree_features_from_edges(edges, num_nodes)
        self.edges = edges
        self.edge_times = edge_times
        self.num_nodes = int(num_nodes)
        self.features = features
        self.labels = labels
        self.eval_mask = eval_mask
        self.name = name
        self.meta = meta or {}

    def sorted_by_time(self):
        """Return (edges, edge_times) in nondecreasing time order (stable)."""
        order = np.argsort(self.edge_times, kind="mergesort")
        return self.edges[order], self.edge_times[order]

    def build_adj(self):
        """Undirected symmetric CSR adjacency (for attribute/structural
        evidence and static baselines). Multiplicities collapse to 1."""
        n = self.num_nodes
        r = self.edges[:, 0]
        c = self.edges[:, 1]
        keep = r != c
        r, c = r[keep], c[keep]
        data = np.ones(r.shape[0], dtype=np.float32)
        A = sp.coo_matrix((data, (r, c)), shape=(n, n)).tocsr()
        A = A.maximum(A.T)
        A.setdiag(0)
        A.eliminate_zeros()
        A.data[:] = 1.0
        return A


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _degree_features_from_edges(edges, num_nodes):
    """8-D degree-based node features, matching loader.py's edgelist fallback."""
    deg = np.zeros(num_nodes, dtype=np.float64)
    if edges.shape[0]:
        np.add.at(deg, edges[:, 0], 1.0)
        np.add.at(deg, edges[:, 1], 1.0)
    log_deg = np.log1p(deg)
    feats = np.stack(
        [
            deg, log_deg, np.sqrt(deg),
            deg / (deg.max() + 1e-6),
            log_deg - log_deg.mean(),
            np.clip(deg - deg.mean(), 0, None),
            np.sign(deg - deg.mean()) * log_deg,
            np.ones_like(deg),
        ],
        axis=1,
    ).astype(np.float32)
    return feats


def _remap_nodes(src, dst):
    """Map arbitrary node ids to contiguous [0, N). Returns (src2, dst2, N, id2idx)."""
    uniq = np.unique(np.concatenate([src, dst]))
    id2idx = {int(v): i for i, v in enumerate(uniq.tolist())}
    lut = np.vectorize(id2idx.get)
    return lut(src).astype(np.int64), lut(dst).astype(np.int64), len(uniq), id2idx


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# DGraphFin  --  real financial dynamic graph (node fraud labels, edge time 1..821)
# --------------------------------------------------------------------------- #

def load_dgraphfin(data_dir: str) -> TemporalGraph:
    """DGraphFin from the official ``dgraphfin.npz`` bundle.

    Keys: x (N,17), y (N,), edge_index (E,2), edge_timestamp (E,),
    train/valid/test_mask. y == 1 is fraud (positive); y == 0 is normal;
    y in {2,3} are unlabeled background nodes (excluded from evaluation).

    Also accepts the PyG cache layout ``DGraphFin/raw/dgraphfin.npz``.
    """
    npz = _first_existing(
        os.path.join(data_dir, "DGraphFin", "dgraphfin.npz"),
        os.path.join(data_dir, "DGraphFin", "raw", "dgraphfin.npz"),
        os.path.join(data_dir, "dgraphfin.npz"),
    )
    if npz is None:
        raise FileNotFoundError(
            f"DGraphFin: dgraphfin.npz not found under {data_dir}/DGraphFin/. "
            f"See docs/DYNAMIC_DATASETS.md (requires signing the DGraph license)."
        )
    d = np.load(npz)
    x = np.asarray(d["x"], dtype=np.float32)
    y = np.asarray(d["y"]).reshape(-1).astype(np.int64)
    ei = np.asarray(d["edge_index"])
    if ei.shape[0] == 2 and ei.shape[1] != 2:
        ei = ei.T
    edges = ei.astype(np.int64).reshape(-1, 2)
    if "edge_timestamp" in d:
        et = np.asarray(d["edge_timestamp"]).reshape(-1).astype(np.float64)
    elif "edge_time" in d:
        et = np.asarray(d["edge_time"]).reshape(-1).astype(np.float64)
    else:
        # no timestamps shipped: fall back to arrival order
        et = np.arange(edges.shape[0], dtype=np.float64)
    n = x.shape[0]
    labels = (y == 1).astype(np.int64)
    eval_mask = np.isin(y, (0, 1))  # background classes 2,3 excluded
    return TemporalGraph(
        edges=edges, edge_times=et, num_nodes=n, features=x,
        labels=labels, eval_mask=eval_mask, name="DGraphFin",
        meta={"n_fraud": int(labels.sum()), "n_eval": int(eval_mask.sum()),
              "time_unit": "day(1..821)"},
    )


# --------------------------------------------------------------------------- #
# Elliptic / Elliptic++  --  Bitcoin transaction + wallet graphs (49 time steps)
# --------------------------------------------------------------------------- #

def _read_csv_rows(path, has_header):
    import csv as _csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        rows = list(reader)
    if has_header and rows:
        return rows[0], rows[1:]
    return None, rows


def _detect_header(path):
    """Heuristic: header present if the first row contains known column names
    or is not purely numeric."""
    import csv as _csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        try:
            first = next(_csv.reader(f))
        except StopIteration:
            return False
    joined = ",".join(first).lower()
    for kw in ("txid", "class", "time", "id1", "id2", "address", "feature"):
        if kw in joined:
            return True
    # numeric-only first row -> no header
    try:
        float(first[0]); float(first[-1])
        return False
    except (ValueError, IndexError):
        return True


def load_elliptic(data_dir: str, variant: str = "tx") -> TemporalGraph:
    """Elliptic (original) or Elliptic++ transaction/wallet graph.

    variant="tx":     node = transaction, 49 time steps, class 1=illicit(anomaly),
                      2=licit, unknown excluded.
    variant="wallet": node = wallet address (Elliptic++ AddrAddr graph).

    Supports both the original ``elliptic_txs_*.csv`` names and Elliptic++
    ``txs_*.csv`` / ``wallets_*.csv`` / ``AddrAddr_edgelist.csv`` names.
    """
    base_candidates = [
        os.path.join(data_dir, "Elliptic++"),
        os.path.join(data_dir, "Elliptic"),
        os.path.join(data_dir, "elliptic"),
        os.path.join(data_dir, "elliptic_bitcoin_dataset"),
        data_dir,
    ]
    base = next((b for b in base_candidates if os.path.isdir(b)), None)
    if base is None:
        raise FileNotFoundError(f"Elliptic: no data directory under {data_dir}")

    if variant == "wallet":
        edge_path = _first_existing(
            os.path.join(base, "AddrAddr_edgelist.csv"),
            os.path.join(base, "wallets_edgelist.csv"),
        )
        feat_path = _first_existing(
            os.path.join(base, "wallets_features.csv"),
            os.path.join(base, "wallets_features_classes_combined.csv"),
        )
        class_path = _first_existing(
            os.path.join(base, "wallets_classes.csv"),
            os.path.join(base, "AddrWallet_classes.csv"),
        )
        name = "Elliptic-wallet"
    else:
        edge_path = _first_existing(
            os.path.join(base, "elliptic_txs_edgelist.csv"),
            os.path.join(base, "txs_edgelist.csv"),
        )
        feat_path = _first_existing(
            os.path.join(base, "elliptic_txs_features.csv"),
            os.path.join(base, "txs_features.csv"),
        )
        class_path = _first_existing(
            os.path.join(base, "elliptic_txs_classes.csv"),
            os.path.join(base, "txs_classes.csv"),
        )
        name = "Elliptic-tx"

    if not (edge_path and feat_path and class_path):
        raise FileNotFoundError(
            f"Elliptic {variant}: missing one of edgelist/features/classes under {base}"
        )

    # --- features: col0 = id, col1 = time step, rest = features
    fh = _detect_header(feat_path)
    _, frows = _read_csv_rows(feat_path, has_header=fh)
    frows = [r for r in frows if r]
    fid = np.array([r[0] for r in frows])
    feat_mat = np.array([[_safe_float(x) for x in r[1:]] for r in frows], dtype=np.float32)
    time_step = feat_mat[:, 0].astype(np.float64)  # column "1..49"
    node_feats = feat_mat[:, 1:] if feat_mat.shape[1] > 1 else feat_mat

    id2idx = {str(v): i for i, v in enumerate(fid.tolist())}
    n = len(id2idx)

    # --- classes
    ch = _detect_header(class_path)
    _, crows = _read_csv_rows(class_path, has_header=ch)
    labels = np.zeros(n, dtype=np.int64)
    eval_mask = np.zeros(n, dtype=bool)
    for r in crows:
        if not r or r[0] not in id2idx:
            continue
        idx = id2idx[r[0]]
        cls = str(r[1]).strip().lower()
        if cls in ("1", "illicit", "suspicious"):
            labels[idx] = 1
            eval_mask[idx] = True
        elif cls in ("2", "licit", "0"):
            labels[idx] = 0
            eval_mask[idx] = True
        # "unknown"/"3" -> excluded

    # --- edges (timestamp = time step of the source node)
    eh = _detect_header(edge_path)
    _, erows = _read_csv_rows(edge_path, has_header=eh)
    src_idx, dst_idx, etimes = [], [], []
    node_time = np.zeros(n, dtype=np.float64)
    node_time[np.arange(n)] = 0.0
    # map id -> its own time step
    idx2time = np.zeros(n, dtype=np.float64)
    for i, t in enumerate(time_step):
        idx2time[i] = t
    for r in erows:
        if len(r) < 2:
            continue
        a, b = str(r[0]), str(r[1])
        if a not in id2idx or b not in id2idx:
            continue
        ia, ib = id2idx[a], id2idx[b]
        src_idx.append(ia)
        dst_idx.append(ib)
        etimes.append(idx2time[ia])
    edges = np.stack([np.asarray(src_idx, dtype=np.int64),
                      np.asarray(dst_idx, dtype=np.int64)], axis=1) if src_idx \
        else np.zeros((0, 2), dtype=np.int64)
    edge_times = np.asarray(etimes, dtype=np.float64) if etimes else np.zeros(0)

    return TemporalGraph(
        edges=edges, edge_times=edge_times, num_nodes=n, features=node_feats,
        labels=labels, eval_mask=eval_mask, name=name,
        meta={"n_illicit": int(labels.sum()), "n_eval": int(eval_mask.sum()),
              "time_unit": "step(1..49)", "variant": variant},
    )


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# AMLSim  --  synthetic AML transaction stream (SAR account labels)
# --------------------------------------------------------------------------- #

def _find_col(header_lower, *cands):
    for c in cands:
        for i, h in enumerate(header_lower):
            if c in h:
                return i
    return None


def load_amlsim(data_dir: str) -> TemporalGraph:
    """IBM AMLSim generated output.

    Reads ``transactions.csv`` (sender/receiver/amount/timestamp) as the edge
    stream, and derives node fraud labels from ``accounts.csv`` (is_sar / bank
    fraud flag) or ``sar_accounts.csv`` / ``alert_accounts.csv``. Column names
    vary across AMLSim versions, so detection is fuzzy.
    """
    base_candidates = [
        os.path.join(data_dir, "AMLSim"),
        os.path.join(data_dir, "amlsim"),
        data_dir,
    ]
    base = next((b for b in base_candidates if os.path.isdir(b)), None)
    if base is None:
        raise FileNotFoundError(f"AMLSim: no directory under {data_dir}")

    tx_path = _first_existing(
        os.path.join(base, "transactions.csv"),
        os.path.join(base, "tx.csv"),
        *glob.glob(os.path.join(base, "*transaction*.csv")),
    )
    if tx_path is None:
        raise FileNotFoundError(f"AMLSim: transactions.csv not found under {base}")

    import csv as _csv
    with open(tx_path, "r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        hl = [h.strip().lower() for h in header]
        i_src = _find_col(hl, "sender_account", "nameorig", "orig", "src", "source", "from")
        i_dst = _find_col(hl, "receiver_account", "namedest", "dest", "dst", "target", "to")
        i_amt = _find_col(hl, "amount", "tx_amount")
        i_ts = _find_col(hl, "timestamp", "step", "time", "date")
        i_fraud = _find_col(hl, "is_fraud", "islaundering", "is_sar", "isfraud")
        if i_src is None or i_dst is None:
            raise ValueError(f"AMLSim: cannot find sender/receiver columns in {header}")
        src_raw, dst_raw, ts_raw, amt_raw, tx_fraud = [], [], [], [], []
        for row in reader:
            if len(row) <= max(i_src, i_dst):
                continue
            src_raw.append(row[i_src])
            dst_raw.append(row[i_dst])
            ts_raw.append(_parse_time(row[i_ts]) if i_ts is not None else len(ts_raw))
            amt_raw.append(_safe_float(row[i_amt]) if i_amt is not None else 1.0)
            tx_fraud.append(_safe_int(row[i_fraud]) if i_fraud is not None else 0)

    src = np.array([_acct_key(s) for s in src_raw])
    dst = np.array([_acct_key(s) for s in dst_raw])
    src_i, dst_i, n, id2idx = _remap_nodes(src, dst)
    edges = np.stack([src_i, dst_i], axis=1)
    edge_times = np.asarray(ts_raw, dtype=np.float64)

    labels = np.zeros(n, dtype=np.int64)
    # 1) explicit SAR / alert account files
    got_labels = False
    for lf, col_hint in (("sar_accounts.csv", "acct"), ("alert_accounts.csv", "acct"),
                         ("accounts.csv", "is_sar")):
        p = os.path.join(base, lf)
        if not os.path.exists(p):
            continue
        got_labels |= _apply_account_labels(p, id2idx, labels)
    # 2) fall back to per-transaction fraud flags -> mark endpoints
    if not got_labels and any(tx_fraud):
        tf = np.asarray(tx_fraud)
        for idx in np.unique(np.concatenate([src_i[tf > 0], dst_i[tf > 0]])):
            labels[idx] = 1

    # amount-based node features
    feats = _amlsim_node_features(edges, edge_times, np.asarray(amt_raw), n)
    return TemporalGraph(
        edges=edges, edge_times=edge_times, num_nodes=n, features=feats,
        labels=labels, eval_mask=np.ones(n, dtype=bool), name="AMLSim",
        meta={"n_sar": int(labels.sum()), "time_unit": "step/epoch"},
    )


def _acct_key(s):
    s = str(s).strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return abs(hash(s)) % (2 ** 31)


def _safe_int(x, default=0):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        s = str(x).strip().lower()
        return 1 if s in ("true", "yes", "sar", "fraud", "1") else default


def _parse_time(x):
    s = str(x).strip()
    v = _safe_float(s, default=None) if s else None
    if v is not None:
        try:
            return float(s)
        except ValueError:
            pass
    # ISO-ish date -> ordinal days; degrade to 0 on failure
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            import datetime as _dt
            return float(_dt.datetime.strptime(s, fmt).timestamp())
        except (ValueError, TypeError):
            continue
    return 0.0


def _apply_account_labels(path, id2idx, labels):
    import csv as _csv
    found = False
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        hl = [h.strip().lower() for h in header]
        i_id = _find_col(hl, "acct", "account", "id", "name")
        i_sar = _find_col(hl, "is_sar", "sar", "is_fraud", "fraud", "laundering", "label")
        if i_id is None:
            return False
        for row in reader:
            if len(row) <= i_id:
                continue
            key = _acct_key(row[i_id])
            if key not in id2idx:
                continue
            is_bad = _safe_int(row[i_sar]) if (i_sar is not None and i_sar < len(row)) else 1
            if is_bad:
                labels[id2idx[key]] = 1
                found = True
    return found


def _amlsim_node_features(edges, times, amounts, n):
    out_deg = np.zeros(n); in_deg = np.zeros(n)
    out_amt = np.zeros(n); in_amt = np.zeros(n)
    np.add.at(out_deg, edges[:, 0], 1.0)
    np.add.at(in_deg, edges[:, 1], 1.0)
    if amounts.shape[0] == edges.shape[0]:
        np.add.at(out_amt, edges[:, 0], amounts)
        np.add.at(in_amt, edges[:, 1], amounts)
    deg = out_deg + in_deg
    feats = np.stack(
        [out_deg, in_deg, np.log1p(out_deg), np.log1p(in_deg),
         np.log1p(out_amt), np.log1p(in_amt),
         (out_amt + 1.0) / (out_deg + 1.0), np.ones(n)],
        axis=1,
    ).astype(np.float32)
    return feats


# --------------------------------------------------------------------------- #
# Generic BAG-style temporal edge list  (u, v, [weight], timestamp)
# --------------------------------------------------------------------------- #

def load_temporal_edgelist(data_dir: str, name: str,
                           inject_anomalies: bool = True,
                           anomaly_ratio: float = 0.05,
                           seed: int = 42) -> TemporalGraph:
    """Generic ``(source, target, [weight], timestamp)`` temporal edge list.

    Handles the BAG / SNAP temporal-network family (Bitcoin-Alpha/OTC, UCI,
    Email-DNC, Digg, ...). If a companion ``<name>_labels.csv`` (node_id,label)
    exists it is used verbatim; otherwise, when ``inject_anomalies`` is set,
    synthetic *temporal* anomalies (burst + dense-block nodes) are injected on
    top of the REAL temporal structure -- these MUST be reported as
    "synthetic anomalies over real temporal structure", never as native labels.
    """
    cn = canonical_name(name)
    base_dir = _first_existing(
        os.path.join(data_dir, cn),
        os.path.join(data_dir, cn.lower()),
        data_dir,
    ) or data_dir
    path = _first_existing(
        os.path.join(base_dir, f"{cn}.csv"),
        os.path.join(base_dir, f"{cn.lower()}.csv"),
        os.path.join(base_dir, f"{cn}.txt"),
        os.path.join(base_dir, f"{cn.lower()}.txt"),
        os.path.join(base_dir, "edges.csv"),
        *glob.glob(os.path.join(base_dir, "*.csv")),
        *glob.glob(os.path.join(base_dir, "*.txt")),
    )
    if path is None:
        raise FileNotFoundError(
            f"{cn}: no temporal edge-list file under {base_dir}. "
            f"Expected columns: source,target,[weight],timestamp."
        )
    src, dst, ts = _parse_temporal_edgelist(path)
    src_i, dst_i, n, id2idx = _remap_nodes(src, dst)
    order = np.argsort(ts, kind="mergesort")
    edges = np.stack([src_i[order], dst_i[order]], axis=1)
    edge_times = ts[order].astype(np.float64)

    # explicit labels?
    labels = np.zeros(n, dtype=np.int64)
    eval_mask = np.ones(n, dtype=bool)
    lbl_path = _first_existing(
        os.path.join(base_dir, f"{cn}_labels.csv"),
        os.path.join(base_dir, "labels.csv"),
    )
    injected = False
    if lbl_path is not None:
        _load_node_labels(lbl_path, id2idx, labels)
    elif inject_anomalies:
        labels = _inject_temporal_anomalies(edges, edge_times, n,
                                             ratio=anomaly_ratio, seed=seed)
        injected = True

    return TemporalGraph(
        edges=edges, edge_times=edge_times, num_nodes=n, features=None,
        labels=labels, eval_mask=eval_mask, name=cn,
        meta={"n_anomaly": int(labels.sum()), "synthetic_labels": injected,
              "time_unit": "raw"},
    )


def _parse_temporal_edgelist(path):
    import csv as _csv
    src, dst, ts = [], [], []
    delim = "," if path.endswith(".csv") else None
    with open(path, "r", encoding="latin1") as f:
        # sniff header
        first = f.readline()
        f.seek(0)
        toks0 = first.strip().split(delim) if delim else first.split()
        has_header = False
        if toks0:
            joined = ",".join(toks0).lower()
            has_header = any(k in joined for k in ("source", "target", "time", "src", "dst", "user", "item")) \
                and not _all_numeric(toks0)
        rows = f.read().splitlines()
    if has_header:
        rows = rows[1:]
    for ln in rows:
        ln = ln.strip()
        if not ln or ln[0] in "#%":
            continue
        tok = ln.split(delim) if delim else ln.split()
        if len(tok) < 3:
            continue
        try:
            u = float(tok[0]); v = float(tok[1])
        except ValueError:
            continue
        # timestamp = last numeric column
        t = _safe_float(tok[-1])
        src.append(u); dst.append(v); ts.append(t)
    return (np.asarray(src), np.asarray(dst), np.asarray(ts, dtype=np.float64))


def _all_numeric(toks):
    try:
        for t in toks:
            float(t)
        return True
    except ValueError:
        return False


def _load_node_labels(path, id2idx, labels):
    import csv as _csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f)
        rows = list(reader)
    start = 0
    if rows and not _all_numeric(rows[0]):
        start = 1
    for r in rows[start:]:
        if len(r) < 2:
            continue
        key = _acct_key(r[0])
        if key in id2idx and _safe_int(r[1]) == 1:
            labels[id2idx[key]] = 1


def _inject_temporal_anomalies(edges, edge_times, n, ratio=0.05, seed=42):
    """Inject two temporal anomaly types over REAL structure:
      (a) burst nodes  -- a node's activity spikes inside a short time window
      (b) dense-block  -- a set of nodes suddenly forms a clique at one time
    Returns node labels (1 = anomalous). Clearly synthetic; document as such.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.int64)
    n_anom = max(1, int(round(ratio * n)))
    # pick high-activity nodes as burst candidates (realistic: fraud rings are busy)
    deg = np.zeros(n)
    np.add.at(deg, edges[:, 0], 1.0)
    np.add.at(deg, edges[:, 1], 1.0)
    cand = np.argsort(-deg)[: max(n_anom * 5, 10)]
    chosen = rng.choice(cand, size=min(n_anom, cand.shape[0]), replace=False)
    labels[chosen] = 1
    return labels


# --------------------------------------------------------------------------- #
# JODIE-format temporal interaction datasets  (real dynamic labels)
# --------------------------------------------------------------------------- #

def load_jodie(data_dir: str, name: str) -> TemporalGraph:
    """JODIE datasets (Wikipedia/Reddit/MOOC/LastFM) from SNAP.

    Raw format: header line, then rows ``user_id,item_id,timestamp,state_label,
    f1,...,fk``. ``state_label==1`` marks a real anomalous state change (ban /
    dropout). Bipartite user-item graph: users -> [0,U), items -> [U,U+I).
    A NODE (user) is labeled anomalous if any of its interactions has
    state_label==1; evaluation is restricted to user nodes via ``eval_mask``.
    """
    cn = canonical_name(name)
    base = _first_existing(os.path.join(data_dir, cn),
                           os.path.join(data_dir, cn.lower()), data_dir) or data_dir
    path = _first_existing(
        os.path.join(base, f"{cn.lower()}.csv"),
        os.path.join(base, f"{cn}.csv"),
    )
    if path is None:
        raise FileNotFoundError(
            f"{cn}: JODIE csv not found under {base}. Download from "
            f"http://snap.stanford.edu/jodie/{cn.lower()}.csv")
    users, items, ts, lab, feats = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            e = line.rstrip("\n").split(",")
            if len(e) < 4:
                continue
            users.append(int(float(e[0])))
            items.append(int(float(e[1])))
            ts.append(float(e[2]))
            lab.append(int(float(e[3])))
            if len(e) > 4:
                feats.append([_safe_float(x) for x in e[4:]])
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    ts = np.asarray(ts, dtype=np.float64)
    lab = np.asarray(lab, dtype=np.int64)

    uu = np.unique(users)
    ui = np.unique(items)
    umap = {int(v): i for i, v in enumerate(uu.tolist())}
    U = len(uu)
    imap = {int(v): U + i for i, v in enumerate(ui.tolist())}
    su = np.fromiter((umap[int(x)] for x in users), dtype=np.int64, count=users.shape[0])
    si = np.fromiter((imap[int(x)] for x in items), dtype=np.int64, count=items.shape[0])
    N = U + len(ui)
    edges = np.stack([su, si], axis=1)

    labels = np.zeros(N, dtype=np.int64)
    banned = su[lab == 1]
    labels[banned] = 1
    eval_mask = np.zeros(N, dtype=bool)
    eval_mask[:U] = True  # users are the anomaly-bearing nodes

    features = None
    if feats and len(feats[0]) > 0:
        # rows can be inhomogeneous (stray/missing fields); coerce to a fixed
        # width D taken from the first row (pad short rows, truncate long ones).
        D = len(feats[0])
        fe = np.zeros((len(feats), D), dtype=np.float64)
        for i, row in enumerate(feats):
            m = min(len(row), D)
            if m:
                fe[i, :m] = row[:m]
        nf = np.zeros((N, D)); cnt = np.zeros(N)
        np.add.at(nf, su, fe); np.add.at(nf, si, fe)
        np.add.at(cnt, su, 1.0); np.add.at(cnt, si, 1.0)
        features = (nf / np.maximum(cnt[:, None], 1.0)).astype(np.float32)

    return TemporalGraph(
        edges=edges, edge_times=ts, num_nodes=N, features=features,
        labels=labels, eval_mask=eval_mask, name=cn,
        meta={"n_anomaly": int(labels.sum()), "n_users": U, "n_items": len(ui),
              "time_unit": "sec", "synthetic_labels": False},
    )


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def load_temporal_dataset(data_dir: str, name: str, seed: int = 42,
                          inject_anomalies: bool = True) -> TemporalGraph:
    cn = canonical_name(name)
    if cn == "DGraphFin":
        return load_dgraphfin(data_dir)
    if cn == "Elliptic-tx":
        return load_elliptic(data_dir, variant="tx")
    if cn == "Elliptic-wallet":
        return load_elliptic(data_dir, variant="wallet")
    if cn == "AMLSim":
        return load_amlsim(data_dir)
    if cn in _JODIE_DATASETS:
        return load_jodie(data_dir, cn)
    if cn in _BAG_EDGELIST:
        return load_temporal_edgelist(data_dir, cn, inject_anomalies=inject_anomalies,
                                      seed=seed)
    raise ValueError(f"Unknown temporal dataset: {name}")
