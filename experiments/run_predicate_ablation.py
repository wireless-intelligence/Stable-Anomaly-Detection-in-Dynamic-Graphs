"""SAG predicate ablation experiment (revision).

This ablation examines how SAG detects attribute-only anomalies if the dominant
evidence path is edge/frequency-based.  We sweep seven variants per dataset
that each restrict the predicate space (or the selection mechanism over it),
holding all other randomness fixed.

Variants
--------
full                    : all predicates + Evidence Selection  (paper Eq. 4-7)
edge_only               : only the streaming-state predicate r^(state)
                          - the edge / frequency / temporal-interaction signal
structural_only         : only the structural-statistics predicate r^(struct)
attribute_only          : only attribute predicates r^(attr) (gcn AE / iforest /
                          pca-mahal blend, depending on dataset config)
no_attribute            : full SAG without attribute predicates
no_exposure             : full SAG without exposure scaling in state evolution
no_evidence_selection   : full predicate set max-pooled WITHOUT the C1/C2/C3
                          rank-consistency filter (paper Eq. 5-7)

Outputs
-------
canonical (latest run):
  outputs/results/revision/predicate_ablation_raw.csv
  outputs/results/revision/predicate_ablation_summary.csv
per-run timestamped copy under
  outputs/results/revision/predicate_ablation_<ts>/
"""

import argparse
import copy
import csv
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from configs import Config, get_dataset_config
from src.data import DataLoader
from src.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_macro_f1, compute_balanced_auprc_f1,
)
from src.models import create_sag_model
from src.utils import set_seed, save_json, setup_logger


# ---------------------------------------------------------------------------
# Variant catalogue:  (key) -> (display name, fusion_mode_override, ablation_key)
# - fusion_mode_override=None means "keep dataset's native fusion_mode"
# - ablation_key=None means "ablation=full" (no architectural modification)
# ---------------------------------------------------------------------------
VARIANTS = [
    ("full",                    "SAG-full",                  "auto",                  None),
    ("edge_only",               "SAG-edge-only",             "h_only",                None),
    ("structural_only",         "SAG-structural-only",       "struct_only",           None),
    ("attribute_only",          "SAG-attribute-only",        "attr_only",             None),
    ("no_attribute",            "SAG-no-attribute",          "auto",                  "no_attribute"),
    ("no_exposure",             "SAG-no-exposure",           None,                    "no_exposure"),
    ("no_evidence_selection",   "SAG-no-evidence-selection", "no_evidence_selection", None),
]
VARIANT_KEYS = [v[0] for v in VARIANTS]
VARIANT_BY_KEY = {v[0]: v for v in VARIANTS}

DEFAULT_DATASETS = ("Cora", "Citeseer", "Pubmed", "BlogCatalog", "YelpReviews", "Enron")
DEFAULT_SEEDS = tuple(range(10))
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Per-run config shim (Config is a singleton; never mutate it directly)
# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc: dict, seed: int, fusion_mode_override, ablation_key):
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
    if fusion_mode_override is not None:
        cfg.fusion_mode = fusion_mode_override
    return cfg


# ---------------------------------------------------------------------------
# Applicability rules
# ---------------------------------------------------------------------------
def variant_applicable(dataset: str, variant_key: str, dc: dict) -> tuple:
    """Return (applicable: bool, reason_if_not: str).

    Rules:
      * attribute_only / no_attribute / structural_only require a usable
        attribute or structural source. SAG always has structural evidence
        (computed from adj), so structural_only is always applicable.
      * For datasets whose native fusion_mode is iforest_only (Amazon-family)
        the attribute path is via iForest and pca_mahal — still classified as
        an attribute predicate for ablation purposes, so attribute_only /
        no_attribute remain applicable.
    """
    return True, ""


def run_one(dataset: str, variant_key: str, seed: int, dc: dict, loader, logger):
    name, fusion_override, ablation_key = (
        VARIANT_BY_KEY[variant_key][1],
        VARIANT_BY_KEY[variant_key][2],
        VARIANT_BY_KEY[variant_key][3],
    )

    set_seed(seed)
    cfg = build_cfg(dc, seed=seed, fusion_mode_override=fusion_override,
                    ablation_key=ablation_key)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))

    model = create_sag_model(
        num_nodes=s["num_nodes"], config=cfg,
        features=loader.features, adj=loader.adj,
        ablation=ablation_key,
    )

    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(seed)

    t0 = time.time()
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    elapsed = time.time() - t0
    scores = model.get_scores()

    y = np.asarray(loader.labels)
    if scores.max() > scores.min():
        s_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        s_norm = scores

    auroc = float(compute_auroc(y, s_norm))
    auprc = float(compute_auprc(y, s_norm))
    macro_f1 = float(compute_macro_f1(y, s_norm))
    auprc_bal, macro_f1_bal = compute_balanced_auprc_f1(y, s_norm, n_trials=20, seed=0)

    return {
        "dataset": dataset, "variant": variant_key, "variant_name": name,
        "fusion_mode": fusion_override or dc.get("fusion_mode", "auto"),
        "ablation": ablation_key or "none",
        "seed": int(seed),
        "auroc": auroc, "auprc": auprc, "macro_f1": macro_f1,
        "auprc_bal": float(auprc_bal), "macro_f1_bal": float(macro_f1_bal),
        "elapsed_s": float(elapsed),
        "num_edges": int(s["num_edges"]),
        "num_nodes": int(s["num_nodes"]),
        "num_anomalies": int(s["num_anomalies"]),
        "skipped": False, "skip_reason": "",
    }


def skip_stub(dataset: str, variant_key: str, seed: int, reason: str) -> dict:
    name = VARIANT_BY_KEY[variant_key][1]
    return {
        "dataset": dataset, "variant": variant_key, "variant_name": name,
        "fusion_mode": "", "ablation": "",
        "seed": int(seed),
        "auroc": float("nan"), "auprc": float("nan"), "macro_f1": float("nan"),
        "auprc_bal": float("nan"), "macro_f1_bal": float("nan"),
        "elapsed_s": 0.0, "num_edges": 0, "num_nodes": 0, "num_anomalies": 0,
        "skipped": True, "skip_reason": reason,
    }


# ---------------------------------------------------------------------------
# Aggregation + CSV emission
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "dataset", "variant", "variant_name", "fusion_mode", "ablation",
    "seed", "auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
    "elapsed_s", "num_edges", "num_nodes", "num_anomalies",
    "skipped", "skip_reason",
]
SUMMARY_METRICS = ("auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal")
SUMMARY_COLUMNS = (
    ["dataset", "variant", "variant_name", "n_seeds", "applicable", "skip_reason"]
    + [f"{m}_mean" for m in SUMMARY_METRICS]
    + [f"{m}_std" for m in SUMMARY_METRICS]
    + [f"{m}_min" for m in SUMMARY_METRICS]
    + [f"{m}_max" for m in SUMMARY_METRICS]
)


def write_raw_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RAW_COLUMNS})


def write_summary_csv(rows, path):
    grouped = {}
    for r in rows:
        grouped.setdefault((r["dataset"], r["variant"]), []).append(r)
    out = []
    for (ds, vk), group in grouped.items():
        ok = [g for g in group if not g["skipped"]]
        name = group[0]["variant_name"] if group else vk
        if not ok:
            row = {
                "dataset": ds, "variant": vk, "variant_name": name, "n_seeds": 0,
                "applicable": False,
                "skip_reason": group[0].get("skip_reason", "") if group else "",
            }
            for m in SUMMARY_METRICS:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
                row[f"{m}_min"] = ""
                row[f"{m}_max"] = ""
        else:
            row = {
                "dataset": ds, "variant": vk, "variant_name": name,
                "n_seeds": len(ok), "applicable": True, "skip_reason": "",
            }
            for m in SUMMARY_METRICS:
                v = np.array([r[m] for r in ok], dtype=np.float64)
                row[f"{m}_mean"] = float(v.mean())
                row[f"{m}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
                row[f"{m}_min"] = float(v.min())
                row[f"{m}_max"] = float(v.max())
        out.append(row)
    out.sort(key=lambda r: (r["dataset"], VARIANT_KEYS.index(r["variant"])
                            if r["variant"] in VARIANT_KEYS else 999))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--variants", default=",".join(VARIANT_KEYS))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    for v in variants:
        if v not in VARIANT_KEYS:
            raise ValueError(f"unknown variant: {v} (valid: {VARIANT_KEYS})")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"predicate_ablation{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"predicate_ablation{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_PRED_ABL", log_dir)
    logger.info(f"datasets={datasets}  variants={variants}  seeds={seeds}")
    logger.info(f"run_dir={run_dir}")

    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    raw_rows = []
    raw_run = os.path.join(run_dir, "predicate_ablation_raw.csv")
    sum_run = os.path.join(run_dir, "predicate_ablation_summary.csv")
    raw_canon = os.path.join(canon_dir, "predicate_ablation_raw.csv")
    sum_canon = os.path.join(canon_dir, "predicate_ablation_summary.csv")

    for ds in datasets:
        try:
            dc = get_dataset_config(ds)
        except ValueError as e:
            logger.error(f"unknown dataset {ds}: {e}")
            continue
        logger.info(f"\n========== dataset = {ds} ==========")
        loader = DataLoader(
            data_dir="./data", dataset_name=ds,
            clique_size=dc.get("clique_size", 15),
            num_cliques=dc.get("num_cliques", 10),
            seed=BASE_SEED,
        )
        try:
            loader.load()
        except Exception as e:
            logger.error(f"  cannot load {ds}: {e}")
            for vk in variants:
                for sd in seeds:
                    raw_rows.append(skip_stub(ds, vk, sd, f"load_error:{type(e).__name__}:{e}"))
            write_raw_csv(raw_rows, raw_run)
            write_raw_csv(raw_rows, raw_canon)
            continue
        logger.info(f"  stats: {loader.get_stats()}")

        for vk in variants:
            applicable, reason = variant_applicable(ds, vk, dc)
            if not applicable:
                logger.info(f"  [SKIP] {ds} / {vk}: {reason}")
                for sd in seeds:
                    raw_rows.append(skip_stub(ds, vk, sd, reason))
                continue
            logger.info(f"\n  --- variant = {vk}  ({VARIANT_BY_KEY[vk][1]}) ---")
            for sd in seeds:
                try:
                    row = run_one(ds, vk, sd, dc, loader, logger)
                    logger.info(
                        "    seed=%d AUROC=%.4f AUPRC=%.4f F1=%.4f AUPRCb=%.4f F1b=%.4f (%.1fs)" %
                        (sd, row["auroc"], row["auprc"], row["macro_f1"],
                         row["auprc_bal"], row["macro_f1_bal"], row["elapsed_s"])
                    )
                    raw_rows.append(row)
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"    seed={sd} FAILED {type(e).__name__}: {e}\n{tb}")
                    err = skip_stub(ds, vk, sd, f"ERROR:{type(e).__name__}:{e}")
                    err["skipped"] = True
                    raw_rows.append(err)
                write_raw_csv(raw_rows, raw_run)
                write_raw_csv(raw_rows, raw_canon)

    summary = write_summary_csv(raw_rows, sum_run)
    write_summary_csv(raw_rows, sum_canon)
    save_json({"raw": raw_rows, "summary": summary},
              os.path.join(run_dir, "predicate_ablation.json"))

    logger.info("=" * 70)
    logger.info(f"WROTE  raw     -> {raw_run}")
    logger.info(f"WROTE  summary -> {sum_run}")
    logger.info(f"CANON  raw     -> {raw_canon}")
    logger.info(f"CANON  summary -> {sum_canon}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
