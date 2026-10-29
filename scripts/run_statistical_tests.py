"""Statistical significance testing: SAG-full vs strong baselines.

For each (dataset, metric, baseline) we collect per-seed paired metric
values for SAG-full and the baseline, then run:
  * paired Student's t-test                       (scipy.stats.ttest_rel)
  * Wilcoxon signed-rank test (zero-method=zsplit) (scipy.stats.wilcoxon)
  * Cohen's d_z effect size (mean(diff) / std(diff))

Win/tie/loss is decided at alpha=0.05 on the Wilcoxon p-value (default).

Modes
-----
  --mode run     : run SAG-full + each baseline freshly per seed and test.
  --mode reuse   : read existing per-seed metrics CSV produced by --mode run.

Outputs (canonical, latest run; --overwrite to replace):
  outputs/results/revision/statistical_tests.csv
  outputs/results/revision/significance_summary.md
  outputs/results/revision/statistical_tests_per_seed.csv  (raw per-run metrics
                                                            usable for --mode reuse)
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
from scipy import stats as sp_stats

from configs import Config, get_dataset_config
from src.baselines import BASELINES
from src.data import DataLoader
from src.evaluation.metrics import (
    compute_auroc, compute_auprc, compute_macro_f1, compute_balanced_auprc_f1,
)
from src.models import create_sag_model
from src.utils import save_json, set_seed, setup_logger


REQUESTED_BASELINES = (
    "MIDAS", "NetWalk", "SedanSpot",
    "DOMINANT", "CoLA", "ANEMONE", "AnomalyDAE", "CONAD",
    "GAE", "AdONE", "DONE", "DMGD", "GAAN", "GUIDE",
    "OCGNN", "ONE", "Radar", "SCAN", "ANOMALOUS",
)
DATASETS_DEFAULT = ("Cora", "Citeseer", "Pubmed", "BlogCatalog",
                    "YelpReviews", "Enron")
SEEDS_DEFAULT = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
METRICS = ("auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal")
BASE_SEED = 42
ALPHA = 0.05


# ---------------------------------------------------------------------------
class CfgShim:
    pass


def build_cfg(dc: dict, seed: int):
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


def normalize(s: np.ndarray) -> np.ndarray:
    if s.max() > s.min():
        return (s - s.min()) / (s.max() - s.min())
    return np.zeros_like(s)


def metrics_for(y, scores) -> dict:
    s = normalize(np.asarray(scores, dtype=np.float64))
    auprc_bal, f1_bal = compute_balanced_auprc_f1(y, s, n_trials=20, seed=0)
    return {
        "auroc": float(compute_auroc(y, s)),
        "auprc": float(compute_auprc(y, s)),
        "macro_f1": float(compute_macro_f1(y, s)),
        "auprc_bal": float(auprc_bal),
        "macro_f1_bal": float(f1_bal),
    }


def run_sag(loader, dc, seed, ae_cache):
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    cfg.window_size = max(128, min(s["num_edges"] // 32, 4096))
    if ae_cache is not None:
        cfg._trained_ae_score_cache = ae_cache
    model = create_sag_model(num_nodes=s["num_nodes"], config=cfg,
                             features=loader.features, adj=loader.adj)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    do_shuffle = (order == "shuffle")
    passes = (2 if s["num_edges"] > 50000 else 3) if do_shuffle else 1
    rng = np.random.default_rng(seed)
    model.process_stream_batch(edges, passes=passes, shuffle=do_shuffle, rng=rng)
    return model.get_scores()


def run_baseline(method, loader, dc, seed):
    set_seed(seed)
    cfg = build_cfg(dc, seed=seed)
    s = loader.get_stats()
    cls = BASELINES[method]
    try:
        bm = cls(num_nodes=s["num_nodes"], config=cfg,
                 features=loader.features, seed=seed)
    except TypeError:
        bm = cls(num_nodes=s["num_nodes"], config=cfg,
                 features=loader.features)
    order = getattr(cfg, "edge_order", "auto")
    if order == "auto":
        order = "shuffle" if cfg.has_ground_truth else "bfs"
    edges = loader.get_edge_stream(mode=order)
    bm.fit(loader.adj, edges, loader.labels)
    return bm.score()


def _build_ae_cache(loader, dc, logger):
    if not dc.get("use_trained_ae", False):
        return None
    try:
        from src.models.trained_attr import UnsupervisedGCNAutoencoder
        ae_cfg = dc.get("trained_ae_cfg", {}) or {}
        cache_key = (id(loader.features), id(loader.adj),
                     tuple(sorted(ae_cfg.items())), BASE_SEED)
        ae = UnsupervisedGCNAutoencoder(seed=BASE_SEED, **ae_cfg)
        sc = ae.fit_score(loader.adj, loader.features)
        if not np.all(np.isfinite(sc)) or sc.std() < 1e-12:
            return None
        return {"key": cache_key, "score": sc}
    except Exception as e:
        logger.warning(f"  AE pre-training failed: {e}")
        return None


# ---------------------------------------------------------------------------
PER_SEED_COLS = ["dataset", "method", "seed",
                 "auroc", "auprc", "macro_f1", "auprc_bal", "macro_f1_bal",
                 "skipped", "error"]
TEST_COLS = [
    "dataset", "method_a", "method_b", "metric", "n_pairs",
    "mean_a", "mean_b", "mean_diff",
    "ttest_t", "ttest_p", "wilcoxon_W", "wilcoxon_p",
    "cohens_dz", "alpha", "verdict_wilcoxon",
]


def write_per_seed_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_SEED_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in PER_SEED_COLS})


def read_per_seed_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_per_seed(args, logger):
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows = []
    for ds in datasets:
        try:
            dc = get_dataset_config(ds)
        except ValueError as e:
            logger.error(f"unknown dataset {ds}: {e}")
            continue
        loader = DataLoader(
            data_dir="./data", dataset_name=ds,
            clique_size=dc.get("clique_size", 15),
            num_cliques=dc.get("num_cliques", 10),
            seed=BASE_SEED,
        )
        try:
            loader.load()
        except Exception as e:
            logger.error(f"  load failed: {e}")
            continue
        ae_cache = _build_ae_cache(loader, dc, logger)
        y = np.asarray(loader.labels)
        FAIL_FAST_THRESHOLD = 2
        for method in methods:
            registered = (method == "SAG-full") or (method in BASELINES)
            consecutive_fails = 0
            first_fail_msg = ""
            skip_remaining = False
            for sd in seeds:
                logger.info(f"  {ds} | {method} | seed={sd}")
                if not registered:
                    rows.append({"dataset": ds, "method": method, "seed": sd,
                                 "auroc": "", "auprc": "", "macro_f1": "",
                                 "auprc_bal": "", "macro_f1_bal": "",
                                 "skipped": True,
                                 "error": "method not registered in BASELINES"})
                    continue
                if skip_remaining:
                    rows.append({"dataset": ds, "method": method, "seed": sd,
                                 "auroc": "", "auprc": "", "macro_f1": "",
                                 "auprc_bal": "", "macro_f1_bal": "",
                                 "skipped": True,
                                 "error": f"FAST_SKIP_after_{consecutive_fails}_consecutive_fails: {first_fail_msg}"})
                    logger.info(f"    [FAST-SKIP] consecutive failures > {FAIL_FAST_THRESHOLD}; "
                                f"skipping remaining seeds for {ds}/{method}")
                    continue
                t0 = time.time()
                try:
                    if method == "SAG-full":
                        sc = run_sag(loader, dc, sd, ae_cache)
                    else:
                        sc = run_baseline(method, loader, dc, sd)
                    m = metrics_for(y, sc)
                    rows.append({"dataset": ds, "method": method, "seed": sd,
                                 **m, "skipped": False, "error": ""})
                    logger.info("    AUROC=%.4f F1b=%.4f (%.1fs)" %
                                (m["auroc"], m["macro_f1_bal"], time.time()-t0))
                    consecutive_fails = 0
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"    FAILED {type(e).__name__}: {e}\n{tb}")
                    err_msg = f"{type(e).__name__}:{e}"
                    rows.append({"dataset": ds, "method": method, "seed": sd,
                                 "auroc": "", "auprc": "", "macro_f1": "",
                                 "auprc_bal": "", "macro_f1_bal": "",
                                 "skipped": True,
                                 "error": err_msg})
                    consecutive_fails += 1
                    if consecutive_fails == 1:
                        first_fail_msg = err_msg[:200]
                    if consecutive_fails >= FAIL_FAST_THRESHOLD:
                        skip_remaining = True
    return rows


# ---------------------------------------------------------------------------
def run_paired_tests(per_seed_rows, anchor: str, alpha: float):
    by_key = {}
    for r in per_seed_rows:
        if str(r.get("skipped")).lower() == "true" or r.get("skipped") is True:
            continue
        key = (r["dataset"], r["method"], int(r["seed"]))
        by_key[key] = r
    datasets = sorted({k[0] for k in by_key.keys()})
    methods = sorted({k[1] for k in by_key.keys()})
    out = []
    for ds in datasets:
        for mb in methods:
            if mb == anchor:
                continue
            paired = []
            seeds = sorted(set(int(k[2]) for k in by_key
                                if k[0] == ds and k[1] in (anchor, mb)))
            for sd in seeds:
                ra = by_key.get((ds, anchor, sd))
                rb = by_key.get((ds, mb, sd))
                if ra is None or rb is None:
                    continue
                paired.append((sd, ra, rb))
            if not paired:
                continue
            for metric in METRICS:
                a_vals, b_vals = [], []
                for sd, ra, rb in paired:
                    try:
                        a_vals.append(float(ra[metric]))
                        b_vals.append(float(rb[metric]))
                    except (KeyError, ValueError, TypeError):
                        continue
                if len(a_vals) < 2:
                    continue
                a = np.array(a_vals, dtype=np.float64)
                b = np.array(b_vals, dtype=np.float64)
                d = a - b
                try:
                    t_stat, t_p = sp_stats.ttest_rel(a, b)
                except Exception:
                    t_stat, t_p = float("nan"), float("nan")
                try:
                    if np.allclose(d, 0):
                        w_stat, w_p = float("nan"), 1.0
                    else:
                        w = sp_stats.wilcoxon(a, b, zero_method="zsplit",
                                              alternative="two-sided")
                        w_stat, w_p = float(w.statistic), float(w.pvalue)
                except Exception:
                    w_stat, w_p = float("nan"), float("nan")
                d_std = d.std(ddof=1) if d.size > 1 else 0.0
                dz = float(d.mean() / d_std) if d_std > 1e-12 else 0.0
                if w_p == w_p and w_p < alpha:
                    verdict = "SAG_better" if a.mean() > b.mean() else "SAG_worse"
                else:
                    verdict = "tie"
                out.append({
                    "dataset": ds, "method_a": anchor, "method_b": mb,
                    "metric": metric, "n_pairs": len(a_vals),
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "mean_diff": float(d.mean()),
                    "ttest_t": float(t_stat), "ttest_p": float(t_p),
                    "wilcoxon_W": float(w_stat), "wilcoxon_p": float(w_p),
                    "cohens_dz": dz, "alpha": float(alpha),
                    "verdict_wilcoxon": verdict,
                })
    return out


def write_test_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TEST_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TEST_COLS})


def write_summary_md(test_rows, missing_methods, alpha, path):
    by_metric = {}
    for r in test_rows:
        by_metric.setdefault(r["metric"], []).append(r)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SAG vs baselines — paired statistical tests\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}  \n")
        f.write(f"Significance: alpha = {alpha} on the Wilcoxon two-sided p-value.\n\n")
        for metric in METRICS:
            block = by_metric.get(metric, [])
            if not block:
                continue
            f.write(f"## {metric}\n\n")
            f.write("| dataset | baseline | n | mean(SAG) | mean(baseline) | diff | t-test p | Wilcoxon p | dz | verdict |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(block, key=lambda r: (r["dataset"], r["method_b"])):
                f.write("| {ds} | {mb} | {n} | {ma:.4f} | {mbv:.4f} | {d:+.4f} | "
                        "{tp} | {wp} | {dz:+.2f} | **{v}** |\n".format(
                            ds=r["dataset"], mb=r["method_b"], n=r["n_pairs"],
                            ma=r["mean_a"], mbv=r["mean_b"], d=r["mean_diff"],
                            tp=("%.4f" % r["ttest_p"]) if r["ttest_p"] == r["ttest_p"] else "nan",
                            wp=("%.4f" % r["wilcoxon_p"]) if r["wilcoxon_p"] == r["wilcoxon_p"] else "nan",
                            dz=r["cohens_dz"], v=r["verdict_wilcoxon"]))
            f.write("\n")
        if missing_methods:
            f.write("## Requested baselines not in this codebase\n\n")
            for m in missing_methods:
                f.write(f"- **{m}** — no `{m}Baseline` registered in "
                        "`src/baselines/__init__.py::BASELINES`. "
                        "See `outputs/results/revision/UNIMPLEMENTED.md`.\n")
            f.write("\n")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--methods",
                   default="SAG-full,MIDAS,NetWalk,GraphSAGE,DOMINANT,DySAT,SedanSpot")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    p.add_argument("--anchor", default="SAG-full",
                   help="reference method (paired tests are anchor vs each other method)")
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--mode", choices=("run", "reuse"), default="run")
    p.add_argument("--reuse_csv",
                   default="./outputs/results/revision/statistical_tests_per_seed.csv",
                   help="path read in --mode reuse (default: canonical per-seed CSV)")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _maybe_skip(path, overwrite, logger):
    if os.path.exists(path) and not overwrite:
        logger.warning(f"  refusing to overwrite {path} (pass --overwrite)")
        return True
    return False


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_dir = os.path.join(args.output_dir, "results", "revision",
                           f"statistical_tests{tag}_{ts}")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"statistical_tests{tag}_{ts}")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(canon_dir, exist_ok=True)

    logger = setup_logger("SAG_STAT_TEST", log_dir)
    Config.data_dir = "./data"
    Config.output_dir = args.output_dir

    methods_requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    missing = [m for m in REQUESTED_BASELINES
               if m not in BASELINES and m not in methods_requested]

    per_seed_run = os.path.join(run_dir, "statistical_tests_per_seed.csv")
    per_seed_canon = os.path.join(canon_dir, "statistical_tests_per_seed.csv")
    test_run = os.path.join(run_dir, "statistical_tests.csv")
    test_canon = os.path.join(canon_dir, "statistical_tests.csv")
    md_canon = os.path.join(canon_dir, "significance_summary.md")

    if args.mode == "run":
        per_seed = collect_per_seed(args, logger)
        write_per_seed_csv(per_seed, per_seed_run)
        if not _maybe_skip(per_seed_canon, args.overwrite, logger):
            write_per_seed_csv(per_seed, per_seed_canon)
    else:
        if not os.path.exists(args.reuse_csv):
            logger.error(f"reuse_csv not found: {args.reuse_csv}")
            return
        per_seed = read_per_seed_csv(args.reuse_csv)

    test_rows = run_paired_tests(per_seed, args.anchor, args.alpha)
    write_test_csv(test_rows, test_run)
    save_json({"per_seed": per_seed, "tests": test_rows,
               "missing_baselines": missing,
               "argv": sys.argv, "timestamp": ts},
              os.path.join(run_dir, "statistical_tests.json"))

    if not _maybe_skip(test_canon, args.overwrite, logger):
        write_test_csv(test_rows, test_canon)
    if not _maybe_skip(md_canon, args.overwrite, logger):
        write_summary_md(test_rows, missing, args.alpha, md_canon)

    logger.info(f"\nWROTE per_seed -> {per_seed_run}")
    logger.info(f"WROTE tests    -> {test_run}")
    logger.info(f"CANON tests    -> {test_canon}")
    logger.info(f"CANON markdown -> {md_canon}")
    if missing:
        logger.info(f"missing baselines (not registered): {', '.join(missing)}")


if __name__ == "__main__":
    main()
