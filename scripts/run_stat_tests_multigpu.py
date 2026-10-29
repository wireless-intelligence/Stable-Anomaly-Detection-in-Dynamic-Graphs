#!/usr/bin/env python
import argparse
import csv
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


N_SQUARED_METHODS = {"DOMINANT", "AnomalyDAE", "CONAD", "GAAN", "Radar", "ANOMALOUS"}
LARGE_DATASETS_NODES = {
    "YelpReviews": 1_035_966,
    "Arxiv": 201_443,
    "CTU-13": 325_471,
    "YelpChi": 45_954,
}


def detect_gpus():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


def is_known_oom(method, dataset, gpu_mem_gb):
    n = LARGE_DATASETS_NODES.get(dataset, 0)
    if method in N_SQUARED_METHODS:
        if n >= 100_000:
            return True
        if dataset == "Pubmed" and method == "GAAN" and gpu_mem_gb < 100:
            return True
    return False


def estimate_priority(method, dataset, gpu_mem_gb):
    n = LARGE_DATASETS_NODES.get(dataset, 0)
    if method in N_SQUARED_METHODS and n >= 10_000:
        return 9
    if dataset == "YelpReviews":
        return 7 if method in {"NetWalk", "GUIDE"} else 5
    if dataset == "Arxiv" or dataset == "CTU-13":
        return 6
    if dataset in {"Pubmed", "ACM", "YelpChi"}:
        return 3
    return 1


def parse_args():
    p = argparse.ArgumentParser(
        description="multi-GPU dispatcher for run_statistical_tests.py: "
                    "splits (dataset, method) pairs across GPUs, runs in "
                    "parallel, then auto-merges per-seed CSVs and re-runs "
                    "paired tests on the merged data.")
    p.add_argument("--datasets", default="Cora,Citeseer,Pubmed,BlogCatalog,YelpReviews,Enron")
    p.add_argument("--methods", default=",".join([
        "MIDAS", "NetWalk", "SedanSpot",
        "DOMINANT", "CoLA", "AnomalyDAE", "CONAD", "GAE", "AdONE", "DONE",
        "GAAN", "GUIDE", "OCGNN", "ONE", "Radar", "SCAN", "ANOMALOUS",
    ]))
    p.add_argument("--anchor", default="SAG-full",
                   help="anchor method always paired with each baseline (will "
                        "be run once per dataset)")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--num_gpus", type=int, default=0,
                   help="0 = auto-detect via torch.cuda.device_count()")
    p.add_argument("--gpus", default="",
                   help="explicit comma-list of physical GPU IDs to use; "
                        "overrides --num_gpus")
    p.add_argument("--gpu_mem_gb", type=float, default=32.0,
                   help="approximate per-GPU memory; used to skip combos that "
                        "are known to OOM (e.g. Pubmed x GAAN on <100GB)")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--tag", default="multigpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_impossible", action="store_true",
                   help="skip (dataset, method) combos known to OOM by "
                        "algorithm design (e.g. YelpReviews x DOMINANT)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_merge", action="store_true",
                   help="skip the final merge + paired-test step")
    return p.parse_args()


def build_jobs(datasets, methods, anchor, gpu_mem_gb, skip_impossible):
    jobs = []
    for ds in datasets:
        jobs.append({
            "kind": "anchor",
            "dataset": ds, "method": anchor,
            "priority": estimate_priority(anchor, ds, gpu_mem_gb),
        })
    for ds in datasets:
        for m in methods:
            if m == anchor:
                continue
            if skip_impossible and is_known_oom(m, ds, gpu_mem_gb):
                jobs.append({
                    "kind": "skipped_known_oom",
                    "dataset": ds, "method": m,
                    "priority": -1,
                })
                continue
            jobs.append({
                "kind": "baseline",
                "dataset": ds, "method": m,
                "priority": estimate_priority(m, ds, gpu_mem_gb),
            })
    jobs.sort(key=lambda j: (-j["priority"], j["dataset"], j["method"]))
    return jobs


def run_one(job, gpu_id, args, log_dir):
    ds, m = job["dataset"], job["method"]
    sub_tag = f"{args.tag}_{ds}_{m}_g{gpu_id}"
    log_path = os.path.join(log_dir, f"{ds}__{m}__g{gpu_id}.log")
    cmd = [
        sys.executable, "scripts/run_statistical_tests.py",
        "--datasets", ds,
        "--methods", m,
        "--seeds", args.seeds,
        "--anchor", args.anchor,
        "--alpha", str(args.alpha),
        "--tag", sub_tag,
        "--output_dir", args.output_dir,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n# gpu: cuda:{gpu_id}\n"
                f"# started: {datetime.now().isoformat()}\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                              stdout=f, stderr=subprocess.STDOUT, check=False)
        elapsed = time.time() - t0
        f.write(f"\n# finished: {datetime.now().isoformat()}\n")
        f.write(f"# elapsed: {elapsed:.1f}s\n")
        f.write(f"# returncode: {proc.returncode}\n")
    return proc.returncode, elapsed, sub_tag


def worker(gpu_id, job_q, result_q, args, log_dir):
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        ds, m = job["dataset"], job["method"]
        if job["kind"] == "skipped_known_oom":
            print(f"[gpu:{gpu_id}] SKIP_OOM   {ds}/{m}", flush=True)
            result_q.put((gpu_id, ds, m, "skipped_known_oom", 0, 0.0, ""))
            job_q.task_done()
            continue
        print(f"[gpu:{gpu_id}] START      {ds}/{m}  (pri={job['priority']})",
              flush=True)
        rc, elapsed, sub_tag = run_one(job, gpu_id, args, log_dir)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[gpu:{gpu_id}] {status:10}  {ds}/{m}  ({elapsed:.0f}s)",
              flush=True)
        result_q.put((gpu_id, ds, m, status, rc, elapsed, sub_tag))
        job_q.task_done()


def merge_per_seed_csvs(args, sub_tags):
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    canon_per_seed = os.path.join(canon_dir, "statistical_tests_per_seed.csv")
    rows = []
    seen = set()
    cols = None
    for tag in sub_tags:
        pat = os.path.join(canon_dir, f"statistical_tests_{tag}_*",
                           "statistical_tests_per_seed.csv")
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if cols is None:
                        cols = list(row.keys())
                    key = (row.get("dataset", ""), row.get("method", ""),
                           row.get("seed", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
    if not rows:
        print("[dispatcher] WARNING: no per-seed CSVs found to merge")
        return None
    os.makedirs(canon_dir, exist_ok=True)
    with open(canon_per_seed, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[dispatcher] merged {len(rows)} rows -> {canon_per_seed}")
    return canon_per_seed


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip() != ""]
    else:
        n = args.num_gpus or detect_gpus() or 1
        gpu_ids = list(range(n))

    jobs = build_jobs(datasets, methods, args.anchor,
                      args.gpu_mem_gb, args.skip_impossible)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, "logs",
                           f"stat_multigpu_{args.tag}_{ts}")
    os.makedirs(log_dir, exist_ok=True)

    runnable = [j for j in jobs if j["kind"] != "skipped_known_oom"]
    skipped = [j for j in jobs if j["kind"] == "skipped_known_oom"]

    print(f"[dispatcher] gpus={gpu_ids}  total_jobs={len(jobs)}  "
          f"runnable={len(runnable)}  skipped_known_oom={len(skipped)}",
          flush=True)
    print(f"[dispatcher] log_dir={log_dir}", flush=True)

    if args.dry_run:
        print("\n=== job plan ===")
        for j in jobs:
            print(f"  pri={j['priority']:>2}  {j['kind']:20}  "
                  f"{j['dataset']:14}  {j['method']}")
        return

    job_q = queue.Queue()
    for j in runnable:
        job_q.put(j)
    result_q = queue.Queue()

    threads = []
    t0 = time.time()
    for gpu_id in gpu_ids:
        th = threading.Thread(target=worker,
                              args=(gpu_id, job_q, result_q, args, log_dir),
                              daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join()
    total_wall = time.time() - t0

    results = []
    while not result_q.empty():
        results.append(result_q.get())

    sub_tags = [r[6] for r in results if r[6]]

    print(f"\n=== dispatcher summary (wall-time {total_wall:.0f}s) ===")
    n_ok = sum(1 for r in results if r[3] == "OK")
    n_fail = sum(1 for r in results if r[3].startswith("FAIL"))
    n_skip = sum(1 for r in results if r[3] == "skipped_known_oom") + len(skipped)
    print(f"  ok={n_ok}  fail={n_fail}  skipped_known_oom={n_skip}  "
          f"total={len(results) + len(skipped)}")
    for gpu, ds, m, status, rc, elapsed, _ in sorted(results,
                                                      key=lambda r: (r[1], r[2])):
        print(f"  gpu:{gpu}  {ds:14}  {m:14}  {status:20}  {elapsed:.0f}s")

    summary_path = os.path.join(log_dir, "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "datasets": datasets, "methods": methods, "anchor": args.anchor,
            "seeds": args.seeds, "gpus": gpu_ids,
            "total_wall_time_s": total_wall,
            "results": [
                {"gpu": gpu, "dataset": ds, "method": m, "status": status,
                 "returncode": rc, "elapsed_s": elapsed, "sub_tag": stag}
                for gpu, ds, m, status, rc, elapsed, stag in results
            ],
            "skipped_known_oom": [
                {"dataset": j["dataset"], "method": j["method"]}
                for j in skipped
            ],
        }, f, indent=2)
    print(f"\n[dispatcher] summary -> {summary_path}")

    if args.no_merge:
        return
    print("\n[dispatcher] merging per-seed CSVs across all sub-runs...")
    merged_csv = merge_per_seed_csvs(args, sub_tags)
    if merged_csv is None:
        return

    print("[dispatcher] re-running paired tests on merged CSV...")
    merge_cmd = [sys.executable, "scripts/run_statistical_tests.py",
                 "--mode", "reuse",
                 "--reuse_csv", merged_csv,
                 "--anchor", args.anchor,
                 "--alpha", str(args.alpha),
                 "--tag", f"{args.tag}_merged",
                 "--output_dir", args.output_dir,
                 "--overwrite"]
    proc = subprocess.run(merge_cmd, cwd=REPO_ROOT, check=False)
    print(f"[dispatcher] paired-tests rc={proc.returncode}")
    print(f"[dispatcher] FINAL outputs:")
    canon_dir = os.path.join(args.output_dir, "results", "revision")
    print(f"  {canon_dir}/statistical_tests_per_seed.csv")
    print(f"  {canon_dir}/statistical_tests.csv")
    print(f"  {canon_dir}/significance_summary.md")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
