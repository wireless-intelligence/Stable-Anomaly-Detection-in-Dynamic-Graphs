#!/usr/bin/env python
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


STAGE_A_JOBS = [
    ("leakage_check",
     [sys.executable, "scripts/check_leakage_protocol.py",
      "--runtime_dataset", "Cora", "--temporal_dataset", "CTU-13",
      "--overwrite"]),
    ("memory_breakdown_Enron",
     [sys.executable, "scripts/profile_sag_memory_breakdown.py",
      "--dataset", "Enron", "--overwrite"]),
    ("memory_breakdown_YelpReviews",
     [sys.executable, "scripts/profile_sag_memory_breakdown.py",
      "--dataset", "YelpReviews", "--overwrite"]),
    ("sag_seed_variance",
     [sys.executable, "experiments/run_sag_seed_variance.py",
      "--datasets", "YelpReviews,Enron,Cora,BlogCatalog",
      "--seeds", "0,1,2,3,4,5,6,7,8,9", "--overwrite"]),
    ("predicate_ablation",
     [sys.executable, "experiments/run_predicate_ablation.py",
      "--datasets", "Cora,Citeseer,Pubmed,BlogCatalog,YelpReviews,Enron",
      "--seeds", "0,1,2,3,4,5,6,7,8,9", "--overwrite"]),
    ("parameter_sensitivity",
     [sys.executable, "experiments/run_parameter_sensitivity.py",
      "--datasets", "YelpReviews,Enron,Cora,BlogCatalog",
      "--seeds", "0,1,2", "--overwrite"]),
    ("calibration_analysis",
     [sys.executable, "experiments/run_calibration_analysis.py",
      "--datasets", "Cora,Citeseer,Pubmed,BlogCatalog,Flickr,ACM,Amazon,YelpChi,YelpReviews,Enron",
      "--seeds", "0,1,2,3,4", "--overwrite"]),
    ("anomaly_type_sweep",
     [sys.executable, "experiments/run_anomaly_type_sweep.py",
      "--datasets", "Cora,Citeseer,Pubmed,BlogCatalog",
      "--seeds", "0,1,2", "--overwrite"]),
    ("scalability_stress",
     [sys.executable, "experiments/run_scalability_stress_test.py",
      "--datasets", "YelpReviews,Enron,BlogCatalog",
      "--seeds", "0,1,2", "--overwrite"]),
    ("time_to_detect",
     [sys.executable, "experiments/run_time_to_detect.py",
      "--datasets", "YelpReviews,Enron,BlogCatalog,Cora",
      "--seeds", "0,1,2", "--overwrite"]),
    ("failure_case",
     [sys.executable, "experiments/run_failure_case_analysis.py",
      "--datasets", "Cora,Citeseer,Pubmed,BlogCatalog",
      "--seeds", "0,1,2", "--overwrite"]),
    ("split_protocol",
     [sys.executable, "experiments/run_split_protocol_selection.py",
      "--datasets", "Cora,Citeseer,Pubmed,BlogCatalog,YelpReviews,Enron,CTU-13",
      "--methods", "SAG-online-strict,MIDAS,DOMINANT,ANEMONE",
      "--seeds", "42,43,44,45,46,47,48,49,50,51", "--overwrite"]),
    ("exposure_bucket",
     [sys.executable, "experiments/run_exposure_bucket_analysis.py",
      "--datasets", "YelpReviews,Enron,BlogCatalog",
      "--methods", "SAG-full,SAG-no-exposure,MIDAS,NetWalk,GraphSAGE,DOMINANT,CoLA,ANEMONE",
      "--seeds", "0,1,2", "--overwrite"]),
]


STAGE_B_JOBS = [
    ("stat_pygod_core",
     [sys.executable, "scripts/run_statistical_tests.py",
      "--datasets", "Cora,Citeseer,Pubmed,BlogCatalog,YelpReviews,Enron",
      "--methods", "SAG-full,MIDAS,NetWalk,SedanSpot,DOMINANT,CoLA,AnomalyDAE,CONAD,GAE,AdONE,DONE,DMGD,GAAN,GUIDE,OCGNN,ONE,Radar,SCAN,ANOMALOUS",
      "--seeds", "0,1,2,3,4,5,6,7,8,9",
      "--tag", "pygod-core", "--overwrite"]),
    ("stat_vendor_torch",
     [sys.executable, "scripts/run_statistical_tests.py",
      "--datasets", "Cora,Citeseer,Pubmed,BlogCatalog",
      "--methods", "SAG-full,ANEMONE,ConsisGAD,PMP,GAAP",
      "--seeds", "0,1,2",
      "--tag", "vendor-torch", "--overwrite"]),
    ("stat_tf1",
     [sys.executable, "scripts/run_statistical_tests.py",
      "--datasets", "Cora,BlogCatalog",
      "--methods", "SAG-full,ComGA,DySAT",
      "--seeds", "0,1,2",
      "--tag", "tf1", "--overwrite"]),
    ("stat_graphormer",
     [sys.executable, "scripts/run_statistical_tests.py",
      "--datasets", "Cora",
      "--methods", "SAG-full,Graphormer",
      "--seeds", "0,1,2",
      "--tag", "graphormer", "--overwrite"]),
    ("stat_llm_llama",
     [sys.executable, "scripts/run_statistical_tests.py",
      "--datasets", "Cora",
      "--methods", "SAG-full,GraphGPT,InstructGLM,TAPE",
      "--seeds", "0",
      "--tag", "llm-llama", "--overwrite"]),
    ("stat_llm_qwen",
     [sys.executable, "scripts/run_statistical_tests.py",
      "--datasets", "Cora",
      "--methods", "SAG-full,HiGPT",
      "--seeds", "0",
      "--tag", "llm-qwen", "--overwrite"]),
]


def detect_gpus():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


def run_job(job_id, name, cmd, gpu_id, log_dir, env_extra=None):
    log_path = os.path.join(log_dir, f"{job_id:03d}_{name}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# job: {name}\n# gpu: cuda:{gpu_id}\n# cmd: {' '.join(cmd)}\n")
        f.write(f"# started: {datetime.now().isoformat()}\n\n")
        f.flush()
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                  stdout=f, stderr=subprocess.STDOUT,
                                  check=False)
            elapsed = time.time() - t0
            f.write(f"\n# finished: {datetime.now().isoformat()}\n")
            f.write(f"# elapsed: {elapsed:.1f}s\n")
            f.write(f"# returncode: {proc.returncode}\n")
            return proc.returncode, elapsed
        except Exception as e:
            f.write(f"\n# EXCEPTION: {type(e).__name__}: {e}\n")
            return -1, time.time() - t0


def worker(gpu_id, job_q, result_q, log_dir, env_extra):
    while True:
        try:
            item = job_q.get_nowait()
        except queue.Empty:
            return
        job_id, name, cmd = item
        print(f"[gpu:{gpu_id}] START job {job_id:03d}: {name}", flush=True)
        rc, elapsed = run_job(job_id, name, cmd, gpu_id, log_dir, env_extra)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[gpu:{gpu_id}] {status:8} job {job_id:03d}: {name} ({elapsed:.0f}s)",
              flush=True)
        result_q.put((job_id, name, gpu_id, rc, elapsed))
        job_q.task_done()


def parse_args():
    p = argparse.ArgumentParser(description="parallel multi-GPU experiment dispatcher")
    p.add_argument("--stage", choices=("A", "B", "all"), default="A",
                   help="A=SAG-only experiments, B=baseline experiments, all=both sequentially")
    p.add_argument("--num_gpus", type=int, default=0,
                   help="number of GPUs to use; 0 = auto-detect")
    p.add_argument("--log_dir", default=None,
                   help="log directory (default: outputs/logs/parallel_<ts>)")
    p.add_argument("--dry_run", action="store_true",
                   help="print job list and exit")
    p.add_argument("--filter", default=None,
                   help="run only jobs whose name contains this substring")
    p.add_argument("--exclude", default=None,
                   help="skip jobs whose name contains this substring")
    p.add_argument("--tf1_python", default=os.environ.get("SAG_TF1_PYTHON"),
                   help="path to TF1.x python (for ComGA/DySAT). If unset, "
                        "stat_tf1 job will fail with a clear error.")
    p.add_argument("--llm_checkpoint", default=os.environ.get("SAG_LLM_CHECKPOINT"),
                   help="path to LLM checkpoint root (for LLM baselines)")
    return p.parse_args()


def select_jobs(jobs, args):
    out = []
    for i, (name, cmd) in enumerate(jobs):
        if args.filter and args.filter not in name:
            continue
        if args.exclude and args.exclude in name:
            continue
        out.append((i + 1, name, list(cmd)))
    return out


def main():
    args = parse_args()
    if args.stage == "A":
        jobs_raw = STAGE_A_JOBS
    elif args.stage == "B":
        jobs_raw = STAGE_B_JOBS
    else:
        jobs_raw = STAGE_A_JOBS + STAGE_B_JOBS
    jobs = select_jobs(jobs_raw, args)

    if args.num_gpus > 0:
        ngpu = args.num_gpus
    else:
        ngpu = detect_gpus()
        if ngpu == 0:
            print("WARNING: no CUDA GPU detected, will run on CPU with 1 worker",
                  flush=True)
            ngpu = 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_dir or os.path.join(REPO_ROOT, "outputs", "logs",
                                            f"parallel_{args.stage}_{ts}")
    os.makedirs(log_dir, exist_ok=True)

    env_extra = {}
    if args.tf1_python:
        env_extra["SAG_TF1_PYTHON"] = args.tf1_python
    if args.llm_checkpoint:
        env_extra["SAG_LLM_CHECKPOINT"] = args.llm_checkpoint

    print(f"[scheduler] stage={args.stage}  gpus={ngpu}  jobs={len(jobs)}", flush=True)
    print(f"[scheduler] log_dir={log_dir}", flush=True)
    if env_extra:
        print(f"[scheduler] env_extra={env_extra}", flush=True)
    print()

    print(f"=== job list ({len(jobs)}) ===")
    for jid, name, cmd in jobs:
        print(f"  {jid:03d}  {name:30s}  {' '.join(cmd[:5])}...")
    if args.dry_run:
        return

    job_q = queue.Queue()
    for j in jobs:
        job_q.put(j)
    result_q = queue.Queue()

    threads = []
    t0 = time.time()
    for gpu_id in range(ngpu):
        th = threading.Thread(target=worker,
                              args=(gpu_id, job_q, result_q, log_dir, env_extra),
                              daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    total_elapsed = time.time() - t0

    results = []
    while not result_q.empty():
        results.append(result_q.get())
    results.sort()

    print()
    print(f"=== summary (total wall-time: {total_elapsed:.0f}s) ===")
    n_ok = sum(1 for r in results if r[3] == 0)
    n_fail = sum(1 for r in results if r[3] != 0)
    print(f"  ok={n_ok}  fail={n_fail}  total={len(results)}")
    for jid, name, gpu, rc, elapsed in results:
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  {jid:03d}  gpu:{gpu}  {status:10}  {elapsed:6.0f}s  {name}")

    summary = {
        "stage": args.stage,
        "num_gpus": ngpu,
        "log_dir": log_dir,
        "total_wall_time_s": total_elapsed,
        "results": [
            {"job_id": jid, "name": name, "gpu_id": gpu,
             "returncode": rc, "elapsed_s": elapsed}
            for jid, name, gpu, rc, elapsed in results
        ],
    }
    summary_path = os.path.join(log_dir, "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[scheduler] summary written to {summary_path}")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
