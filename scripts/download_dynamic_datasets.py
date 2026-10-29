"""Helper for the real dynamic-graph datasets.

Most of these are license-gated (DGraphFin) or large repos (Elliptic++), so this
script does NOT bulk-download; it prints per-dataset instructions and, when a
dataset is already present, verifies the layout that ``TemporalDataLoader``
expects. See docs/DYNAMIC_DATASETS.md for the full guide.

    python scripts/download_dynamic_datasets.py --inventory
    python scripts/download_dynamic_datasets.py --check DGraphFin --data_dir ./data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


INSTRUCTIONS = {
    "DGraphFin": (
        "Register + download dgraphfin.npz from https://dgraph.xinye.com/dataset\n"
        "  -> place at data/DGraphFin/dgraphfin.npz"
    ),
    "Elliptic-tx": (
        "https://github.com/git-disl/EllipticPlusPlus (or Kaggle elliptic-data-set)\n"
        "  -> data/Elliptic++/{elliptic_txs_edgelist,features,classes}.csv"
    ),
    "Elliptic-wallet": (
        "https://github.com/git-disl/EllipticPlusPlus\n"
        "  -> data/Elliptic++/{AddrAddr_edgelist,wallets_features,wallets_classes}.csv"
    ),
    "AMLSim": (
        "Generate with https://github.com/IBM/AMLSim\n"
        "  -> data/AMLSim/{transactions,accounts,sar_accounts}.csv"
    ),
    "Bitcoin-Alpha": "SNAP soc-sign-bitcoinalpha -> data/Bitcoin-Alpha/bitcoin-alpha.csv",
    "Bitcoin-OTC": "SNAP soc-sign-bitcoinotc -> data/Bitcoin-OTC/bitcoin-otc.csv",
    "UCI": "http://konect.cc/networks/opsahl-ucsocial -> data/UCI/uci.csv",
    "Email-DNC": "http://konect.cc/networks/dnc-temporalGraph -> data/Email-DNC/email-dnc.csv",
}


def inventory():
    print("Real dynamic-graph datasets (place under --data_dir):\n")
    for name, msg in INSTRUCTIONS.items():
        print(f"[{name}]\n  {msg}\n")
    print("Then run, e.g.:")
    print("  python experiments/run_dynamic_comparison.py --dataset DGraphFin \\")
    print("      --methods SAG,TGN,JODIE,MIDAS,SedanSpot,DOMINANT --n_runs 3")


def check(name, data_dir):
    from src.data import TemporalDataLoader
    try:
        loader = TemporalDataLoader(data_dir=data_dir, dataset_name=name).load()
    except Exception as e:
        print(f"[{name}] NOT READY: {type(e).__name__}: {e}")
        print("  " + INSTRUCTIONS.get(name, "see docs/DYNAMIC_DATASETS.md"))
        return 1
    s = loader.get_stats()
    print(f"[{name}] OK")
    for k in ("num_nodes", "num_edges", "num_features", "num_eval_nodes",
              "num_anomalies", "anomaly_ratio", "time_span", "synthetic_labels"):
        print(f"    {k}: {s.get(k)}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", action="store_true")
    p.add_argument("--check", type=str, default=None)
    p.add_argument("--data_dir", type=str, default="./data")
    args = p.parse_args()
    if args.check:
        raise SystemExit(check(args.check, args.data_dir))
    inventory()


if __name__ == "__main__":
    main()
