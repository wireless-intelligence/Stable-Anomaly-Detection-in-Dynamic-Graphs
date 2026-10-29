import os
import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_synthetic_data(num_nodes=500, num_features=100, num_edges=2000,
                          num_anomalies=50, save_path="./data/Synthetic"):
    print(f"Creating synthetic dataset with {num_nodes} nodes...")
    rng = np.random.default_rng(42)
    row = rng.integers(0, num_nodes, num_edges)
    col = rng.integers(0, num_nodes, num_edges)
    data = np.ones(num_edges)
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    adj.setdiag(0)
    adj.eliminate_zeros()

    features = sp.csr_matrix(rng.standard_normal((num_nodes, num_features)))
    labels = np.zeros((num_nodes, 1), dtype=np.int64)
    labels[rng.choice(num_nodes, num_anomalies, replace=False)] = 1

    os.makedirs(save_path, exist_ok=True)
    sio.savemat(
        os.path.join(save_path, "Synthetic.mat"),
        {"Network": adj, "Attributes": features, "Label": labels},
    )
    return save_path


def test_imports():
    print("Testing imports...")
    from configs import Config, get_dataset_config        
    print("  [ok] configs")
    from src.data import DataLoader, MatDataLoader, AnomalyInjector        
    print("  [ok] src.data")
    from src.models import SAG, SAGNoStable, SAGNoDecay, SAGNoExposure, create_sag_model        
    print("  [ok] src.models")
    from src.trainer import SAGTrainer, MultiRunSAGTrainer, SAG_FACTORY        
    print("  [ok] src.trainer")
    from src.evaluation import compute_all_metrics, Evaluator        
    print("  [ok] src.evaluation")
    from src.visualization import plot_training_curves, create_d3_visualization        
    print("  [ok] src.visualization")
    from src.utils import set_seed, Timer, MemoryTracker, save_json        
    print("  [ok] src.utils")
    from src.baselines import BASELINES        
    print(f"  [ok] src.baselines ({len(BASELINES)} baselines)")


def test_data_loading():
    print("\nTesting data loading...")
    from src.data import DataLoader

    create_synthetic_data()
    loader = DataLoader(data_dir="./data", dataset_name="Synthetic", inject_anomaly=False, seed=42)
    loader.load()
    stats = loader.get_stats()
    print(f"  nodes={stats['num_nodes']} edges={stats['num_edges']} feats={stats['num_features']} anomalies={stats['num_anomalies']}")
    edges = loader.get_edge_stream()
    print(f"  edge stream length: {edges.shape[0]}")
    return loader


def test_model():
    print("\nTesting SAG model...")
    from src.models import create_sag_model
    from configs import Config

    model = create_sag_model(num_nodes=500, config=Config)
    print(f"  model class: {type(model).__name__}")
    model.reset()
    for u, v in [(0, 1), (1, 2), (2, 3), (0, 3), (1, 4)]:
        model.process_edge(u, v)
    scores = model.get_scores()
    print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")


def test_training():
    print("\nTesting training pipeline...")
    from src.data import DataLoader
    from src.trainer import SAGTrainer, SAG_FACTORY
    from configs import Config

    Config.num_epochs = 3
    Config.eval_freq = 1
    loader = DataLoader(data_dir="./data", dataset_name="Synthetic", inject_anomaly=False, seed=42)
    loader.load()
    model = SAG_FACTORY(num_nodes=loader.stats["num_nodes"], config=Config, features=loader.features, adj=loader.adj)
    trainer = SAGTrainer(model, loader, Config)
    history = trainer.train(num_epochs=3, eval_freq=1, save_freq=100)
    if history["metrics"]:
        m = history["metrics"][-1]
        print(f"  final AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} F1={m['macro_f1']:.4f}")


def test_visualization():
    print("\nTesting visualization...")
    from src.visualization import plot_score_distribution

    os.makedirs("./outputs/test_figures", exist_ok=True)
    rng = np.random.default_rng(0)
    scores = rng.random(500)
    labels = np.zeros(500)
    labels[:50] = 1
    path = "./outputs/test_figures/test_distribution.png"
    plot_score_distribution(scores, labels, path, dataset_name="Test")
    print(f"  saved: {os.path.exists(path)}")


def test_ablations():
    print("\nTesting ablation variants...")
    from src.models import create_sag_model, ABLATION_TYPES
    from configs import Config

    for key in ABLATION_TYPES:
        m = create_sag_model(num_nodes=100, config=Config, ablation=ABLATION_TYPES[key])
        m.reset()
        for u, v in [(0, 1), (1, 2), (2, 0)]:
            m.process_edge(u, v)
        s = m.get_scores()
        print(f"  [{key:<14}] ok, scores in [{s.min():.3f}, {s.max():.3f}]")


if __name__ == "__main__":
    test_imports()
    test_data_loading()
    test_model()
    test_training()
    test_visualization()
    test_ablations()
    print("\nAll tests passed.")
