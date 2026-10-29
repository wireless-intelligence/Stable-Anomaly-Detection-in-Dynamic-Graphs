import os


def _auto_device():
    """Pick GPU if available; CPU otherwise. Override with SAG_DEVICE env var."""
    explicit = os.environ.get("SAG_DEVICE")
    if explicit:
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Config:
    seed = 42
    device = _auto_device()

    gamma = 0.95
    beta_upd = 1.0
    alpha = 1.0
    delta = 1.0
    rho = 2.0
    kernel_dim = 4
    kernel_lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
    time_mode = "global"
    prior_anomaly_rate = 0.1
    likelihood_anomaly = 0.9
    likelihood_normal = 0.1
    epsilon = 1e-6
    h_max = None
    window_size = 4096
    fusion_mode = "auto"
    has_ground_truth = False
    edge_order = "auto"
    state_gate_tau = 0.0

    num_epochs = 20
    learning_rate = 0.001
    weight_decay = 0.0
    dropout = 0.0

    n_runs = 10
    eval_freq = 1
    save_freq = 9999

    data_dir = "./data"
    output_dir = "./outputs"
    log_dir = "./outputs/logs"
    checkpoint_dir = "./outputs/checkpoints"
    result_dir = "./outputs/results"
    figure_dir = "./outputs/figures"
    vis_dir = "./outputs/visualizations"

    @classmethod
    def create_dirs(cls):
        for d in [cls.output_dir, cls.log_dir, cls.checkpoint_dir,
                  cls.result_dir, cls.figure_dir, cls.vis_dir]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def to_dict(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(getattr(cls, k))
        }

    @classmethod
    def from_dict(cls, d):
        for k, v in d.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
        return cls
