from .sag import SAG, SAGNoStable, SAGNoDecay, SAGNoExposure, SAGNoKernel, create_sag_model


ABLATION_TYPES = {
    "full": None,
    "no_stable": "no_stable",
    "no_decay": "no_decay",
    "no_exposure": "no_exposure",
    "no_kernel": "no_kernel",
    "no_attribute": "no_attribute",
    "no_structure": "no_structure",
    "no_selection": "no_selection",
}

ABLATION_NAMES = {
    "full": "SAG (Full)",
    "no_stable": "w/o Stable Update",
    "no_decay": "w/o Temporal Decay",
    "no_exposure": "w/o Exposure Scaling",
    "no_kernel": "w/o Kernel Sketch",
    "no_attribute": "w/o Attribute Evidence",
    "no_structure": "w/o Structural Evidence",
    "no_selection": "w/o Evidence Selection",
}

ABLATION_CLASSES = {
    "full": SAG,
    "no_stable": SAGNoStable,
    "no_decay": SAGNoDecay,
    "no_exposure": SAGNoExposure,
    "no_kernel": SAGNoKernel,
}


def get_ablation_models(num_nodes, config, features=None):
    out = {}
    for key, ablation in ABLATION_TYPES.items():
        out[key] = create_sag_model(num_nodes, config, features=features, ablation=ablation)
    return out


def get_ablation_name(key: str) -> str:
    return ABLATION_NAMES.get(key, key)
