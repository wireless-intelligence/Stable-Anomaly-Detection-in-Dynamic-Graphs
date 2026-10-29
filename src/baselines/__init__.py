import os
import warnings

from .base import BaselineModel
from .midas import MIDASBaseline
from .netwalk import NetWalkBaseline
from .sedanspot import SedanSpotBaseline
from .vendor_official import (
    VENDOR_OFFICIAL_BASELINES,
    VENDOR_REQUIREMENTS,
    ANEMONEOfficial,
    ConsisGADOfficial,
    PMPOfficial,
    GAAPOfficial,
    ComGAOfficial,
    DySATOfficial,
    GraphormerOfficial,
    GraphGPTOfficial,
    HiGPTOfficial,
    TAPEOfficial,
    InstructGLMOfficial,
)


_VERBOSE = os.environ.get("SAG_BASELINES_VERBOSE", "1") not in ("0", "false", "False", "")


# Dynamic (temporal) baselines. Official DyGLib/PyG-Temporal implementations
# take the canonical names (TGN/JODIE/DyRep/TGAT/EvolveGCN); torch-only in-repo
# reimplementations register under `*-reimpl` as a clone-free fallback.
try:
    from .dynamic_official import (OFFICIAL_DYNAMIC_BASELINES,
                                   OFFICIAL_DYNAMIC_REQUIREMENTS)
except Exception as _e:  # pragma: no cover - defensive
    OFFICIAL_DYNAMIC_BASELINES = {}
    OFFICIAL_DYNAMIC_REQUIREMENTS = {}
    if os.environ.get("SAG_BASELINES_VERBOSE", "1") not in ("0", "false", "False", ""):
        warnings.warn(f"[SAG.baselines] official dynamic baselines unavailable: "
                      f"{type(_e).__name__}: {_e}")

try:
    from .dynamic import REIMPL_DYNAMIC_BASELINES, REIMPL_REQUIREMENTS
except Exception as _e:  # pragma: no cover - defensive
    REIMPL_DYNAMIC_BASELINES = {}
    REIMPL_REQUIREMENTS = {}
    if os.environ.get("SAG_BASELINES_VERBOSE", "1") not in ("0", "false", "False", ""):
        warnings.warn(f"[SAG.baselines] reimpl dynamic baselines unavailable: "
                      f"{type(_e).__name__}: {_e}")

DYNAMIC_BASELINES = {**OFFICIAL_DYNAMIC_BASELINES, **REIMPL_DYNAMIC_BASELINES}
DYNAMIC_REQUIREMENTS = {**OFFICIAL_DYNAMIC_REQUIREMENTS, **REIMPL_REQUIREMENTS}


def list_dynamic_baselines():
    return sorted(DYNAMIC_BASELINES.keys())


def _try_import_official():
    try:
        from .official import OFFICIAL_BASELINES
        return OFFICIAL_BASELINES
    except Exception as e:
        if _VERBOSE:
            warnings.warn(f"[SAG.baselines] OFFICIAL_BASELINES module failed to import: "
                          f"{type(e).__name__}: {e}")
        return {}


OFFICIAL_BASELINES = _try_import_official()


def _smoke_test_official():
    import numpy as np
    import scipy.sparse as sp
    from configs import Config
    dropped = {}
    for name, cls in list(OFFICIAL_BASELINES.items()):
        try:
            n = 32
            rows = np.random.default_rng(0).integers(0, n, size=80)
            cols = np.random.default_rng(1).integers(0, n, size=80)
            data = np.ones(80, dtype=np.float32)
            A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
            A = A.maximum(A.T)
            A.setdiag(0); A.eliminate_zeros()
            feat = sp.csr_matrix(np.random.default_rng(2).standard_normal((n, 6)).astype(np.float32))
            m = cls(num_nodes=n, config=Config, features=feat, seed=0)
            m.fit(A, np.array([[0, 1]], dtype=np.int64), np.zeros(n, dtype=np.int64))
            m.score()
        except Exception as e:
            dropped[name] = f"{type(e).__name__}: {e}"
    for name in dropped:
        OFFICIAL_BASELINES.pop(name, None)
    if dropped and _VERBOSE:
        msg = "; ".join(f"{k} ({v})" for k, v in dropped.items())
        warnings.warn(f"[SAG.baselines] dropped from OFFICIAL_BASELINES: {msg}")
    return dropped


_OFFICIAL_DROPPED = {}
try:
    _OFFICIAL_DROPPED = _smoke_test_official()
except Exception as e:
    if _VERBOSE:
        warnings.warn(f"[SAG.baselines] _smoke_test_official failed: "
                      f"{type(e).__name__}: {e}")


try:
    import torch_sparse  # noqa
    _HAVE_TORCH_SPARSE = True
except Exception:
    _HAVE_TORCH_SPARSE = False
if not _HAVE_TORCH_SPARSE:
    if "DOMINANT" in OFFICIAL_BASELINES and _VERBOSE:
        warnings.warn("[SAG.baselines] torch_sparse unavailable; PyGOD DOMINANT "
                      "may degrade or be unavailable")


BASELINES = {
    **OFFICIAL_BASELINES,
    "MIDAS":     MIDASBaseline,
    "NetWalk":   NetWalkBaseline,
    "SedanSpot": SedanSpotBaseline,
    **VENDOR_OFFICIAL_BASELINES,
}

# Native streaming/temporal baselines that consume the real timestamped stream.
# MIDAS/SedanSpot/NetWalk are edge-stream methods; TGN/JODIE are temporal GNNs.
STREAMING_BASELINES = {"MIDAS", "NetWalk", "SedanSpot"}


BASELINE_PROVENANCE = {
    "MIDAS":     "in-repo faithful re-implementation of count-min sketch (MIDAS-N variant)",
    "NetWalk":   "in-repo faithful re-implementation of random-walk co-occurrence + SVD",
    "SedanSpot": "in-repo faithful re-implementation",
    "ANEMONE":   "vendor official: vendor/ANEMONE/model.py Model class + faithful training loop",
    "ConsisGAD": "vendor official: vendor/ConsisGAD/models.py simpleGNN_MR class",
    "PMP":       "vendor official: vendor/PMP/model/SAGE.py or LASAGE_S.py",
    "GAAP":      "vendor official: vendor/GAAP/mycode/nn/gnn.py SAGEConv",
    "ComGA":     "subprocess to vendor/ComGA/run.py (TF1.x via SAG_TF1_PYTHON)",
    "DySAT":     "subprocess to vendor/DySAT (TF1.x via SAG_TF1_PYTHON)",
    "Graphormer":"subprocess to vendor/Graphormer (fairseq required)",
    "GraphGPT":  "subprocess + LLM checkpoint at vendor/llama-2-7b-hf/",
    "HiGPT":     "subprocess + LLM checkpoint at vendor/Qwen3-8B/",
    "TAPE":      "subprocess; uses vendor/TAPE GPT-augmented features path",
    "InstructGLM": "subprocess + Llama checkpoint",
}


def list_baselines():
    return sorted(BASELINES.keys())


def baseline_provenance():
    return {
        "registered": sorted(BASELINES.keys()),
        "pygod_official": sorted(OFFICIAL_BASELINES.keys()),
        "pygod_dropped": dict(_OFFICIAL_DROPPED),
        "streaming_native": ["MIDAS", "SedanSpot"],
        "dynamic_temporal": sorted(DYNAMIC_BASELINES.keys()),
        "dynamic_official": sorted(OFFICIAL_DYNAMIC_BASELINES.keys()),
        "dynamic_reimpl": sorted(REIMPL_DYNAMIC_BASELINES.keys()),
        "dynamic_requirements": dict(DYNAMIC_REQUIREMENTS),
        "vendor_official": sorted(VENDOR_OFFICIAL_BASELINES.keys()),
        "vendor_requirements": dict(VENDOR_REQUIREMENTS),
        "torch_sparse_available": _HAVE_TORCH_SPARSE,
        "provenance": dict(BASELINE_PROVENANCE),
    }


__all__ = [
    "BaselineModel",
    "BASELINES",
    "OFFICIAL_BASELINES",
    "DYNAMIC_BASELINES",
    "DYNAMIC_REQUIREMENTS",
    "OFFICIAL_DYNAMIC_BASELINES",
    "REIMPL_DYNAMIC_BASELINES",
    "STREAMING_BASELINES",
    "list_dynamic_baselines",
    "VENDOR_OFFICIAL_BASELINES",
    "VENDOR_REQUIREMENTS",
    "BASELINE_PROVENANCE",
    "MIDASBaseline",
    "NetWalkBaseline",
    "SedanSpotBaseline",
    "ANEMONEOfficial",
    "ConsisGADOfficial",
    "PMPOfficial",
    "GAAPOfficial",
    "ComGAOfficial",
    "DySATOfficial",
    "GraphormerOfficial",
    "GraphGPTOfficial",
    "HiGPTOfficial",
    "TAPEOfficial",
    "InstructGLMOfficial",
    "list_baselines",
    "baseline_provenance",
]
