#!/usr/bin/env bash
# Vendor official baseline repositories for fully-faithful gradient-trained
# implementations. Run AFTER the main setup_cloud.sh.
#
# Each repo is shallow-cloned (--depth 1) into vendor/<NAME>/. Wrappers in
# src/baselines/<name>.py import from there and fall back to a clean PyTorch
# implementation if the vendor dir is missing.

set -e
mkdir -p vendor
cd vendor

clone_or_skip() {
  local name="$1"
  local url="$2"
  if [ -d "$name" ]; then
    echo "==> [$name] already vendored"
  else
    echo "==> [$name] cloning $url"
    git clone --depth 1 "$url" "$name" || echo "  (failed; baseline will use PyTorch fallback)"
  fi
}

# --- GNN-based ---
clone_or_skip "ConsisGAD"  "https://github.com/Xtra-Computing/ConsisGAD.git"
clone_or_skip "ANEMONE"    "https://github.com/GRAND-Lab/ANEMONE.git"
clone_or_skip "ComGA"      "https://github.com/DASE4/ComGA.git"
clone_or_skip "PMP"        "https://github.com/Xtra-Computing/PMP.git"
clone_or_skip "GAAP"       "https://github.com/jdduan-fudan/GAAP.git"
clone_or_skip "Graphormer" "https://github.com/microsoft/Graphormer.git"

# --- Dynamic / temporal-graph baselines (official) ---
# DyGLib ships the original authors' JODIE / DyRep / TGAT / TGN (+ CAWN / TCL /
# GraphMixer / DyGFormer). src/baselines/dynamic_official.py imports these
# official model classes. EvolveGCN uses the maintained PyG-Temporal library
# (pip, below) -- no clone needed.
clone_or_skip "DyGLib"     "https://github.com/yule-BUAA/DyGLib.git"

# --- LLM-based (heavy: each needs LLaMA-2 weights separately) ---
clone_or_skip "TAPE"        "https://github.com/XiaoxinHe/TAPE.git"
clone_or_skip "GraphGPT"    "https://github.com/HKUDS/GraphGPT.git"
clone_or_skip "HiGPT"       "https://github.com/HKUDS/HiGPT.git"
clone_or_skip "InstructGLM" "https://github.com/agiresearch/InstructGLM.git"

cd ..
cat <<'EOF'

==> Vendor setup done.

==> NOTE for LLM baselines (TAPE / GraphGPT / HiGPT / InstructGLM):
    The cloned repos are PyTorch + transformers code only. They require
    LLaMA-2 / GPT-NeoX checkpoints (multi-GB) downloaded separately:

      huggingface-cli download meta-llama/Llama-2-7b-hf

    Place the weights under vendor/<name>/checkpoints/ following each
    repo's README. Without weights, our wrappers fall back to a fully
    PyTorch-trained GNN-only proxy (clearly labeled in logs).

==> All non-LLM baselines (ConsisGAD/ANEMONE/ComGA/PMP/GAAP/Graphormer)
    work end-to-end with our gradient-trained reference implementations,
    independent of vendor/.

==> DYNAMIC baselines (TGN/JODIE/DyRep/TGAT): official code from vendor/DyGLib.
    EvolveGCN: pip install torch-geometric-temporal
    Without vendor/DyGLib, use the clone-free *-reimpl variants
    (e.g. --methods SAG,TGN-reimpl,JODIE-reimpl,...).
EOF
