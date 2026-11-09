#!/usr/bin/env bash
# Install Cosmos Transfer1 (inference only) for an RTX 5090 in a dedicated conda env. CPU-only work.
# Follows INSTALL.md at the commit Isaac Lab's augmented-imitation guide pins (e4055e39), with four
# deliberate deviations for Blackwell (sm_120), recorded in docs/writeup.md (Cosmos Transfer1 on a 32 GB card) and
# as rows T16 to T19 of docs/tutorial.md:
#   1. CUDA 12.8 toolchain from conda-forge instead of the yaml's 12.4 (nvcc 12.4 cannot target sm_120).
#   2. torch 2.7.0 / torchvision 0.22.0 from the PyTorch cu128 index (PyPI's 2.7.0 is a cu126 build
#      without sm_120). The flashinfer wheel NVIDIA pins is the cu128/torch2.7 build, so this is consistent.
#   3. The PyPI `apex` line is dropped: it is an unrelated package, and NVIDIA Apex is only imported for training.
#   4. An activate.d hook puts the pip cuDNN 9 libraries on LD_LIBRARY_PATH, and transformers is pinned back to
#      4.49.0 after vllm upgrades it.
# Usage: scripts/setup/install_cosmos_transfer1.sh [repo_dir]   (default ~/cosmos-transfer1)
set -euo pipefail
REPO="${1:-$HOME/cosmos-transfer1}"
COMMIT=e4055e39ee9c53165e85275bdab84ed20909714a
ENV=cosmos-transfer1
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

if [ ! -d "$REPO/.git" ]; then
  log "cloning cosmos-transfer1 @ $COMMIT"
  git clone https://github.com/nvidia-cosmos/cosmos-transfer1.git "$REPO"
fi
cd "$REPO"
git checkout -q "$COMMIT"
git submodule update --init --recursive
log "repo at $(git rev-parse --short HEAD)"

cat > "$REPO/cosmos-transfer1-cu128.yaml" <<'YAML'
name: cosmos-transfer1
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip=25.0
  - cmake
  - ninja
  - libgl
  - ffmpeg
  - gcc=12.4.0
  - gxx=12.4.0
  - cuda=12.8
  - cuda-nvcc=12.8
  - cuda-toolkit=12.8
YAML

# conda base: CONDA_BASE if set, else `conda info --base`, else ~/anaconda3
source "${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")}/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^$ENV\s"; then
  log "creating conda env $ENV (CUDA 12.8 toolchain)"
  conda env create --file "$REPO/cosmos-transfer1-cu128.yaml"
fi
set +u; conda activate "$ENV"; set -u   # the gcc activation hook references an unset SYS_SYSROOT
export CUDA_HOME="$CONDA_PREFIX"
log "python $(python -V 2>&1), nvcc $(nvcc --version | grep -oE 'release [0-9.]+')"

log "torch 2.7.0+cu128"
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, 'archs', torch.cuda.get_arch_list())"

log "requirements.txt (minus apex: the PyPI 'apex==0.9.10dev' pin is an unrelated web framework whose build fails; NVIDIA Apex is only imported by training code)"
grep -vE '^apex==' requirements.txt > requirements-inference.txt
pip install -r requirements-inference.txt

log "flashinfer + vllm + decord (per INSTALL.md)"
pip install "https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl"
pip install vllm==0.9.0
pip install decord==0.6.0

log "transformer-engine (torch extension built with nvcc 12.8 for sm_120; 20-60 min)"
ln -sf "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/include/* "$CONDA_PREFIX/include/" || true
ln -sf "$CONDA_PREFIX"/lib/python3.12/site-packages/nvidia/*/include/* "$CONDA_PREFIX/include/python3.12" || true
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation "transformer-engine[pytorch]==2.4" || pip install --no-build-isolation "transformer-engine[pytorch]"

log "post-fixes: cuDNN loader path for transformer-engine/megatron, transformers back to NVIDIA's pin"
# transformer_engine_cu12 and megatron dlopen libcudnn_graph.so.9 directly; it lives in the pip nvidia-cudnn wheel
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz-nvidia-wheel-libs.sh" <<'HOOK'
export LD_LIBRARY_PATH="$(ls -d $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
HOOK
export LD_LIBRARY_PATH="$(ls -d $CONDA_PREFIX/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
# vllm 0.9.0 upgrades transformers to 4.57, which breaks vllm's own import ('aimv2' already registered); 4.49.0 works
pip install transformers==4.49.0

log "huggingface_hub for the checkpoint downloader"
pip install "huggingface_hub[cli]"

log "environment check (CPU parts only; GPU check is run separately when the GPU is free)"
PYTHONPATH=$(pwd) python scripts/test_environment.py || log "test_environment.py reported a problem (see above)"
python - <<'EOF'
import torch, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda, "archs", torch.cuda.get_arch_list())
for m in ("transformer_engine", "flashinfer", "vllm", "decord", "sam2", "megatron.core"):
    try:
        importlib.import_module(m); print("import ok:", m)
    except Exception as e:
        print("import FAILED:", m, "->", str(e)[:120])
EOF
log "done"
