#!/usr/bin/env bash
set -euo pipefail

if [[ "${BSVRPP_SETUP_DEBUG:-0}" == "1" ]]; then
  set -x
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
MODELS_DIR="${MODELS_DIR:-${WORKSPACE_ROOT}/models}"
TMP_PRE_DIR="${TMP_PRE_DIR:-${WORKSPACE_ROOT}/tmp_pre}"
TMP_SR_DIR="${TMP_SR_DIR:-${WORKSPACE_ROOT}/tmp_sr}"
OUT_DIR="${OUT_DIR:-${WORKSPACE_ROOT}/out}"
SAMPLE_DIR="${SAMPLE_DIR:-${WORKSPACE_ROOT}/sample}"

BASICVSRPP_REPO_URL="${BASICVSRPP_REPO_URL:-https://github.com/0xmrwn/BasicVSR_PlusPlus}"
BASICVSRPP_DIR="${BASICVSRPP_DIR:-/opt/BasicVSR_PlusPlus}"

VENV_DIR="${BSVRPP_VENV_DIR:-/opt/bsvrpp-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_FFMPEG="${INSTALL_FFMPEG:-auto}"

log() { printf '\n[%s] %s\n' "setup" "$*"; }
warn() { printf '\n[%s] %s\n' "setup:warn" "$*" >&2; }

ensure_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    warn "Required command '$1' not found in PATH."
    exit 1
  fi
}

install_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg already present."
    return
  fi

  if [[ "$INSTALL_FFMPEG" == "skip" ]]; then
    warn "ffmpeg missing and INSTALL_FFMPEG=skip; install manually if needed."
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    log "Installing ffmpeg via sudo apt-get."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
  elif [[ $EUID -eq 0 ]]; then
    log "Installing ffmpeg via apt-get."
    apt-get update
    apt-get install -y ffmpeg
  else
    warn "ffmpeg missing and sudo not available; install manually or rerun with sudo."
  fi
}

TORCH_PACKAGES=(
  "torch==2.6.0+cu124"
  "torchvision==0.21.0+cu124"
  "torchaudio==2.6.0+cu124"
)

BASE_PACKAGES=(
  "numpy==2.2.6"
  "scipy==1.16.2"
  "opencv-python==4.12.0.88"
)

log "Preparing RunPod environment (workspace root: ${WORKSPACE_ROOT})."

ensure_command "$PYTHON_BIN"
ensure_command git
install_ffmpeg

log "Ensuring workspace directories exist."
mkdir -p "$MODELS_DIR" "$TMP_PRE_DIR" "$TMP_SR_DIR" "$OUT_DIR" "$SAMPLE_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating Python venv at $VENV_DIR."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Reusing existing venv at $VENV_DIR."
fi

PYTHON_VENV_BIN="$VENV_DIR/bin/python"
PIP_CMD=("$PYTHON_VENV_BIN" -m pip)

"${PIP_CMD[@]}" install --upgrade pip setuptools wheel

log "Installing PyTorch stack (${TORCH_PACKAGES[*]})."
"${PIP_CMD[@]}" install --index-url "$TORCH_INDEX_URL" "${TORCH_PACKAGES[@]}"

log "Installing core scientific packages."
"${PIP_CMD[@]}" install "${BASE_PACKAGES[@]}"

log "Building mmcv-full==1.7.2 from source."
"${PIP_CMD[@]}" install --no-binary mmcv-full "mmcv-full==1.7.2"

if [[ ! -d "$BASICVSRPP_DIR/.git" ]]; then
  log "Cloning BasicVSR++ repo from $BASICVSRPP_REPO_URL."
  git clone "$BASICVSRPP_REPO_URL" "$BASICVSRPP_DIR"
else
  log "BasicVSR++ repo already present at $BASICVSRPP_DIR."
  if [[ "${BASICVSRPP_SKIP_PULL:-0}" != "1" ]]; then
    log "Fetching latest changes."
    if ! git -C "$BASICVSRPP_DIR" pull --ff-only; then
      warn "git pull failed; resolve manually if needed."
    fi
  fi
fi

log "Installing BasicVSR++ in editable mode (no deps)."
"${PIP_CMD[@]}" install --no-deps -e "$BASICVSRPP_DIR"

log "Applying BasicVSR++ patches for MMCV and NumPy."
PATCH_ROOT="$BASICVSRPP_DIR" "$PYTHON_VENV_BIN" <<'PY'
import os
from pathlib import Path

root = Path(os.environ["PATCH_ROOT"])

init_path = root / "mmedit" / "__init__.py"
text = init_path.read_text()
if "MMCV_MAX = '1.6'" in text and "MMCV_MAX = '1.8'" not in text:
    init_path.write_text(text.replace("MMCV_MAX = '1.6'", "MMCV_MAX = '1.8'"))
    print(" - Updated MMCV_MAX to 1.8")
else:
    print(" - MMCV_MAX already >= 1.8")

utils_path = root / "mmedit" / "datasets" / "pipelines" / "utils.py"
text = utils_path.read_text()
needle = "    np.bool8: (False, True),"
replacement = "    # np.bool8 removed for NumPy 2.x"
if needle in text:
    utils_path.write_text(text.replace(needle, replacement))
    print(" - Commented np.bool8 entry")
elif replacement in text:
    print(" - np.bool8 entry already commented")
else:
    print(" - np.bool8 entry not found; check file manually")
PY

log "Verifying Torch and mmcv installs."
"$PYTHON_VENV_BIN" - <<'PY'
import torch, mmcv
from mmcv.ops import deform_conv2d
print(" - Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print(" - MMCV:", mmcv.__version__, "ops OK")
PY

cat <<EOF

RunPod setup complete.

Activate the environment with:
  source "$VENV_DIR/bin/activate"

Example streaming run (replace paths as needed):
  python /workspace/restoration_video_streaming.py \\
    --config ${BASICVSRPP_DIR}/configs/basicvsr_plusplus_reds4.py \\
    --checkpoint /workspace/models/basicvsr_plusplus_reds4.pth \\
    --input /workspace/tmp_pre/INPUT.mov \\
    --output /workspace/tmp_sr \\
    --max-seq-len 250 \\
    --device 0
EOF
