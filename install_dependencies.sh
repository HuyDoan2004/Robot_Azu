#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./install_dependencies.sh [options]

Sets up Python dependencies for the whole Robot_Azu workspace.

Options:
  --apt               Install required Ubuntu packages via apt (uses sudo)
  --venv              Create/use a local virtualenv at .venv (default)
  --no-venv           Do not use a virtualenv; install with system python (uses --user when possible)
  --system-site       Create venv with --system-site-packages (helpful if ROS python packages are installed system-wide)
  --jetson-ort-gpu    Install onnxruntime-gpu from Jetson AI Lab extra index (uninstalls onnxruntime first)
  -h, --help          Show this help

Environment variables:
  PYTHON_BIN          Python executable to use (default: python3)
  VENV_DIR            Virtualenv directory (default: <repo>/.venv)
EOF
}

DO_APT=0
USE_VENV=1
VENV_SYSTEM_SITE=0
JETSON_ORT_GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apt) DO_APT=1; shift ;;
    --venv) USE_VENV=1; shift ;;
    --no-venv) USE_VENV=0; shift ;;
    --system-site) VENV_SYSTEM_SITE=1; shift ;;
    --jetson-ort-gpu) JETSON_ORT_GPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ $DO_APT -eq 1 ]]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    cmake git \
    python3 python3-pip python3-venv \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev alsa-utils \
    libsndfile1 \
    libgl1 libglib2.0-0
fi

# ================== DEPTH CAMERA DRIVER (CUSTOM) ==================
# Add your depth camera driver installation steps here.
#
# Why this exists:
# - Depth cameras often need vendor drivers/udev rules/kernel modules.
# - Those steps are very hardware- and OS-specific, so they are intentionally
#   left for you to customize.
#
# Suggested structure:
# - Prefer apt installs + udev rules inside the `--apt` block above.
# - Keep commands idempotent (safe to re-run).
#
# Example (keep commented):
#   # Intel RealSense (librealsense2):
#   # sudo apt-get install -y --no-install-recommends librealsense2-utils librealsense2-dev
#   # sudo apt-get install -y --no-install-recommends python3-pyrealsense2
# ================================================================

pip_install() {
  # shellcheck disable=SC2068
  "$@"
}

PIP_USER_FLAG=()
if [[ $USE_VENV -eq 1 ]]; then
  VENV_ARGS=()
  if [[ $VENV_SYSTEM_SITE -eq 1 ]]; then
    VENV_ARGS+=(--system-site-packages)
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$VENV_DIR"
  fi

  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
else
  # Installing into system python: prefer --user when not root.
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    PIP_USER_FLAG=(--user)
  fi
fi

PY="${PYTHON_BIN}"
if command -v python >/dev/null 2>&1; then
  PY=python
fi

# Upgrade pip tooling
pip_install "$PY" -m pip install "${PIP_USER_FLAG[@]}" --upgrade pip setuptools wheel

REQ_FILE="$ROOT_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Missing requirements.txt at repo root: $REQ_FILE"
  exit 1
fi

# Install Python requirements
pip_install "$PY" -m pip install "${PIP_USER_FLAG[@]}" --no-cache-dir -r "$REQ_FILE"

# Optional: Jetson Orin (JetPack 6 / CUDA 12.6) onnxruntime-gpu wheel
if [[ $JETSON_ORT_GPU -eq 1 ]]; then
  pip_install "$PY" -m pip uninstall -y onnxruntime || true
  pip_install "$PY" -m pip install "${PIP_USER_FLAG[@]}" --no-cache-dir \
    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ \
    onnxruntime-gpu==1.23.0
fi

cat <<EOF
Done.

Next steps:
- If you used venv: source "$VENV_DIR/bin/activate"
- For ROS 2 deps: install via apt/rosdep (e.g. nav2, rtabmap, cv_bridge, message packages)
EOF
