#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./install_dependencies_full.sh [options]

Installs dependencies for this repo.

Options:
  --apt               Install required Ubuntu packages via apt (uses sudo)
  --venv              Create/use a local virtualenv at .venv
  --no-venv           Do not use a virtualenv; install with system python (default, uses --user when possible)
  --system-site       Create venv with --system-site-packages (useful if ROS python packages are installed system-wide)
  --jetson-ort-gpu    Install onnxruntime-gpu from Jetson AI Lab extra index (uninstalls onnxruntime first)
  --realsense         Build+install librealsense from source (uses sudo)
  -h, --help          Show this help

Environment variables:
  PYTHON_BIN          Python executable to use (default: python3)
  VENV_DIR            Virtualenv directory (default: <repo>/.venv)
EOF
}

DO_APT=0
# Default: match typical Jetson/ROS2 setup (system python + --user), no venv.
USE_VENV=0
VENV_SYSTEM_SITE=0
JETSON_ORT_GPU=0
DO_REALSENSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apt) DO_APT=1; shift ;;
    --venv) USE_VENV=1; shift ;;
    --no-venv) USE_VENV=0; shift ;;
    --system-site) VENV_SYSTEM_SITE=1; shift ;;
    --jetson-ort-gpu) JETSON_ORT_GPU=1; shift ;;
    --realsense) DO_REALSENSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

is_jetson() {
  [[ "$(uname -m)" == "aarch64" ]] && [[ -f /etc/nv_tegra_release ]]
}

if [[ $DO_APT -eq 1 ]]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    ca-certificates curl \
    cmake git pkg-config build-essential \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev alsa-utils \
    libsndfile1 \
    libgl1 libglib2.0-0
fi

install_librealsense() {
  local src_dir="$ROOT_DIR/_deps/librealsense"

  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found. Re-run with --apt first." >&2
    exit 1
  fi

  if [[ ! -d "$src_dir/.git" ]]; then
    mkdir -p "$(dirname "$src_dir")"
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git "$src_dir"
  fi

  pushd "$src_dir" >/dev/null
  git submodule update --init --recursive

  mkdir -p build
  pushd build >/dev/null
  cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDINGS=OFF
  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
  popd >/dev/null

  popd >/dev/null
}

if [[ $DO_REALSENSE -eq 1 ]]; then
  # librealsense needs build tools and system libs.
  # If you didn't pass --apt, we'll still attempt install; it will fail fast if deps are missing.
  install_librealsense
fi

pip_install() {
  "$@"
}

PIP_USER_FLAG=()
PY="$PYTHON_BIN"

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
  PY=python
else
  # Installing into system python: prefer --user when not root.
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    PIP_USER_FLAG=(--user)
  fi
fi

# Upgrade pip tooling
pip_install "$PY" -m pip install "${PIP_USER_FLAG[@]}" --upgrade pip setuptools wheel

REQ_FILE="$ROOT_DIR/requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "Missing requirements.txt at repo root: $REQ_FILE" >&2
  exit 1
fi

# Optional: Jetson Orin (JetPack 6 / CUDA 12.6) onnxruntime-gpu wheel
if [[ $JETSON_ORT_GPU -eq 1 ]]; then
  if ! is_jetson; then
    echo "--jetson-ort-gpu was set but this doesn't look like a Jetson (aarch64 + /etc/nv_tegra_release)." >&2
  fi
  pip_install "$PY" -m pip uninstall -y onnxruntime || true
  pip_install "$PY" -m pip install "${PIP_USER_FLAG[@]}" --no-cache-dir \
    --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/ \
    onnxruntime-gpu==1.23.0
fi

# Install Python requirements
pip_install "$PY" -m pip install "${PIP_USER_FLAG[@]}" --no-cache-dir -r "$REQ_FILE"

cat <<EOF
Done.

Next steps:
- Run: python3 check_env.py
- Run: python3 assistant2.py

ROS2 note:
- If mapping launch fails, run in a shell with:
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash
EOF
