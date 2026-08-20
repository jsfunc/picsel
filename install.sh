#!/usr/bin/env bash
# Installation script for picSel.
# Checks requirements and creates a Python virtual environment with dependencies installed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
MIN_PY_MAJOR=3
MIN_PY_MINOR=9

info()  { echo "[install] $*"; }
error() { echo "[install] ERROR: $*" >&2; }

# --- Parse flags ----------------------------------------------------------
INSTALL_RECOGNITION=1
FORCE_CPU_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --no-recognition)
            INSTALL_RECOGNITION=0
            ;;
        --cpu)
            FORCE_CPU_ONLY=1
            ;;
        -h|--help)
            echo "Usage: ./install.sh [--no-recognition] [--cpu]"
            echo "  --no-recognition  Skip face recognition (PySide6/Pillow only)."
            echo "  --cpu             Force the CPU-only face-recognition build even if an"
            echo "                    NVIDIA GPU is detected (smaller download, no CUDA)."
            exit 0
            ;;
        *)
            error "Unknown option: $arg (see --help)"
            exit 1
            ;;
    esac
done

# --- Find a suitable python interpreter ---------------------------------
PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    error "No python3 interpreter found. Please install Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ and re-run this script."
    exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[0])')"
PY_MINOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])')"

if [ "$PY_MAJOR" -lt "$MIN_PY_MAJOR" ] || { [ "$PY_MAJOR" -eq "$MIN_PY_MAJOR" ] && [ "$PY_MINOR" -lt "$MIN_PY_MINOR" ]; }; then
    error "Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ is required, found ${PY_VERSION} (${PYTHON_BIN})."
    exit 1
fi
info "Found ${PYTHON_BIN} (Python ${PY_VERSION})"

# --- Check the venv module is available ---------------------------------
if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    error "The 'venv' module is not available for ${PYTHON_BIN}."
    if command -v apt-get >/dev/null 2>&1; then
        error "On Debian/Ubuntu, install it with: sudo apt-get install python${PY_MAJOR}.${PY_MINOR}-venv"
    fi
    exit 1
fi

# --- Create the virtual environment -------------------------------------
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at ${VENV_DIR}, reusing it."
else
    info "Creating virtual environment at ${VENV_DIR}..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# --- Install dependencies -------------------------------------------------
info "Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip >/dev/null

REQUIREMENTS_FILE="$REPO_DIR/requirements.txt"
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    error "requirements.txt not found at ${REQUIREMENTS_FILE}."
    exit 1
fi

info "Installing dependencies from requirements.txt..."
"$VENV_PIP" install -r "$REQUIREMENTS_FILE"

# --- Sanity check: PySide6 needs a usable Qt platform plugin -------------
if ! "$VENV_PYTHON" -c "import PySide6" >/dev/null 2>&1; then
    error "PySide6 failed to import after installation. Check the pip output above."
    exit 1
fi

# --- Face recognition (on by default) -------------------------------------
if [ "$INSTALL_RECOGNITION" -eq 1 ]; then
    RECOGNITION_FILE="$REPO_DIR/requirements-recognition.txt"
    if [ ! -f "$RECOGNITION_FILE" ]; then
        error "requirements-recognition.txt not found at ${RECOGNITION_FILE}."
        exit 1
    fi

    USE_CPU_ONLY=0
    if [ "$FORCE_CPU_ONLY" -eq 1 ]; then
        USE_CPU_ONLY=1
    elif ! command -v nvidia-smi >/dev/null 2>&1; then
        USE_CPU_ONLY=1
    fi

    if [ "$USE_CPU_ONLY" -eq 1 ]; then
        info "Installing face recognition (CPU-only build -- ~1.7GB total with its own"
        info "dependencies, vs ~5.5GB for the GPU build; detection/recognition will run"
        info "on the CPU, slower but fully functional)..."
        "$VENV_PIP" install --extra-index-url https://download.pytorch.org/whl/cpu -r "$RECOGNITION_FILE"
    else
        info "NVIDIA GPU detected -- installing face recognition with GPU (CUDA) support."
        info "WARNING: this build is much larger than the alternative -- roughly 5.5GB"
        info "total with its own dependencies (it bundles NVIDIA's CUDA runtime"
        info "libraries), vs about 1.7GB for the CPU-only build. Re-run with --cpu to"
        info "force the lighter CPU-only build instead if you'd rather not download that."
        "$VENV_PIP" install -r "$RECOGNITION_FILE"
    fi

    if ! "$VENV_PYTHON" -c "import torch" >/dev/null 2>&1; then
        error "torch failed to import after installation. Check the pip output above."
        exit 1
    fi
else
    info "Skipping face recognition (--no-recognition). Install it later with:"
    info "  pip install -r requirements-recognition.txt"
fi

info "Installation complete."
info "Activate the virtual environment with:"
info "  source ${VENV_DIR}/bin/activate"
info "Then run picSel with:"
info "  python main.py [folder]"
