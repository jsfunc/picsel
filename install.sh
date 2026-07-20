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

info "Installation complete."
info "Activate the virtual environment with:"
info "  source ${VENV_DIR}/bin/activate"
info "Then run picSel with:"
info "  python main.py [folder]"
