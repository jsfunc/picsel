# picSel

picSel is a small desktop app for quickly culling and lightly editing a folder of
photos: browse a shoot, mark shots as selected/rejected, star-rate them, crop
or rotate the keepers, and sort the results into folders — all from the
keyboard, without leaving a single window.

## Features

- **Browse & rate** — filmstrip of thumbnails, full-size preview, EXIF/GPS
  metadata panel. Mark each photo selected, rejected, or unrated, and give it
  a 0-5 star rating. Ratings and status persist alongside the folder (a
  `.picsel_state.json` sidecar) so you can close and resume later.
- **Edit** — rotate, flip, crop, and adjust brightness/contrast/saturation,
  with undo/redo. Save as a copy, overwrite the original, or save as a new
  file.
- **Apply culling** — move or copy selected/rejected photos into
  `selected/` and `rejected/` subfolders in one step.
- **Face recognition** *(installed by default, see [Installation](#installation))* —
  detect faces in a photo and suggest who they are, entirely offline (no
  cloud APIs, nothing ever leaves your machine). Suggestions are ranked by
  confidence, shown color-coded (green = likely, red = unlikely) rather than
  a hard yes/no cutoff — confirm one with a click and it strengthens future
  suggestions for that person. Includes manual add/remove of face boxes for
  anything the detector misses or gets wrong, a Manage People dialog
  (rename, merge duplicates, forget), gallery import/export to move your
  identities to another machine, and a progressive Search by Name that finds
  every photo of someone across the whole folder. See
  [docs/face_recognition.html](docs/face_recognition.html) for how the
  detection/recognition pipeline actually works.
- **Rename**
  - Rename the current photo to `<name><sequence number>.<ext>`.
  - Renumber an existing `<name><digits>.<ext>` sequence so numbering matches
    actual capture order.
  - Rename every photo in the folder to `pYYYYmmdd_hhmmss.ext` based on its
    capture date (from EXIF, falling back to file modification time).
- Supports JPEG, PNG, BMP, TIFF, WebP, and HEIC/HEIF.

## Installation

Requires Python 3.9+.

```bash
./install.sh
```

This checks your Python version, creates a `.venv` virtual environment, and
installs the dependencies (`PySide6`, `Pillow`, `pillow-heif`), **including
face recognition by default**. It auto-detects an NVIDIA GPU (`nvidia-smi`)
and installs the matching PyTorch build:

- **GPU present**: the standard (CUDA-enabled) wheels — faster detection/
  recognition, but a much larger download (~5.5GB total for the recognition
  dependencies, since it bundles NVIDIA's CUDA runtime libraries).
- **No GPU found**: the CPU-only wheels automatically (~1.7GB total) —
  detection/recognition still works, just slower.

Flags to override the default:

```bash
./install.sh --cpu             # force the CPU-only build even with a GPU present
./install.sh --no-recognition  # skip face recognition entirely (PySide6/Pillow only)
```

Without face recognition installed, the app runs normally with the Face
Recognition and Search by Name tabs simply absent. To add it later:

```bash
source .venv/bin/activate
pip install -r requirements-recognition.txt                                            # GPU-capable
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-recognition.txt  # CPU-only
```

## Usage

```bash
source .venv/bin/activate
python main.py [folder]
```

Pass a folder to open it immediately, or use File > Open Folder from the app.
See Help > Keyboard Shortcuts inside the app for the full shortcut list
(navigation, rating, editing, renaming).

## Running tests

`pytest` is a dev-only dependency (not installed by `./install.sh`, so a
plain end-user install doesn't pull in a test runner):

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

## Standalone executables

Before tagging a release, bump `__version__` in
[picsel/\_\_init\_\_.py](picsel/__init__.py) to match — it drives the window
title, `Help > About picSel`, and `python main.py --version`, but nothing
derives it from the git tag automatically.

Pushing a tag like `v1.0.0` triggers [.github/workflows/release.yml](.github/workflows/release.yml),
which builds a standalone executable for Linux, Windows, and macOS (Apple
Silicon) with [PyInstaller](https://pyinstaller.org/) and attaches them to a
GitHub Release. You can also run the workflow manually from the Actions tab.

These are unsigned builds, so Windows SmartScreen and macOS Gatekeeper will
warn on first run; you'll need to explicitly allow the app to run.

These prebuilt executables do **not** include face recognition — PyTorch is
too large to bundle into a portable single-file build. Run from source with
`requirements-recognition.txt` installed (see
[Installation](#face-recognition-optional)) to use that feature.

To build one locally instead:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pyinstaller picsel.spec
```

The executable is written to `dist/picSel` (`dist/picSel.exe` on Windows).
PyInstaller doesn't cross-compile, so this must be run on each target OS.

## License

GPLv3 — see [LICENSE](LICENSE). Copyright (C) 2026 jsfunc.
