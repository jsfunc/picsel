# picSel

picSel is a desktop app for quickly culling and lightly editing a folder of
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
installs the dependencies (`PySide6`, `Pillow`, `pillow-heif`, `pytest`).

## Usage

```bash
source .venv/bin/activate
python main.py [folder]
```

Pass a folder to open it immediately, or use File > Open Folder from the app.
See Help > Keyboard Shortcuts inside the app for the full shortcut list
(navigation, rating, editing, renaming).

## Running tests

```bash
source .venv/bin/activate
python -m pytest
```
