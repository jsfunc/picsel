# Changelog

All notable changes to Tamis are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); dates are when the
work landed, not necessarily when a version was tagged.

## [Unreleased]

## [2.1.0]

A large batch of work: the entire face-recognition feature (already usable,
behind an optional install) plus a wide-ranging correctness/architecture
pass, and a project rename.

### Added

- **Face recognition** (installed by default — see the README's
  Installation section for the `--cpu`/`--no-recognition` override flags):
  detect faces in a photo and suggest who they are, entirely offline (no
  cloud APIs). Suggestions are ranked by confidence and color-coded rather
  than a hard yes/no cutoff; confirming one strengthens future suggestions
  for that person. Includes manual add/remove of face boxes, a Manage
  People dialog (rename, merge duplicates, forget), gallery import/export,
  and a progressive Search by Name tab. See
  [docs/face_recognition.html](docs/face_recognition.html) for how the
  detection/embedding/identity-matching pipeline actually works.
- `install.sh` now installs face recognition by default, auto-detecting an
  NVIDIA GPU and picking the matching PyTorch build (CUDA-enabled or
  CPU-only) automatically.
- Packaged executables now come in three variants per OS: no recognition,
  CPU-only recognition, and GPU (CUDA) recognition — Linux and Windows get
  all three, macOS gets the first two (no CUDA support on Apple hardware).
- `--version`/`-v` CLI flag, a `Help > About Tamis` dialog, and a
  versioned window title — previously there was no way to tell which
  release a running copy actually was.
- Logging to `~/.tamis/tamis.log` plus a global exception handler, so an
  uncaught error leaves a diagnostic trail instead of vanishing silently
  (particularly useful for the packaged executable, which has no terminal).
- A CI workflow (`.github/workflows/tests.yml`) that runs the full test
  suite, including the optional recognition-dependent tests, on every push
  and PR to `main` — previously the only workflow was the tag-triggered
  release build, which never ran `pytest` at all.
- A pick/reject badge on filmstrip thumbnails, in addition to the existing
  background tint, so status reads without relying on color alone.
- A size-budget check in the release workflow, failing the build if a
  future dependency change accidentally bloats a packaged executable
  beyond what's expected for its recognition variant.

### Changed

- **Renamed from picSel to Tamis** — the old name collided with an
  unrelated commercial product in the same space (photo culling/editing
  with facial recognition). Package/import path, on-disk sidecar file names
  (`.tamis_state.json`, `.tamis_faces.json`, `~/.tamis/people.json.gz`),
  and the GitHub repository all changed accordingly; existing data migrates
  automatically from the old names with no user action needed.
- `main_window.py` split from a single ~1,700-line file into
  `tamis/controllers/` (`EditController`, `FaceRecognitionController`) plus
  several `tamis/views/` dialog and panel modules, for testability and to
  stop one file from doing six unrelated jobs at once.
- State files now write atomically (temp file + rename via
  `tamis/persistence.py`) instead of directly, so a crash or full disk
  mid-write can't corrupt them.
- Thumbnail generation now uses JPEG draft-mode decoding where safe
  (roughly halving folder-open time for large folders), and face-catalog/
  person-gallery saves moved off the UI thread (previously up to ~1s of
  felt lag on every face-name confirmation, since each save rewrote its
  entire file synchronously).
- Metadata (EXIF/GPS panel) now loads asynchronously instead of blocking
  the UI thread on every photo navigation.
- Dependency versions in `requirements*.txt` now have upper bounds, to
  stop a fresh install from silently pulling an untested next-major-version
  release.

### Fixed

- A cropped or rotated photo's saved EXIF no longer carries a stale
  embedded thumbnail depicting the pre-edit framing.
- Real camera metadata (capture date, GPS, exposure/lens info) is no
  longer silently stripped from saved photos on Pillow versions below
  11.1 — see `requirements.txt`'s comment on the `Pillow` lower bound for
  the full story.
- Several reload-ordering and race-condition bugs: stale ratings/face data
  surviving an Apply Culling move, a corrupted per-folder or central
  face-recognition data file silently resetting (and risking overwrite) on
  the next save instead of warning, cross-folder face-cache contamination
  on a fast folder switch, one corrupted photo aborting an entire folder
  search instead of being skipped, and a Manage People merge/forget
  silently truncating an in-flight search for the person it affected.
- Crop mode and Face Recognition "Edit Faces" mode are now mutually
  exclusive (both interpreted mouse drags on the shared image viewer).
- The image viewer no longer discards a manual zoom on every viewport
  resize (e.g. nudging the side-panel splitter), only on an actual
  double-click-to-fit.

## [2.0.1]

- Released cached GPU memory after each face-detection/embedding call.
- Documented the face-recognition feature in the README.

## [2.0.0]

- Added face detection and recognition as a new, optional feature.

## [1.0.0]

- Initial release: browse and rate a folder of photos (filmstrip, EXIF/GPS
  panel, select/reject/star-rating with a `.tamis_state.json` sidecar),
  rotate/flip/crop/adjust with undo-redo, Apply Culling to move or copy
  into `selected/`/`rejected/` subfolders, and three renaming modes
  (sequence, renumber-by-creation-order, rename-by-capture-date). JPEG,
  PNG, BMP, TIFF, WebP, and HEIC/HEIF.
