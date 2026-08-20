"""Shared color constants for the views layer.

Previously each view file defined its own ad hoc QColor literals with no
coordination -- e.g. ImageViewer's crop-selection color and its detected/
confirmed face-box color turned out to be the exact same yellow, purely by
coincidence of each being written independently, not a deliberate choice.
Named here so that kind of overlap is visible and intentional instead of
hidden. No dark-mode awareness -- just names, no behavior change.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

# -- ImageViewer: crop selection + face-box overlay -----------------------
VIEWER_BACKGROUND = QColor("#202020")
CROP_SELECTION_COLOR = QColor("yellow")
FACE_BOX_COLOR = QColor("yellow")  # detected/confirmed face boxes -- same yellow as crop selection, coincidentally
FACE_BOX_FILL = QColor(255, 220, 0, 40)
FACE_DRAW_IN_PROGRESS_COLOR = QColor("cyan")  # a face box while still being dragged out

# -- ThumbnailList: pick/reject background tint + status badge ------------
SELECTED_TINT = QColor(46, 125, 50, 120)
REJECTED_TINT = QColor(125, 46, 46, 120)
NEUTRAL_TINT = QColor(0, 0, 0, 0)
SELECTED_BADGE_COLOR = QColor(46, 140, 60)
REJECTED_BADGE_COLOR = QColor(170, 45, 45)
BADGE_TEXT_COLOR = QColor("white")

# -- FacePanel: confidence/similarity gradient endpoints -------------------
LOW_SIMILARITY_COLOR = QColor(200, 70, 70)
HIGH_SIMILARITY_COLOR = QColor(60, 160, 90)
