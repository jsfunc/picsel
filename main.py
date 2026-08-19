#!/usr/bin/env python3
"""Entry point for picSel.

Usage:
    python main.py [folder]
    python main.py --version
"""

import sys

from picsel import __version__
from picsel.app import run

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-v"):
        print(f"picSel {__version__}")
        sys.exit(0)
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run(folder))
