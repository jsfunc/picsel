#!/usr/bin/env python3
"""Entry point for picSel.

Usage:
    python main.py [folder]
"""

import sys

from picsel.app import run

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run(folder))
