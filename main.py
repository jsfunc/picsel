#!/usr/bin/env python3
"""Entry point for Tamis.

Usage:
    python main.py [folder]
    python main.py --version
"""

import sys

from tamis import __version__
from tamis.app import run
from tamis.features import summary

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-v"):
        # Also reports which optional extras this build has. Both features
        # hide themselves when their dependencies are missing, so without
        # this there is no way to tell a correctly-lean build from a packaged
        # one that was meant to include them -- which is how two releases
        # shipped without quality scoring and looked fine.
        print(f"Tamis {__version__}")
        for line in summary():
            print(line)
        sys.exit(0)
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run(folder))
