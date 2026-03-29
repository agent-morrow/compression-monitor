#!/usr/bin/env python3
"""Compatibility wrapper for the packaged ghost lexicon tool."""

from compression_monitor.ghost_lexicon import *  # noqa: F401,F403
from compression_monitor.ghost_lexicon import main


if __name__ == "__main__":
    main()
