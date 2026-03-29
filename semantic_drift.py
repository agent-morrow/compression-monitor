#!/usr/bin/env python3
"""Compatibility wrapper for the packaged semantic drift tool."""

from compression_monitor.semantic_drift import *  # noqa: F401,F403
from compression_monitor.semantic_drift import main


if __name__ == "__main__":
    main()
