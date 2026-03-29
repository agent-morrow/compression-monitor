#!/usr/bin/env python3
"""Compatibility wrapper for the packaged behavioral footprint tool."""

from compression_monitor.behavioral_footprint import *  # noqa: F401,F403
from compression_monitor.behavioral_footprint import main


if __name__ == "__main__":
    main()
