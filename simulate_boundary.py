#!/usr/bin/env python3
"""Compatibility wrapper for the packaged boundary simulator."""

from compression_monitor.simulate_boundary import *  # noqa: F401,F403
from compression_monitor.simulate_boundary import main


if __name__ == "__main__":
    main()
