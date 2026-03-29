#!/usr/bin/env python3
"""Compatibility wrapper for the packaged preregistration CLI."""

from compression_monitor.preregister import *  # noqa: F401,F403
from compression_monitor.preregister import main


if __name__ == "__main__":
    main()
