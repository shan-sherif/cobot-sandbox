#!/usr/bin/env python
"""Cobot Sandbox launcher for a source checkout (same as the `cobot-sandbox` command after
`pip install`). Run `python run.py --help` for options; see cobot_sandbox/cli.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cobot_sandbox.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
