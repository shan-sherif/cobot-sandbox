#!/usr/bin/env python
"""Source-checkout wrapper for cobot_sandbox/tools/collect.py (the `cobot-sandbox-collect` command after
`pip install`). Run with --help for options."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cobot_sandbox.tools.collect import main  # noqa: E402

if __name__ == "__main__":
    main()
