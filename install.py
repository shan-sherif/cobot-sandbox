#!/usr/bin/env python3
"""Cross-platform one-step setup (macOS, Linux, Windows). Run it with the Python you want to use:

    python3 install.py          macOS / Linux   (or ./setup.sh, which calls this)
    py -3.12 install.py         Windows
    python3 install.py --dev    + pytest (to run the tests)
    python3 install.py --check  only check an existing .venv (no installs)

Creates .venv next to this file, installs this repo into it as an editable package (so the
`cobot-sandbox` commands work and code changes apply immediately; --dev adds the [dev] extra),
pre-generates textures and runs a quick headless check. Uses only the standard library.
(Just want to use it, not edit it? `pip install cobot-sandbox` works too, see README.)
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MIN_PY, TESTED_PY = (3, 10), (3, 12)
WINDOWS = os.name == "nt"


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")


def _short(a) -> str:
    s = str(a)
    return os.path.relpath(s, ROOT) if os.path.isabs(s) and s.startswith(str(ROOT)) else s


def run(*args, quiet=False):
    print("  $", " ".join(_short(a) for a in args))
    subprocess.run([str(a) for a in args], check=True, cwd=ROOT,
                   stdout=subprocess.DEVNULL if quiet else None)


def check(py: Path) -> bool:
    """Import everything and build + step a scene without opening a window or rendering."""
    code = ("import sys; sys.path.insert(0, '.'); "
            "from cobot_sandbox.sim import Sim; s = Sim('lift'); [s.step() for _ in range(10)]; "
            "print('  sandbox OK:', s.arm.title, '| mujoco', __import__('mujoco').__version__)")
    env = dict(os.environ, SDL_VIDEODRIVER="dummy")
    return subprocess.run([str(py), "-c", code], cwd=ROOT, env=env).returncode == 0


def next_steps():
    act = r".venv\Scripts\activate" if WINDOWS else "source .venv/bin/activate"
    print("\nReady. Start the sandbox with:")
    print(f"  {act}")
    print("  cobot-sandbox          (or: python run.py)")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("\nNo display detected (headless Linux). The app window needs a desktop; for headless")
        print("data collection with camera images, set MUJOCO_GL=egl (GPU) or MUJOCO_GL=osmesa")
        print("(CPU, needs: sudo apt install libosmesa6). See README > Headless Linux.")


def main():
    sys.stdout.reconfigure(line_buffering=True)        # keep our lines in order with child output
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true", help="also install pytest")
    ap.add_argument("--check", action="store_true", help="only verify an existing .venv")
    a = ap.parse_args()

    v = sys.version_info[:2]
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if v < MIN_PY:
        sys.exit(f"Python {MIN_PY[0]}.{MIN_PY[1]}+ is required (tested with {TESTED_PY[0]}.{TESTED_PY[1]}). "
                 "macOS: brew install python@3.12 | Windows: python.org or winget install Python.Python.3.12 "
                 "| Ubuntu: sudo apt install python3.12-venv")
    if v != TESTED_PY:
        print(f"  note: tested with Python {TESTED_PY[0]}.{TESTED_PY[1]}; {v[0]}.{v[1]} should work")

    py = venv_python()
    if a.check:
        sys.exit(0 if py.exists() and check(py) else "check failed (run install.py without --check)")

    if not py.exists():
        print("Creating .venv ...")
        try:
            run(sys.executable, "-m", "venv", VENV)
        except subprocess.CalledProcessError:
            sys.exit("Could not create a virtual environment. On Debian/Ubuntu: sudo apt install python3-venv")
    print("Installing packages ...")
    run(py, "-m", "pip", "install", "-q", "--upgrade", "pip")
    run(py, "-m", "pip", "install", "-q", "-e", ".[dev]" if a.dev else ".")
    print("Generating textures ...")
    run(py, "-m", "cobot_sandbox.textures", quiet=True)

    models = ROOT / "cobot_sandbox/assets/third_party/menagerie"
    if not (models / "franka_fr3").is_dir() or not (models / "universal_robots_ur5e").is_dir():
        print("  warning: robot models missing from cobot_sandbox/assets/third_party/menagerie/ "
              "(re-create with: python scripts/fetch_models.py)")
    print("Checking ...")
    if not check(py):
        sys.exit("The quick check failed; see the error above.")
    next_steps()


if __name__ == "__main__":
    main()
