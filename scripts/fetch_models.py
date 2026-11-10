#!/usr/bin/env python
"""Re-create the bundled robot models in cobot_sandbox/assets/third_party/menagerie/.

The models are committed to this repo (an unmodified subset of MuJoCo Menagerie, see
THIRD_PARTY_NOTICES.md), so you normally never need this. Use it to rebuild or update them:

    python scripts/fetch_models.py            # download the pinned version, keep only the used files
    python scripts/fetch_models.py --prune    # just drop unused files from the existing folder

Only files the robots in cobot_sandbox/assets/robot/*.yaml load are kept, plus each model folder's
LICENSE, README.md and CHANGELOG.md. Needs git for downloading.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cobot_sandbox.paths import MENAGERIE_DIR  # noqa: E402

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"      # pinned so the models never change underneath us
FOLDERS = ["franka_fr3", "franka_emika_panda", "universal_robots_ur5e", "robotiq_2f85"]
KEEP_ALWAYS = {"LICENSE", "README.md", "CHANGELOG.md"}

NOTE = f"""# Third-party robot models (MuJoCo Menagerie subset)

Source: {REPO}
Commit: {COMMIT}

These folders are an **unmodified subset** of MuJoCo Menagerie: only the model and mesh files this
sandbox loads are included; other files (other variants, previews, demo scenes) were omitted.
Each folder keeps its own `LICENSE` (the license that applies to that folder) and `README.md`
(origin and credits). This sandbox's MIT license does not apply to these files.

| Folder | Used for | License |
|---|---|---|
| `franka_fr3/` | Franka Research 3 arm | Apache-2.0 |
| `franka_emika_panda/` | Franka Hand (`hand.xml`) used on the FR3 | Apache-2.0 |
| `universal_robots_ur5e/` | UR5e arm | BSD-3-Clause |
| `robotiq_2f85/` | Robotiq 2F-85 gripper used on the UR5e | BSD-2-Clause |

The models are adapted in memory at load time by `cobot_sandbox/robots.py` (gripper attached, position
servos replaced by torque motors, sites/camera/sensors added); the files themselves are unchanged.
Re-create this folder with `python scripts/fetch_models.py`.
"""


def used_files() -> set:
    """Absolute paths of every model / mesh file the configured robots load."""
    from cobot_sandbox import robots as R
    used = set()
    for name in R.robot_names():
        cfg = R.load_cfg(name)
        if cfg.get("source") != "menagerie":
            continue
        used.add((MENAGERIE_DIR / cfg["arm"]).resolve())
        used.add((MENAGERIE_DIR / cfg["gripper_model"]["file"]).resolve())
        spec = R.build(name)
        for item in list(spec.meshes) + list(spec.textures):
            if item.file:
                used.add(Path(item.file).resolve())
    return used


def prune():
    keep = used_files()
    removed = 0
    for f in sorted(MENAGERIE_DIR.rglob("*"), reverse=True):
        if f.is_file() and f.resolve() not in keep and f.name not in KEEP_ALWAYS and f.parent != MENAGERIE_DIR:
            f.unlink()
            removed += 1
        elif f.is_dir() and not any(f.iterdir()):
            f.rmdir()
    (MENAGERIE_DIR / "README.md").write_text(NOTE)
    stale = MENAGERIE_DIR / "SOURCE.txt"
    if stale.exists():
        stale.unlink()
    size = sum(f.stat().st_size for f in MENAGERIE_DIR.rglob("*") if f.is_file())
    print(f"kept {len(keep)} model files ({size / 1e6:.1f} MB incl. licenses), removed {removed} unused files")


def download():
    if shutil.which("git") is None:
        sys.exit("git is required (macOS: xcode-select --install)")
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "menagerie"
        run = lambda *a: subprocess.run(a, check=True, cwd=clone if clone.exists() else tmp)  # noqa: E731
        print(f"downloading {', '.join(FOLDERS)} from MuJoCo Menagerie @ {COMMIT[:8]} ...")
        run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO, str(clone))
        run("git", "sparse-checkout", "set", *FOLDERS)
        run("git", "checkout", "-q", COMMIT)
        MENAGERIE_DIR.mkdir(parents=True, exist_ok=True)
        for f in FOLDERS:
            shutil.rmtree(MENAGERIE_DIR / f, ignore_errors=True)
            shutil.copytree(clone / f, MENAGERIE_DIR / f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prune", action="store_true", help="don't download; only drop unused files")
    a = ap.parse_args()
    if not a.prune:
        download()
    prune()
    print(f"done: {MENAGERIE_DIR}")


if __name__ == "__main__":
    main()
