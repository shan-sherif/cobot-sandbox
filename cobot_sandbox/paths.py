"""Where things live.

Package files (robots, scenes, textures) sit inside the installed package. User files (recordings,
settings.yaml, exported scenes) go to USER_DIR:
  * a source checkout (git clone):  the repository folder, as before (data/, settings.yaml)
  * a pip install:                  ~/cobot-sandbox/
  * either, if set:                 $COBOT_SANDBOX_HOME
"""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
SOURCE_CHECKOUT = (PROJECT_DIR / "pyproject.toml").exists() and (PROJECT_DIR / "run.py").exists()

ASSET_DIR = PKG_DIR / "assets"
ROBOT_DIR = ASSET_DIR / "robot"
TEXTURE_DIR = ASSET_DIR / "textures"
SCENE_DIR = PKG_DIR / "scenes"
MENAGERIE_DIR = ASSET_DIR / "third_party" / "menagerie"   # bundled Menagerie subset (THIRD_PARTY_NOTICES.md)

_home = os.environ.get("COBOT_SANDBOX_HOME")
USER_DIR = Path(_home).expanduser() if _home else (PROJECT_DIR if SOURCE_CHECKOUT else Path.home() / "cobot-sandbox")
DATA_DIR = USER_DIR / "data"
EPISODE_DIR = DATA_DIR / "episodes"
SETTINGS_FILE = USER_DIR / "settings.yaml"
BUILD_DIR = USER_DIR / "build"
