"""Per-user app settings: optional `settings.yaml` in paths.USER_DIR (the repo folder in a checkout,
~/cobot-sandbox/ for a pip install; gitignored).

Only keys you want to change need to be in the file; everything else uses DEFAULTS. Command-line
flags override the file for one run. Example settings.yaml:

    show_labels: false      # floating text labels in the 3D view (also: G in the app, --no-labels)
    robot: ur5e             # robot preselected in the picker of single-robot scenes (also: --robot)

The app writes `robot` back when you pick one; other keys are only read.
"""
from __future__ import annotations

from typing import Any, Dict

from .paths import SETTINGS_FILE

DEFAULTS: Dict[str, Any] = {
    "show_labels": True,
    "robot": None,          # None: the scene's first available robot
}


def load_settings() -> Dict[str, Any]:
    out = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        from .scene_builder import read_yaml
        data = read_yaml(SETTINGS_FILE) or {}
        unknown = set(data) - set(DEFAULTS)
        if unknown:
            print(f"settings.yaml: ignoring unknown keys {sorted(unknown)} (known: {sorted(DEFAULTS)})")
        out.update({k: v for k, v in data.items() if k in DEFAULTS})
    return out


def save_setting(key: str, value) -> None:
    """Set one key in settings.yaml, keeping the other keys (and creating the file if needed)."""
    import yaml
    from .scene_builder import read_yaml
    data = (read_yaml(SETTINGS_FILE) or {}) if SETTINGS_FILE.exists() else {}
    if data.get(key) == value:
        return
    data[key] = value
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text("# Per-user settings for the Cobot Sandbox app (see cobot_sandbox/settings.py).\n"
                             + yaml.safe_dump(data, sort_keys=False))
