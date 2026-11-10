"""Robot definitions: one YAML per robot in assets/robot/<name>.yaml, built into an MjSpec that a
scene attaches at a mount pose (pedestal / table top). Every robot ends up with the same interface
the controller in sim.py needs:

  * arm joints driven by **motor** actuators (`<joint>_motor` for Menagerie robots, whose built-in
    position servos are removed): sim.py runs PD + gravity compensation and sends torques, like a
    real cobot's joint controller;
  * a TCP site (z axis = approach direction, y axis = finger closing direction);
  * a gripper actuator with `open` / `closed` control values;
  * a wrist camera `wrist_cam` and a wrist force/torque sensor pair `wrist_force` / `wrist_torque`.

`source: menagerie` builds from the bundled MuJoCo Menagerie files (assets/third_party/menagerie), attaching
the gripper to the arm's flange site. `source: xml` loads a model file from assets/robot (for a
custom robot of your own; give it the same names the YAML refers to).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import mujoco
import numpy as np

from .paths import MENAGERIE_DIR, ROBOT_DIR



class RobotModelsMissing(RuntimeError):
    """None of a scene's robots is available (its bundled model files are missing)."""


def robot_names() -> List[str]:
    return sorted(p.stem for p in ROBOT_DIR.glob("*.yaml"))


@lru_cache(maxsize=None)
def _cfg_cached(name: str) -> dict:
    from .scene_builder import read_yaml
    path = ROBOT_DIR / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"unknown robot '{name}' (available: {', '.join(robot_names())})")
    cfg = read_yaml(path)
    cfg.setdefault("name", name)
    cfg.setdefault("title", name)
    return cfg


def load_cfg(name: str) -> dict:
    import copy
    return copy.deepcopy(_cfg_cached(name))


def available(name: str) -> bool:
    try:
        cfg = _cfg_cached(name)
    except KeyError:
        return False
    if cfg.get("source") != "menagerie":
        return True
    return (MENAGERIE_DIR / cfg["arm"]).exists() and (MENAGERIE_DIR / cfg["gripper_model"]["file"]).exists()


def _load_abs(path: Path) -> mujoco.MjSpec:
    """Load a model and make its mesh/texture paths absolute, so it can be attached anywhere."""
    s = mujoco.MjSpec.from_file(str(path))
    for attr, items in (("meshdir", s.meshes), ("texturedir", s.textures)):
        d = path.parent / (getattr(s, attr) or "")
        for it in items:
            if it.file and not Path(it.file).is_absolute():
                it.file = str((d / it.file).resolve())
        setattr(s, attr, "")
    return s


def match_options(parent: mujoco.MjSpec, child: mujoco.MjSpec):
    """Copy physics options onto a child before attaching (avoids MuJoCo 'attach conflict' warnings;
    the scene's options are the ones that apply)."""
    for k in ("timestep", "impratio", "cone", "integrator", "jacobian", "solver", "iterations", "tolerance"):
        try:
            setattr(child.option, k, getattr(parent.option, k))
        except (AttributeError, TypeError):
            pass


def build(name: str) -> mujoco.MjSpec:
    """MjSpec of robot `name`, root body at the origin = the mount point."""
    cfg = _cfg_cached(name)
    if cfg.get("source") == "menagerie":
        s = _load_abs(MENAGERIE_DIR / cfg["arm"])
        g = cfg["gripper_model"]
        grip = _load_abs(MENAGERIE_DIR / g["file"])
        match_options(s, grip)
        s.attach(grip, site=s.site(g.get("attach_site", "attachment_site")), prefix=g.get("prefix", "grip_"))
        arm = set(cfg["arm_joints"])
        for a in list(s.actuators):          # replace position servos on the arm with torque motors
            if a.target in arm:
                s.delete(a)
        lim = cfg["torque_limits"]
        for j, tl in zip(cfg["arm_joints"], lim):
            a = s.add_actuator()
            a.name, a.target, a.trntype = f"{j}_motor", j, mujoco.mjtTrn.mjTRN_JOINT
            a.ctrllimited, a.ctrlrange = 1, [-float(tl), float(tl)]
    else:
        s = _load_abs(ROBOT_DIR / cfg["xml"])
    for k in list(s.keys):                   # keyframes would not match the scene's qpos layout
        s.delete(k)
    # Softer finger contacts: a light object pinched between a stiff finger and the table can
    # otherwise blow up (see CLAUDE.md gotchas).
    finger = {n for side in cfg.get("finger_bodies", []) for n in side}
    solref = cfg.get("finger_solref", [0.01, 1.0])
    for b in s.bodies:
        if b.name in finger:
            for g in b.geoms:
                if g.contype or g.conaffinity:
                    g.solref = list(solref)

    tcp = cfg["tcp"]
    if "body" in tcp:
        b = s.body(tcp["body"])
        site = b.add_site()
        site.name, site.pos = "tcp", list(tcp.get("pos", [0, 0, 0]))
        if "quat" in tcp:
            site.quat = list(tcp["quat"])
        site.size, site.group, site.rgba = [0.006, 0, 0], 2, [1, 0.3, 0.1, 0.6]
    cam = cfg.get("wrist_camera") or {}
    if "body" in cam:
        c = s.body(cam["body"]).add_camera()
        c.name, c.pos, c.fovy = "wrist_cam", list(cam["pos"]), float(cam.get("fovy", 70))
        c.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        c.alt.xyaxes = list(cam["xyaxes"])
    ft = cfg.get("ft_site")
    if ft and not any(x.name == "wrist_force" for x in s.sensors):
        for kind, nm in ((mujoco.mjtSensor.mjSENS_FORCE, "wrist_force"), (mujoco.mjtSensor.mjSENS_TORQUE, "wrist_torque")):
            se = s.add_sensor()
            se.name, se.type, se.objtype, se.objname = nm, kind, mujoco.mjtObj.mjOBJ_SITE, ft
    return s


def attach(scene: mujoco.MjSpec, name: str, pos, yaw: float, prefix: str):
    child = build(name)
    match_options(scene, child)
    q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    frame = scene.worldbody.add_frame(pos=[float(v) for v in pos], quat=q)
    scene.attach(child, frame=frame, prefix=prefix)


def resolve(mounts: List[dict], select: str, choice: Optional[str] = None) -> List[dict]:
    """Scene mounts -> the robots to simulate. select "all": every listed robot that is available;
    "one": the chosen (or first available) one. Raises RobotModelsMissing if none is available."""
    mounts = [dict(m, index=i) for i, m in enumerate(mounts)]
    ok = [m for m in mounts if available(m["robot"])]
    if select == "all":
        out = ok
    else:
        pick = [m for m in ok if m["robot"] == choice] if choice else []
        out = (pick or ok)[:1]
    if not out and mounts:
        names = ", ".join(dict.fromkeys(m["robot"] for m in mounts))
        raise RobotModelsMissing(f"robot models not found ({names}). They ship with the repo in "
                                 f"cobot_sandbox/assets/third_party/menagerie/; if that folder is missing or "
                                 f"incomplete, reinstall the package or, in a source checkout, re-create it with:\n"
                                 f"    python scripts/fetch_models.py   (needs git)")
    return [dict(m, prefix=f"r{i}_") for i, m in enumerate(out)]
