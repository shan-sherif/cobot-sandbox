"""Episode recording for imitation learning.

One episode = one compressed .npz:
    obs/<key>    (T, ...)  every observation array from Sim.get_obs()
    action       (T, 8)    the Command actually applied (see command.ACTION_LABELS)
    action_label (T, 8)    optional clean expert command when noise was injected (DART)
    task/<key>   (T, ...)  optional goal info from scripted policies (target / place positions)
    init/<key>             full simulator state at the first step (Sim.get_state)
    img/<camera> (T, H, W, 3) uint8, only when cameras are requested
    meta                   JSON: scene, seed, randomize, control rate, robot, success, ...
Replaying `action` after Sim(scene, seed, randomize, robot) + set_state(init) reproduces the episode.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .command import ACTION_LABELS
from .paths import EPISODE_DIR


class Recorder:
    def __init__(self, out_dir: Optional[Path] = None, cameras: Sequence[str] = (), image_size=(128, 128)):
        self.out_dir = Path(out_dir or EPISODE_DIR)
        self.cameras, self.image_size = list(cameras), tuple(image_size)
        self.active = False
        self._clear()

    def _clear(self):
        self.obs: Dict[str, List[np.ndarray]] = {}
        self.task: Dict[str, List[np.ndarray]] = {}
        self.actions: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.images: Dict[str, List[np.ndarray]] = {c: [] for c in self.cameras}
        self.meta: dict = {}
        self.init: Dict[str, np.ndarray] = {}

    @property
    def length(self) -> int:
        return len(self.actions)

    def start(self, sim, source: str, **extra):
        self._clear()
        self.active = True
        # seed = this episode's reset seed; scene_seed = the seed the Sim was built with (it fixes
        # procedural clutter: object types, sizes, masses). Replay needs both.
        self.meta = dict(scene=sim.spec.name, scene_raw=sim.spec.raw, seed=sim.last_seed, scene_seed=sim.seed,
                         randomize=sim.last_randomize, control_dt=sim.control_dt, physics_dt=sim.dt,
                         obs_noise=sim.obs_noise, action_latency=sim.action_latency,
                         robot=sim.robot_name, robot_title=sim.arm.title, arm_dof=len(sim.arm.arm_q),
                         source=source, action_labels=ACTION_LABELS,
                         objects=list(sim.objects), cameras=self.cameras, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                         **extra)
        st = sim.get_state()
        self.init = {k: v for k, v in st.items() if isinstance(v, np.ndarray)}
        self.meta["init"] = {k: v for k, v in st.items() if not isinstance(v, np.ndarray)}

    def add(self, obs: dict, action: np.ndarray, task: Optional[dict] = None, sim=None, label=None):
        """Call with the observation the action was chosen from, then step the sim."""
        if not self.active:
            return
        for k, v in obs.items():
            self.obs.setdefault(k, []).append(np.asarray(v, np.float32))
        for k, v in (task or {}).items():
            self.task.setdefault(k, []).append(np.asarray(v, np.float32))
        self.actions.append(np.asarray(action, np.float32))
        if label is not None:
            self.labels.append(np.asarray(label, np.float32))
        if self.cameras and sim is not None:
            w, h = self.image_size
            for c in self.cameras:
                self.images[c].append(sim.render(c, w, h))

    def stop(self, save: bool = True, success: Optional[bool] = None, name: Optional[str] = None) -> Optional[Path]:
        self.active = False
        if not save or not self.actions:
            self._clear()
            return None
        self.meta.update(success=success, length=len(self.actions))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        path = self.out_dir / f"{name or self.meta['scene']}_{stamp}.npz"
        arrays = {f"obs/{k}": np.stack(v) for k, v in self.obs.items()}
        arrays.update({f"task/{k}": np.stack(v) for k, v in self.task.items() if len(v) == len(self.actions)})
        arrays.update({f"img/{c}": np.stack(v) for c, v in self.images.items() if v})
        arrays.update({f"init/{k}": v for k, v in self.init.items()})
        arrays["action"] = np.stack(self.actions)
        if len(self.labels) == len(self.actions):
            arrays["action_label"] = np.stack(self.labels)
        arrays["meta"] = np.array(json.dumps(self.meta, default=_json_default))
        np.savez_compressed(path, **arrays)
        self._clear()
        return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def load_episode(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        ep = {"obs": {}, "task": {}, "img": {}, "init": {}, "action": z["action"], "meta": json.loads(str(z["meta"]))}
        for k in z.files:
            for group in ("obs", "task", "img", "init"):
                if k.startswith(group + "/"):
                    ep[group][k[len(group) + 1:]] = z[k]
        if "action_label" in z.files:
            ep["action_label"] = z["action_label"]
    ep["init"].update(ep["meta"].get("init", {}))
    return ep


def sim_for_episode(ep: dict):
    """Rebuild the simulator in exactly the state an episode started from."""
    from .sim import Sim
    m = ep["meta"]
    scene_seed = m.get("scene_seed", m["seed"])     # older episodes: assume the Sim was built with `seed`
    sim = Sim(m.get("scene_raw") or m["scene"], seed=scene_seed, randomize=m["randomize"], robot=m.get("robot"),
              obs_noise=m.get("obs_noise", 0.0), action_latency=m.get("action_latency", 0))
    if scene_seed != m["seed"]:                     # re-apply the episode's own randomization draw
        sim.reset(seed=m["seed"], randomize=m["randomize"])
    if ep["init"]:
        sim.set_state(ep["init"])
    return sim


def list_episodes(folder: Optional[Path] = None, success_only: bool = False) -> List[Path]:
    paths = sorted(Path(folder or EPISODE_DIR).rglob("*.npz"))
    if success_only:
        paths = [p for p in paths if load_episode(p)["meta"].get("success")]
    return paths
