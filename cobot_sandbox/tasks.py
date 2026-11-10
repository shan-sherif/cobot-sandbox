"""Task success checks for scene `tasks:` (benchmarks, the test bench checklist, data labels).

Each task is a dict from the scene YAML:
    {type: lift,   object: cube, height: 0.10}          object raised `height` m above its start
    {type: place,  object: cube, in: bin_a}              object inside a bin fixture
    {type: stack,  object: cube_a, on: cube_b}           object resting on top of another object
    {type: push,   object: cube, to: goal, tol: 0.03}    object centre within `tol` of a target fixture
    {type: insert, object: peg, into: hole_a}            object lowered into a hole fixture
Optional on every task: `text` (HUD label) and `robot` (mount index or robot name it belongs to;
used when several robots are live, e.g. the test bench).

A task is *done* while its condition holds now; *achieved* latches once it has held (both are
reported, benchmarks normally use the final `done`).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

TYPES = ("lift", "place", "stack", "push", "insert")


def describe(t: dict) -> str:
    if t.get("text"):
        return str(t["text"])
    o = t["object"]
    return {"lift": f"Lift {o} {100 * float(t.get('height', 0.1)):.0f} cm",
            "place": f"Put {o} in {t.get('in')}",
            "stack": f"Stack {o} on {t.get('on')}",
            "push": f"Push {o} to {t.get('to')}",
            "insert": f"Insert {o} into {t.get('into')}"}.get(t["type"], t["type"])


def check(sim, t: dict, start: Dict[str, np.ndarray]) -> bool:
    kind, name = t["type"], t["object"]
    p, _ = sim.object_pose(name)
    o = sim.objects[name]["spec"]
    if kind == "lift":
        return p[2] - start[name][2] > float(t.get("height", 0.1))
    if kind == "place":
        b = sim.spec.fixture(t["in"])
        return b.contains(p[0], p[1]) and p[2] < b.base_z + b.size[2] + 0.03
    if kind == "stack":
        q, _ = sim.object_pose(t["on"])
        u = sim.objects[t["on"]]["spec"]
        dz = p[2] - q[2] - (o.half_height + u.half_height)
        return float(np.linalg.norm(p[:2] - q[:2])) < 0.7 * min(u.size[0], u.size[1] if len(u.size) > 1 else u.size[0]) \
            and abs(dz) < 0.012
    if kind == "push":
        g = sim.spec.fixture(t["to"])
        return float(np.linalg.norm(p[:2] - np.asarray(g.pos[:2]))) < float(t.get("tol", 0.03))
    if kind == "insert":
        h = sim.spec.fixture(t["into"])
        return bool(np.all(np.abs(np.asarray(h.to_local(p[0], p[1]))) < h.hole)) and p[2] - o.half_height < h.top - 0.02
    raise ValueError(f"unknown task type '{kind}' (known: {', '.join(TYPES)})")


def belongs(t: dict, mount_index: int, robot: str) -> bool:
    r = t.get("robot")
    return r is None or r == mount_index or r == robot


class TaskTracker:
    """Evaluates the scene's tasks every control tick."""

    def __init__(self, sim):
        self.sim = sim
        self.reset()

    def reset(self):
        self.start = {n: self.sim.object_pose(n)[0] for n in self.sim.objects}
        self.achieved = [False] * len(self.sim.spec.tasks)
        self.done = list(self.achieved)

    def update(self):
        for i, t in enumerate(self.sim.spec.tasks):
            self.done[i] = check(self.sim, t, self.start)
            self.achieved[i] = self.achieved[i] or self.done[i]

    def status(self, arm=None) -> List[dict]:
        out = []
        for i, t in enumerate(self.sim.spec.tasks):
            if arm is None or belongs(t, arm.mount_index, arm.name):
                out.append({"index": i, "text": describe(t), "done": self.done[i], "achieved": self.achieved[i], "task": t})
        return out

    def success(self, arm=None) -> bool:
        """Every task (of `arm`) has been achieved at some point, like robosuite's success check."""
        s = self.status(arm)
        return bool(s) and all(x["achieved"] for x in s)
