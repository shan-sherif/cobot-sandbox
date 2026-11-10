"""Scripted (automated) behaviours. They emit the same velocity Commands a human does, so their
rollouts are valid demonstrations for imitation learning."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..command import Command
from ..mathutil import clip_norm, quat_err, quat_yaw, tool_down_quat, tool_yaw
from .base import Policy, Sequence

LIN_GAIN, ANG_GAIN = 4.0, 3.0
LIN_MAX, ANG_MAX = 0.22, 1.0


def servo(sim, goal_pos_b, goal_quat_b, lin_max=LIN_MAX):
    """Velocity command that drives the TCP target toward a base-frame goal pose."""
    ep = goal_pos_b - sim.ee_target_pos
    er = quat_err(goal_quat_b, sim.ee_target_quat)
    return clip_norm(LIN_GAIN * ep, lin_max), clip_norm(ANG_GAIN * er, ANG_MAX), ep, er


class PickPlace(Policy):
    """Pick one object (nearest reachable, or `target`) and put it somewhere:
        default        the nearest reachable bin (or bin `place_in`)
        place=(x, y)   on the surface at a world x, y
        place_on=name  on top of another object (stacking)
        into=name      above a hole fixture, released so it drops in (peg insertion)
        lift=h         lift it h m, hold until the scene's lift task is met, then put it back down
    """
    name = "pick_place"

    def __init__(self, target: Optional[str] = None, place=None, seed: int = 0, exclude=(),
                 place_in: Optional[str] = None, place_on: Optional[str] = None, into: Optional[str] = None,
                 lift: Optional[float] = None):
        super().__init__()
        self.target_name, self.place_arg = target, place
        self.place_in, self.place_on, self.into, self.lift = place_in, place_on, into, lift
        self.exclude = set(exclude)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- planning helpers
    @staticmethod
    def in_bin(sim, p_w) -> bool:
        return any(b.contains(p_w[0], p_w[1]) and p_w[2] < b.base_z + b.size[2] + 0.03 for b in sim.spec.bins())

    @staticmethod
    def reachable(sim, p_w) -> bool:
        p_b = sim.to_base(p_w)
        r = math.hypot(p_b[0], p_b[1])
        lo, hi = sim.arm.reach
        return lo < r < hi and p_b[2] > -0.15

    def candidates(self, sim) -> List[str]:
        out = []
        for name in sim.objects:
            p, _ = sim.object_pose(name)
            if name not in self.exclude and self.reachable(sim, p) and not self.in_bin(sim, p):
                out.append(name)
        return out

    def _choose_target(self, sim):
        if self.target_name:
            return self.target_name
        c = self.candidates(sim)
        if not c:
            return None
        tcp = sim.tcp_pose("world")[0]
        return min(c, key=lambda n: np.linalg.norm(sim.object_pose(n)[0][:2] - tcp[:2]))

    def _place_point(self, sim, obj):
        half = obj.half_height
        if self.lift is not None:                      # put it back where it was
            p, _ = sim.object_pose(self.obj_name)
            return np.array([p[0], p[1], p[2] + 0.005])
        if self.place_on is not None:
            q, _ = sim.object_pose(self.place_on)
            u = sim.objects[self.place_on]["spec"]
            return np.array([q[0], q[1], q[2] + u.half_height + half + 0.008])
        if self.into is not None:
            h = sim.spec.fixture(self.into)
            return np.array([h.pos[0], h.pos[1], h.top + half + 0.012])
        if self.place_arg is not None:
            x, y = self.place_arg
            return np.array([x, y, sim.spec.surface_height(x, y) + half + 0.02])
        bins = [b for b in sim.spec.bins() if (self.place_in is None or b.name == self.place_in)
                and self.reachable(sim, np.array([*b.pos, b.base_z + 0.1]))]
        if bins:
            tcp = sim.tcp_pose("world")[0]
            b = min(bins, key=lambda b: np.hypot(*(b.pos - tcp[:2])))
            # drop into the emptiest part of the bin so objects don't land on (and bounce off) each other
            inside = [sim.object_pose(n)[0] for n in sim.objects if self.in_bin(sim, sim.object_pose(n)[0])]
            m = max(0.0, min(b.size[0], b.size[1]) - obj.footprint - 0.045)   # keep clear of the walls
            best, best_d = np.array(b.pos, float), -1.0
            for lx, ly in [(0, 0), (m, m), (m, -m), (-m, m), (-m, -m), (m, 0), (-m, 0), (0, m), (0, -m)]:
                xy = np.array(b.to_world(lx, ly))
                d = min([np.linalg.norm(xy - p[:2]) for p in inside], default=1.0)
                if d > best_d + 1e-3:
                    best, best_d = xy, d
            # release as low as the hand can go while staying above the walls (short, clean drop)
            wall_top = b.base_z + b.size[2]
            return np.array([best[0], best[1], max(b.top + half + 0.008, wall_top + 0.015)])
        return None

    def _grasp_yaw(self, sim, name, current):
        o = sim.objects[name]["spec"]
        _, q_b = sim.object_pose(name, "base")
        psi = quat_yaw(q_b)
        if o.type == "cube":
            period, ref = math.pi / 2, psi
        elif o.type == "box":
            period = math.pi
            ref = psi if o.size[0] >= o.size[1] else psi + math.pi / 2
        else:
            return current
        k = round((current - ref) / period)
        return ref + k * period

    # ---------------------------------------------------------------- policy API
    def reset(self, sim):
        super().reset(sim)
        self._sim = sim
        self.obj_name = self._choose_target(sim)
        self.phase_i, self.phase_t = 0, 0.0
        if self.obj_name is None:
            self.done, self.success, self.status = True, False, "nothing reachable to pick"
            return
        spec = sim.objects[self.obj_name]["spec"]
        p_obj, _ = sim.object_pose(self.obj_name)
        place = self._place_point(sim, spec)
        if place is None:
            self.done, self.success, self.status = True, False, "no place location in reach"
            return
        surf = sim.spec.surface_height(p_obj[0], p_obj[1])
        grasp_z = max(p_obj[2], surf + 0.03)
        cur_yaw = tool_yaw(sim.ee_target_quat)
        yaw_g = self._grasp_yaw(sim, self.obj_name, cur_yaw)
        self.start_z = p_obj[2]
        self.lifted = False
        self.place_w = place
        P = lambda xy, z: np.array([xy[0], xy[1], z])
        lift_h = 0.15 if self.lift is None else self.lift + 0.04
        place = place + np.array([0, 0, grasp_z - p_obj[2]])      # keep the grasp offset when placing
        # (name, world pos, tool yaw, gripper, min time, tolerance)
        self.phases = [
            ("approach", P(p_obj, grasp_z + 0.12), yaw_g, 0.0, 0.0, 0.008),
            ("descend", P(p_obj, grasp_z), yaw_g, 0.0, 0.0, 0.004),
            ("grasp", P(p_obj, grasp_z), yaw_g, 1.0, 0.7, 0.01),
            ("lift", P(p_obj, grasp_z + lift_h), yaw_g, 1.0, 0.0, 0.01),
        ]
        if self.lift is not None:
            self.phases.append(("hold", P(p_obj, grasp_z + lift_h), yaw_g, 1.0, 0.8, 0.01))
        self.phases += [
            ("transit", P(place, place[2] + 0.08), yaw_g, 1.0, 0.0, 0.01),
            ("lower", place, yaw_g, 1.0, 0.0, 0.006 if self.into is None else 0.004),
            ("release", place, yaw_g, 0.0, 0.6, 0.02),
            ("retreat", P(place, place[2] + 0.12), yaw_g, 0.0, 0.0, 0.01),
        ]

    def act(self, obs, sim):
        if self.done:
            return Command(gripper=0.0)
        name, goal_w, yaw, grip, min_t, tol = self.phases[self.phase_i]
        goal_b = sim.to_base(goal_w)
        lin, ang, ep, er = servo(sim, goal_b, tool_down_quat(yaw), lin_max=0.12 if name in ("descend", "lower") else LIN_MAX)
        self.phase_t += sim.control_dt
        self.status = f"{self.name}: {name} {self.obj_name}"
        tcp_err = np.linalg.norm(obs["tcp_pos"] - goal_b)
        if name == "hold" and sim.object_pose(self.obj_name)[0][2] - self.start_z > self.lift:
            self.lifted = True
        if self.phase_t >= min_t and np.linalg.norm(ep) < tol and tcp_err < tol + 0.012 and np.linalg.norm(er) < 0.05:
            if name == "lift" and sim.object_pose(self.obj_name)[0][2] < self.start_z + min(0.05, lift_h_check(self)):
                self.done, self.success, self.status = True, False, f"grasp failed on {self.obj_name}"
            elif self.phase_i + 1 == len(self.phases):
                p = sim.object_pose(self.obj_name)[0]
                self.done = True
                if self.lift is not None:
                    self.success = self.lifted
                elif self.place_on is not None or self.into is not None or self.place_arg is not None:
                    self.success = bool(np.linalg.norm(p[:2] - self.place_w[:2]) < 0.02 + 0.03 * (self.place_arg is not None))
                else:
                    self.success = bool(self.in_bin(sim, p))
                self.status = f"placed {self.obj_name}" if self.success else f"missed placing {self.obj_name}"
            else:
                self.phase_i += 1
                self.phase_t = 0.0
        elif self.phase_t > 8.0:
            self.done, self.success, self.status = True, False, f"timeout in {name}"
        return Command(ee_lin=lin, ee_ang=ang, gripper=grip)

    def task_info(self):
        if self.obj_name is None:
            return {}
        return {"task_target_pos": np.asarray(self.phases[0][1]), "task_place_pos": np.asarray(self.place_w),
                "task_obj_pos": self._sim.object_pose(self.obj_name)[0],
                "task_phase": np.array([self.phase_i], dtype=np.float32)}


def lift_h_check(p) -> float:
    return 0.05 if p.lift is None else 0.5 * p.lift


class ClearTable(Policy):
    """Keep picking reachable objects into the bin until none are left."""
    name = "clear_table"

    def __init__(self, max_objects: int = 20, seed: int = 0):
        super().__init__()
        self.max_objects, self.seed = max_objects, seed

    def reset(self, sim):
        super().reset(sim)
        self.count, self.fails = 0, 0
        self.failed = set()
        self._next(sim)

    def _next(self, sim):
        self.cur = PickPlace(seed=self.seed + self.count, exclude=self.failed)
        self.cur.reset(sim)
        if self.cur.done:
            self.done, self.success = True, self.count > 0 or self.fails == 0
            self.status = f"cleared {self.count} object(s)"

    def act(self, obs, sim):
        if self.done:
            return Command()
        cmd = self.cur.act(obs, sim)
        self.status = f"clear_table #{self.count + 1}: {self.cur.status}"
        if self.cur.done:
            self.count += bool(self.cur.success)
            self.fails += not self.cur.success
            if not self.cur.success and self.cur.obj_name:
                self.failed.add(self.cur.obj_name)
            if self.count >= self.max_objects or self.fails >= 3:
                self.done, self.success = True, self.count > 0 or self.fails == 0
            else:
                self._next(sim)
        return cmd

    def task_info(self):
        return self.cur.task_info()


class Push(Policy):
    """Push an object along the table to a target fixture with the closed gripper."""
    name = "push"

    def __init__(self, target: str, to: str, tol: float = 0.03):
        super().__init__()
        self.target_name, self.to, self.tol = target, to, tol

    def reset(self, sim):
        super().reset(sim)
        self.phase, self.t = "above", 0.0
        self.goal = np.asarray(sim.spec.fixture(self.to).pos[:2], float)
        self.yaw = tool_yaw(sim.ee_target_quat)

    def _geometry(self, sim):
        p, _ = sim.object_pose(self.target_name)
        o = sim.objects[self.target_name]["spec"]
        d = self.goal - p[:2]
        dist = float(np.linalg.norm(d))
        u = d / dist if dist > 1e-6 else np.array([1.0, 0.0])
        z = sim.spec.surface_height(p[0], p[1]) + 0.012          # fingertips just above the table
        behind = p[:2] - u * (o.footprint + 0.03)
        return p, u, dist, z, behind

    def act(self, obs, sim):
        if self.done:
            return Command(gripper=1.0)
        self.t += sim.control_dt
        p, u, dist, z, behind = self._geometry(sim)
        if dist < self.tol:
            self.phase = "retreat"
        if self.phase == "above":
            goal = np.array([behind[0], behind[1], z + 0.1])
        elif self.phase == "down":
            goal = np.array([behind[0], behind[1], z])
        elif self.phase == "push":
            tcp = sim.tcp_pose("world")[0][:2]
            lateral = (p[:2] - tcp) - u * float(np.dot(p[:2] - tcp, u))   # steer back onto the line
            step = u * min(0.04, dist + 0.01) + 0.8 * lateral
            goal = np.array([tcp[0] + step[0], tcp[1] + step[1], z])
        else:
            tcp = sim.tcp_pose("world")[0]
            goal = np.array([tcp[0] - u[0] * 0.02, tcp[1] - u[1] * 0.02, z + 0.12])
        goal_b = sim.to_base(goal)
        lin, ang, ep, er = servo(sim, goal_b, tool_down_quat(self.yaw), lin_max=0.08 if self.phase == "push" else LIN_MAX)
        near = np.linalg.norm(ep) < 0.008
        if self.phase == "above" and near:
            self.phase = "down"
        elif self.phase == "down" and near:
            self.phase = "push"
        elif self.phase == "retreat" and near:
            self.done, self.success = True, dist < self.tol
        elif self.t > 25:
            self.done, self.success = True, False
        self.status = f"push: {self.phase} {self.target_name} ({100 * dist:.1f} cm to go)"
        return Command(ee_lin=lin, ee_ang=ang, gripper=1.0)


class TaskPolicy(Sequence):
    """Solve the scene's tasks (for the active robot) one after another with the scripts above."""
    name = "task"

    def __init__(self, sim):
        steps = []
        for st in sim.tasks.status(sim.arm):
            tk = st["task"]
            kind, obj = tk["type"], tk["object"]
            if kind == "lift":
                pol = PickPlace(target=obj, lift=float(tk.get("height", 0.1)))
            elif kind == "place":
                pol = PickPlace(target=obj, place_in=tk["in"])
            elif kind == "stack":
                pol = PickPlace(target=obj, place_on=tk["on"])
            elif kind == "insert":
                pol = PickPlace(target=obj, into=tk["into"])
            elif kind == "push":
                pol = Push(obj, tk["to"], float(tk.get("tol", 0.03)))
            else:
                continue
            pol.optional = True          # keep going if one task fails; success is judged at the end
            steps.append(pol)
        super().__init__(steps or [Policy()])
        self._tasks = sim.tasks

    def act(self, obs, sim):
        cmd = super().act(obs, sim)
        if self.done:
            s = self._tasks.status(sim.arm)
            self.success = bool(s) and all(x["achieved"] for x in s)
            self.status = f"tasks: {sum(x['achieved'] for x in s)}/{len(s)} achieved"
        return cmd
