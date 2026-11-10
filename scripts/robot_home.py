#!/usr/bin/env python
"""Solve a robot's home joint configuration: TCP at a base-frame point, tool pointing straight down,
fingers closing along the base y axis. All robots share the same home TCP so scenes and scripts
work unchanged across robots.

    python scripts/robot_home.py ur5e                 # print the solution
    python scripts/robot_home.py ur5e --write         # also update `home:` in assets/robot/ur5e.yaml
    python scripts/robot_home.py franka_fr3 --tcp 0.45 0 0.37
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cobot_sandbox.mathutil import quat_err, tool_down_quat  # noqa: E402
from cobot_sandbox.paths import ROBOT_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot")
    ap.add_argument("--tcp", type=float, nargs=3, default=[0.45, 0.0, 0.37], help="base-frame TCP (m)")
    ap.add_argument("--write", action="store_true", help="write the result into the robot's YAML")
    a = ap.parse_args()

    import mujoco
    from cobot_sandbox.sim import Sim
    scene = {"name": "home_probe", "robots": {"mounts": [{"robot": a.robot, "pos": [0, 0], "stand": None, "z": 0.0}]}}
    sim = Sim(scene, robot=a.robot)
    arm, m, d = sim.arm, sim.model, sim.data
    goal_p = np.array(a.tcp)
    lo = m.jnt_range[[m.dof_jntid[i] for i in arm.arm_d], 0]
    hi = m.jnt_range[[m.dof_jntid[i] for i in arm.arm_d], 1]
    best = None
    # A parallel gripper is symmetric: yaw 0 and yaw pi give the same grasp; keep the one whose
    # joints stay furthest from their limits.
    for yaw in (0.0, math.pi):
        goal_q = tool_down_quat(yaw)
        q = arm.home.copy()
        p0, _ = arm.ik.fk(d.qpos, q)       # seed: turn the first (vertical) joint toward the goal
        q[0] += math.atan2(goal_p[1], goal_p[0]) - math.atan2(p0[1], p0[0])
        for _ in range(400):
            q, _, _, _, _ = arm.ik.solve(d.qpos, q, goal_p, goal_q)
        q = np.clip(q, lo, hi)
        p, qa = arm.ik.fk(d.qpos, q)
        perr, rerr = np.linalg.norm(p - goal_p), np.linalg.norm(quat_err(goal_q, qa))
        margin = float(np.min(np.minimum(q - lo, hi - q)))
        score = perr * 100 + rerr - 0.2 * min(margin, 0.5)
        print(f"  yaw {math.degrees(yaw):5.1f}: pos err {1000 * perr:.2f} mm, rot err {math.degrees(rerr):.2f} deg, "
              f"closest joint limit {math.degrees(margin):.1f} deg")
        if best is None or score < best[0]:
            best = (score, q, perr, rerr)
    _, q, perr, rerr = best
    print(f"{a.robot}: position error {1000 * perr:.2f} mm, orientation error {math.degrees(rerr):.2f} deg")
    vals = "[" + ", ".join(f"{v:.4f}" for v in q) + "]"
    print("home:", vals)
    if a.write:
        path = ROBOT_DIR / f"{a.robot}.yaml"
        t = path.read_text()
        t, n = re.subn(r"^home: \[[^\]]*\]", f"home: {vals}", t, count=1, flags=re.M)
        if not n:
            sys.exit("no `home:` line found")
        path.write_text(t)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
