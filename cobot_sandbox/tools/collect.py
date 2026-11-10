"""Headless demonstration collection with a scripted policy + domain randomization.

    cobot-sandbox-collect --scene lift --robot franka_fr3 --episodes 50
    cobot-sandbox-collect --scene stack --robot ur5e --episodes 20 --cameras wrist_cam,front --image-size 128
    cobot-sandbox-collect --scene pick_place --robot franka_fr3 --policy clear_table --episodes 10

The default policy "task" solves the scene's own tasks. Episodes go to data/episodes/<scene>/<robot>/
(robots have different joint counts, so keep their data apart). Only successful episodes are kept
unless --keep-failures is given; episodes with a physics instability are always marked failed.
"""
import argparse
import sys
import time

import numpy as np


from cobot_sandbox.paths import EPISODE_DIR
from cobot_sandbox.policies import SCRIPTED, make_policy
from cobot_sandbox.recorder import Recorder
from cobot_sandbox.sim import Sim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="lift")
    ap.add_argument("--robot", default=None, help="franka_fr3 or ur5e (default: the scene's first)")
    ap.add_argument("--policy", default="task", choices=SCRIPTED)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--cameras", default="", help="comma-separated camera names to record images from")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--max-seconds", type=float, default=0, help="per-episode sim-time limit (default: by task)")
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--latency", type=int, default=0, help="action latency in control ticks")
    ap.add_argument("--out", default=None, help=f"output folder (default {EPISODE_DIR}/<scene>)")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--dart", type=float, default=0.0,
                    help="std (m/s) of correlated noise added to executed TCP commands; clean labels are saved too")
    a = ap.parse_args()

    cams = [c for c in a.cameras.split(",") if c]
    from cobot_sandbox.robots import RobotModelsMissing
    try:
        sim = Sim(a.scene, seed=a.seed, randomize=not a.no_randomize, robot=a.robot, obs_noise=a.obs_noise,
                  action_latency=a.latency)
    except RobotModelsMissing as e:
        sys.exit(str(e))
    if not sim.spec.recording:
        sys.exit(f"Scene '{sim.spec.name}' is for free testing and has recording disabled; use a benchmark scene "
                 f"(cobot-sandbox --list).")
    if a.robot and sim.robot_name != a.robot:
        sys.exit(f"robot '{a.robot}' is not available in '{sim.spec.name}' (got {sim.robot_name}); "
                 f"its model files may be missing")
    out = a.out or EPISODE_DIR / sim.spec.name / sim.robot_name
    print(f"collecting {a.episodes} x '{a.policy}' on {sim.spec.name} with {sim.arm.title} -> {out}")
    rec = Recorder(out_dir=out, cameras=cams, image_size=(a.image_size, a.image_size))
    kept = 0
    t0 = time.time()
    for ep in range(a.episodes):
        obs = sim.reset(seed=a.seed + ep)
        pol = make_policy(a.policy, sim)
        pol.reset(sim)
        rec.start(sim, source=a.policy)
        unstable0 = getattr(sim, "unstable_count", 0)
        max_s = a.max_seconds or ({"pick_place": 60, "clear_table": 300}.get(a.policy) or 30 + 30 * len(sim.spec.tasks))
        limit = int(max_s / sim.control_dt)
        rng = np.random.default_rng(a.seed + ep)
        drift = np.zeros(3)
        while not pol.done and rec.length < limit:
            cmd = pol.act(obs, sim)
            label = cmd.to_array()
            if a.dart:   # Ornstein-Uhlenbeck noise on the executed command (DART-style data augmentation)
                drift += -0.15 * drift + rng.normal(0, a.dart * 0.55, 3)
                cmd.ee_lin = cmd.ee_lin + drift
            rec.add(obs, cmd.to_array(), pol.task_info(), sim, label=label if a.dart else None)
            obs = sim.step(cmd)
        blew_up = getattr(sim, "unstable_count", 0) > unstable0   # physics reset mid-episode: bad data
        ok = bool(pol.success) and not blew_up
        path = rec.stop(save=ok or a.keep_failures, success=ok, name=f"{sim.spec.name}_{sim.robot_name}_{a.policy}_s{a.seed + ep}")
        kept += path is not None
        print(f"[{ep + 1}/{a.episodes}] seed {a.seed + ep}: {'OK ' if ok else 'FAIL'} {pol.status:40s} "
              f"{'-> ' + path.name if path else '(discarded)'}{'  [physics instability]' if blew_up else ''}")
    print(f"kept {kept}/{a.episodes} episodes in {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
