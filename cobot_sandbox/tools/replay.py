"""Inspect or verify recorded episodes.

    cobot-sandbox-replay data/episodes/lift/franka_fr3/x.npz            # summary
    cobot-sandbox-replay data/episodes/lift/franka_fr3/x.npz --verify   # headless replay, check determinism
    cobot-sandbox --replay data/episodes/lift/franka_fr3/x.npz              # watch it in the sandbox
"""
import argparse
import sys

import numpy as np


from cobot_sandbox.command import ACTION_LABELS, Command
from cobot_sandbox.recorder import load_episode, sim_for_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+")
    ap.add_argument("--verify", action="store_true", help="re-simulate and compare against recorded observations")
    a = ap.parse_args()
    for path in a.episodes:
        ep = load_episode(path)
        m = ep["meta"]
        print(f"{path}\n  scene={m['scene']} seed={m['seed']} randomize={m['randomize']} source={m['source']} "
              f"success={m.get('success')} steps={len(ep['action'])} ({len(ep['action']) * m['control_dt']:.1f}s)")
        print(f"  obs keys: {', '.join(sorted(ep['obs']))[:200]}...")
        if ep["task"]:
            print(f"  task keys: {', '.join(ep['task'])}")
        if ep["img"]:
            print("  images: " + ", ".join(f"{k} {v.shape}" for k, v in ep["img"].items()))
        act = ep["action"]
        print("  action range: " + "  ".join(f"{l}[{act[:, i].min():+.2f},{act[:, i].max():+.2f}]"
                                             for i, l in enumerate(ACTION_LABELS) if np.ptp(act[:, i]) > 0))
        if a.verify:
            sim = sim_for_episode(ep)
            ref = ep["obs"]["tcp_pos_world"]
            err = 0.0
            for t, row in enumerate(act):
                err = max(err, float(np.abs(sim.get_obs()["tcp_pos_world"] - ref[t]).max()))
                sim.step(Command.from_array(row))
            print(f"  replay max TCP deviation: {err * 1000:.3f} mm  ->  {'deterministic' if err < 1e-4 else 'DIVERGED'}")


if __name__ == "__main__":
    main()
