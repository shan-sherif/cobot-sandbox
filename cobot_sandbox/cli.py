"""Cobot Sandbox app launcher (installed as the `cobot-sandbox` command; `cobot-sandbox ` in a checkout).

    cobot-sandbox                                # pick a scene from the menu, manual control (H for help)
    cobot-sandbox  --scene lift --robot ur5e     # go straight to a scene with a given robot (no menus)
    cobot-sandbox  --scene stack --policy task   # watch the scripted solver (P toggles, any key takes over)
    cobot-sandbox  --replay data/episodes/x.npz
    cobot-sandbox  --list                   # list scenes
"""
import argparse
import sys


def main():
    from cobot_sandbox.policies import SCRIPTED
    from cobot_sandbox.scene_builder import list_scenes, scene_info

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", help="scene name or path to a scene YAML (default: show the scene menu)")
    ap.add_argument("--policy", default="teleop", choices=["teleop"] + SCRIPTED)
    ap.add_argument("--robot", help="robot for single-robot scenes: franka_fr3 or ur5e "
                                    "(default: ask in the app, remembered in settings.yaml)")
    ap.add_argument("--replay", help="episode .npz to play back")
    ap.add_argument("--size", default="1280x720", help="window size WxH (max 1920x1080)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true", help="apply the scene's domain randomization on reset")
    ap.add_argument("--labels", action=argparse.BooleanOptionalAction, default=None,
                    help="floating labels in the 3D view (default: settings.yaml show_labels, else on)")
    ap.add_argument("--record-cameras", default="", help="comma-separated cameras to save images from, e.g. wrist_cam,overview")
    ap.add_argument("--list", action="store_true", help="list scenes and exit")
    ap.add_argument("--frames", type=int, default=0, help=argparse.SUPPRESS)       # smoke testing
    ap.add_argument("--screenshot", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.list:
        for s in scene_info():
            num = f"{s['number']:>3}" if s["number"] is not None else "  -"
            print(f"{num}  {s['name']:<16} {s['title']:<18} {s['file']}")
        return

    from cobot_sandbox.app import App
    w, h = (int(v) for v in a.size.lower().split("x"))
    show_menu = a.scene is None and not a.replay and a.policy == "teleop"
    from cobot_sandbox.robots import RobotModelsMissing
    try:
        app = App(scene=a.scene or list_scenes()[0],   # first scene is shown behind the menu
                  policy=None if a.policy == "teleop" else a.policy, size=(w, h), seed=a.seed,
                  randomize=a.randomize, record_cameras=[c for c in a.record_cameras.split(",") if c],
                  replay=a.replay, show_menu=show_menu, show_labels=a.labels,
                  robot=a.robot)
    except RobotModelsMissing as e:
        sys.exit(f"Can't start: {e}")
    if a.frames:                     # headless smoke test: run N frames, optionally save the last one
        import pygame
        for _ in range(a.frames):
            app.handle_events()
            app.tick_sim(1 / 60)
            app.render()
        if a.screenshot:
            pygame.image.save(app.screen, a.screenshot)
        print("ok", app.sim.robot_name, f"{app.obs['time'][0]:.2f}s")
        return
    app.run()


if __name__ == "__main__":
    main()
