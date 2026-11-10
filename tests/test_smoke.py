"""Fast end-to-end checks. Run:  python -m pytest -q tests

Uses the bundled Franka FR3 / UR5e models (cobot_sandbox/assets/third_party/menagerie/).
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cobot_sandbox import robots as R  # noqa: E402
from cobot_sandbox.command import Command  # noqa: E402
from cobot_sandbox.policies import PickPlace, make_policy  # noqa: E402
from cobot_sandbox.recorder import Recorder, load_episode, sim_for_episode  # noqa: E402
from cobot_sandbox.scene_builder import build_model, build_xml, compile_scene, list_scenes, scene_info  # noqa: E402
from cobot_sandbox.sim import Sim  # noqa: E402

ROBOTS = ["franka_fr3", "ur5e"]
if not all(R.available(r) for r in ROBOTS):
    pytest.exit("Robot models missing from cobot_sandbox/assets/third_party/menagerie/: "
                "re-create them with  python scripts/fetch_models.py", returncode=1)



def _gl_available() -> bool:
    """Off-screen OpenGL rendering works here (CI machines often have none; Linux needs MUJOCO_GL)."""
    import mujoco
    try:
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8)
        r.close()
        return True
    except Exception:
        return False


needs_gl = pytest.mark.skipif(not _gl_available(), reason="no OpenGL for off-screen rendering "
                              "(on headless Linux set MUJOCO_GL=egl or MUJOCO_GL=osmesa)")

def run(policy, sim, limit_s=60):
    obs = sim.get_obs()
    policy.reset(sim)
    t0 = sim.data.time
    while not policy.done and sim.data.time - t0 < limit_s:
        obs = sim.step(policy.act(obs, sim))
    return policy.success


def test_scenes_listed_in_menu_order():
    info = scene_info()
    assert [s["name"] for s in info] == list_scenes()
    nums = [s["number"] for s in info if s["number"] is not None]
    assert nums == sorted(nums) and len(nums) == len(set(nums)), "scene numbers must be unique and ordered"
    assert info[0]["name"] == "test_bench" and not info[0]["recording"]


def test_all_scenes_compile():
    for s in list_scenes():
        spec = compile_scene(s)
        build_model(spec, build_xml(spec))


@pytest.mark.parametrize("robot", ROBOTS)
def test_arm_holds_and_tracks(robot):
    sim = Sim("lift", robot=robot)
    assert sim.robot_name == robot
    for _ in range(50):
        obs = sim.step()
    assert np.abs(obs["arm_qpos"] - obs["arm_qdes"]).max() < 2e-3
    assert np.allclose(obs["tcp_pos"], [0.45, 0.0, 0.37], atol=3e-3), "all robots share the same home TCP"
    start = obs["tcp_pos"].copy()
    for _ in range(50):
        sim.step(Command(ee_lin=np.array([0.0, 0.1, 0.0])))
    for _ in range(25):
        obs = sim.step()
    assert abs(obs["tcp_pos"][1] - start[1] - 0.1) < 0.005


@pytest.mark.parametrize("robot", ROBOTS)
def test_lift_task(robot):
    sim = Sim("lift", robot=robot)
    run(make_policy("task", sim), sim)
    assert sim.success and not getattr(sim, "unstable_count", 0)


@pytest.mark.parametrize("scene", ["stack", "push", "peg_insertion"])
def test_benchmark_tasks(scene):
    sim = Sim(scene, robot="franka_fr3")
    run(make_policy("task", sim), sim)
    assert sim.success


def test_pick_place_into_bin():
    assert run(PickPlace(), Sim("pick_place", robot="ur5e"))


def test_test_bench_two_live_robots():
    sim = Sim("test_bench")
    assert [a.name for a in sim.arms] == ["franka_fr3", "ur5e"] and not sim.spec.recording
    idle = sim.arms[1]
    q0 = sim.data.qpos[idle.arm_q].copy()
    for _ in range(40):                       # move the active robot; the other must not move
        sim.step(Command(ee_lin=np.array([0.0, 0.0, -0.1])))
    assert np.abs(sim.data.qpos[idle.arm_q] - q0).max() < 1e-4
    sim.next_robot()
    assert sim.robot_name == "ur5e" and len(sim.get_obs()["arm_qpos"]) == 6


def test_missing_models_give_a_clear_error(monkeypatch):
    monkeypatch.setattr(R, "available", lambda name: False)
    spec = compile_scene("test_bench")
    with pytest.raises(R.RobotModelsMissing, match="fetch_models.py"):
        R.resolve(spec.mounts, spec.robot_select)


def test_one_robot_missing_leaves_the_other(monkeypatch):
    monkeypatch.setattr(R, "available", lambda name: name == "ur5e")
    spec = compile_scene("lift")
    assert [m["robot"] for m in R.resolve(spec.mounts, spec.robot_select, choice="franka_fr3")] == ["ur5e"]


@pytest.mark.parametrize("robot", ROBOTS)
def test_record_and_replay_deterministic(robot):
    sim = Sim("stack", randomize=True, seed=3, robot=robot)
    for _ in range(10):      # start recording mid-session on purpose
        sim.step(Command(ee_lin=np.array([0.05, 0, 0])))
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(out_dir=d)
        pol = make_policy("task", sim)
        pol.reset(sim)
        rec.start(sim, source="test")
        obs = sim.get_obs()
        for _ in range(120):
            cmd = pol.act(obs, sim)
            rec.add(obs, cmd.to_array(), pol.task_info(), sim)
            obs = sim.step(cmd)
        ep = load_episode(rec.stop(success=True))
        assert ep["meta"]["robot"] == robot
        sim2 = sim_for_episode(ep)
        assert sim2.robot_name == robot
        for t, a in enumerate(ep["action"]):
            assert np.allclose(sim2.get_obs()["tcp_pos_world"], ep["obs"]["tcp_pos_world"][t], atol=1e-5)
            sim2.step(Command.from_array(a))


def test_randomization_changes_layout():
    sim = Sim("pick_place", randomize=True, robot="franka_fr3")
    a = sim.reset(seed=1)
    b = sim.reset(seed=2)
    keys = [k for k in a if k.startswith("obj.") and k.endswith(".pos")]
    assert any(np.linalg.norm(a[k] - b[k]) > 0.01 for k in keys)


@needs_gl
@pytest.mark.parametrize("robot", ROBOTS)
def test_wrist_camera(robot):
    img = Sim("lift", robot=robot).render("wrist_cam", 64, 48)
    assert img.shape == (48, 64, 3) and img.std() > 1


def test_settings_save(tmp_path, monkeypatch):
    from cobot_sandbox import settings
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.yaml")
    (tmp_path / "settings.yaml").write_text("show_labels: false\n")
    settings.save_setting("robot", "ur5e")
    s = settings.load_settings()
    assert s["robot"] == "ur5e" and s["show_labels"] is False


@needs_gl
def test_scene_menu_and_recording_flag():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from cobot_sandbox.app import App
    app = App(scene="lift", size=(640, 360), show_menu=True, robot="ur5e")
    try:
        assert app.menu.open and app.menu.scenes[app.menu.sel]["name"] == "lift"
        t0 = app.obs["time"][0]
        app.tick_sim(0.2)
        assert app.obs["time"][0] == t0, "sim must be paused while the menu is open"
        for k in (pygame.K_UP, pygame.K_RETURN):          # Up from scene 1 -> scene 0 (test bench)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))
        assert app.handle_events()
        assert not app.menu.open and app.scene_name == "test_bench"
        app.toggle_recording()
        assert not app.recorder.active, "recording is disabled in the test bench"
    finally:
        app.sim.close()
        pygame.quit()
