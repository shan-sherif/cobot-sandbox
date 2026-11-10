"""Interactive sandbox window (pygame + MuJoCo offscreen renderer). Works on macOS with plain
`python` (no mjpython needed), on Linux and on Windows."""
from __future__ import annotations

import math
import time
from typing import Optional

import mujoco
import numpy as np
import pygame

from .command import Command
from .icon import make_icon
from .scene_menu import SceneMenu
from .policies import SCRIPTED, make_policy
from .recorder import Recorder
from .robot_picker import RobotPicker
from .scene_builder import list_scenes, scene_label
from .settings import load_settings, save_setting
from .sim import Sim
from .teleop import Teleop

HELP = [
    ("Arm", "W/S A/D R/F move TCP  |  Q/E yaw  Up/Down pitch  Left/Right roll  |  Shift precision"),
    ("Gripper", "Space open/close   Z arm home"),
    ("Robots", "Tab switch robot (scenes with several)   R choose robot (single-robot scenes)"),
    ("Autopilot", "P run/stop scripted policy  O cycle policy  (any move key takes over)"),
    ("Data", "Enter start/stop recording  (saved to data/episodes/; off in the test bench)"),
    ("Camera", "Drag orbit  Right-drag pan  Wheel zoom  C cycle cameras  T follow  V wrist view"),
    ("Scene", "L scene menu  [ / ] next/prev scene  Backspace reset  N randomization  K contacts  F2 pause"),
    ("View", "G labels on/off (default in settings.yaml)"),
    ("Gamepad", "Sticks move  A grip  X home  Y switch robot  LB/RB roll  LT slow  Start rec  Back reset"),
    ("", "H / F1 toggle this help   Esc quit"),
]


class App:
    def __init__(self, scene="test_bench", policy: Optional[str] = None, size=(1280, 720), seed=0,
                 randomize=False, record_cameras=(), replay=None, show_menu=False, show_labels=None,
                 robot: Optional[str] = None):
        pygame.init()
        self.W, self.H = size
        pygame.display.set_icon(make_icon())        # before set_mode, so the window/Dock picks it up
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Cobot Sandbox")
        self.font = pygame.font.SysFont("menlo,monaco,consolas,dejavusansmono", 14)
        self.font_b = pygame.font.SysFont("menlo,monaco,consolas,dejavusansmono", 16, bold=True)
        self.font_big = pygame.font.SysFont("menlo,monaco,consolas,dejavusansmono", 22, bold=True)
        self.menu = SceneMenu(on_select=self.load_from_menu)
        self.picker = RobotPicker(on_select=self.choose_robot)
        self.clock = pygame.time.Clock()
        self.scenes = list_scenes()
        self.seed, self.randomize = seed, randomize
        self.record_cameras = list(record_cameras)
        self.replay_path = replay
        self.teleop = Teleop()
        self.auto_names = list(SCRIPTED)
        self.auto_name = policy if policy and policy != "teleop" else None
        self.auto_running = False
        self.show_help, self.show_pip, self.follow, self.paused, self.contacts = False, True, True, False, False
        self.settings = load_settings()
        self.show_labels = self.settings["show_labels"] if show_labels is None else bool(show_labels)
        self.robot_choice = robot or self.settings.get("robot")
        self.ask_robot = robot is None and not self.auto_name and not replay   # picker only when interactive
        self.toasts = []
        self.load_scene(scene)
        if self.auto_name or replay:
            self.start_auto()
        if show_menu:
            self.menu.show(current=self.scene_name)

    # ------------------------------------------------------------------ scene / policy
    def load_scene(self, scene, ask_robot: Optional[bool] = None):
        if getattr(self, "sim", None):
            self.sim.close()
        t = time.time()
        if self.replay_path:     # rebuild exactly the state the episode was recorded from
            from .recorder import load_episode, sim_for_episode
            self.sim = sim_for_episode(load_episode(self.replay_path))
        else:
            self.sim = Sim(scene, seed=self.seed, randomize=self.randomize, robot=self.robot_choice)
        self.scene_name = self.sim.spec.name
        self.obs = self.sim.get_obs()
        self.recorder = Recorder(cameras=self.record_cameras)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 145.0, -24.0, 2.6
        self.cam.lookat[:] = self._follow_point()
        view = self.sim.spec.raw.get("view") or {}       # optional per-scene starting camera
        if view:
            self.cam.azimuth = float(view.get("azimuth", self.cam.azimuth))
            self.cam.elevation = float(view.get("elevation", self.cam.elevation))
            self.cam.distance = float(view.get("distance", self.cam.distance))
            if "lookat" in view:
                self.cam.lookat[:] = view["lookat"]
            self.follow = bool(view.get("follow", self.follow))
        self.cam_modes = ["free"] + self.sim.cameras
        self.cam_i = 0
        self.opt = mujoco.MjvOption()
        self.main_r = self.sim.renderer(self.W, self.H)
        self.pip_r = self.sim.renderer(320, 240)
        self.accum = 0.0
        self.resets = 0
        self.auto_running = False
        self.teleop.grip = 0.0
        self.toast(f"Loaded {scene_label(self.sim.spec.raw)} with {self.sim.arm.title} in {time.time() - t:.1f}s  -  H for help")
        from . import robots as R
        if self.robot_choice and self.robot_choice != self.sim.robot_name and not R.available(self.robot_choice):
            self.toast(f"{self.robot_choice}: model files missing, using {self.sim.arm.title}")
        if len(self.sim.arms) > 1:
            self.toast(f"{len(self.sim.arms)} robots live: Tab switches")
        if not self.sim.spec.recording:
            self.toast("No recording (.npz) in this scene")
        if (self.ask_robot if ask_robot is None else ask_robot) and self.robot_options():
            self.picker.show(self.robot_options(), current=self.sim.robot_name)

    def robot_options(self):
        """Robots the user can choose between in this scene ([] if there's no choice to make)."""
        if self.replay_path or self.sim.spec.robot_select != "one":
            return []
        names = list(dict.fromkeys(m["robot"] for m in self.sim.spec.mounts))
        return names if len(names) > 1 else []

    def choose_robot(self, name):
        self.robot_choice = name
        try:
            save_setting("robot", name)
        except OSError as e:                     # read-only checkout etc.: keep going without saving
            self.toast(f"Could not save settings.yaml: {e}")
        if name != self.sim.robot_name:
            self.stop_recording(save=True)
            self.load_scene(self.scene_name, ask_robot=False)

    def switch_robot(self):
        if len(self.sim.arms) < 2:
            self.toast("One robot in this scene" + ("  (R to choose another)" if self.robot_options() else ""))
            return
        if self.recorder.active:
            self.stop_recording(save=True)
        self.stop_auto("stopped (robot switched)")
        self.sim.next_robot()
        self.teleop.grip = float(self.sim.grip_cmd)

    def load_from_menu(self, name):
        self.stop_recording(save=True)
        self.screen.blit(self.font_big.render("Loading...", True, (235, 235, 235)), (self.W // 2 - 60, self.H // 2 - 12))
        pygame.display.flip()
        self.replay_path = None
        self.load_scene(name)

    def _make_auto(self):
        if self.replay_path:
            from .policies.replay import Replay
            return Replay(self.replay_path)
        name = self.auto_name or ("task" if self.sim.spec.tasks else "clear_table")
        self.auto_name = name
        return make_policy(name, self.sim)

    def start_auto(self):
        try:
            self.auto = self._make_auto()
            self.auto.reset(self.sim)
            self.auto_running = True
            self.toast(f"Autopilot: {self.auto.name}")
        except Exception as e:  # keep the sandbox alive on bad policy config
            self.toast(f"Autopilot error: {e}")
            self.auto_running = False

    def stop_auto(self, why="stopped"):
        if self.auto_running:
            self.auto_running = False
            self.teleop.grip = float(self.sim.grip_cmd)
            self.toast(f"Autopilot {why}")

    def reset(self):
        self.stop_recording(save=False)
        if self.replay_path:
            self.load_scene(self.scene_name)
            return
        self.resets += 1
        self.obs = self.sim.reset(seed=self.seed + self.resets)
        self.teleop.grip = 0.0
        self.auto_running = False
        self.toast("Reset" + (" (randomized)" if self.sim.randomize else ""))

    # ------------------------------------------------------------------ recording
    def toggle_recording(self):
        if not self.sim.spec.recording and not self.recorder.active:
            self.toast("Recording is disabled in this scene (no .npz) - use a benchmark scene")
            return
        if self.recorder.active:
            self.stop_recording(save=True)
        else:
            src = self.auto.name if self.auto_running else "teleop"
            self.recorder.start(self.sim, source=src)
            self.toast("Recording started")

    def stop_recording(self, save=True, success=None):
        if not self.recorder.active:
            return
        n = self.recorder.length
        path = self.recorder.stop(save=save, success=success)
        if path:
            self.toast(f"Saved {n} steps -> {path.relative_to(path.parents[2])}")

    # ------------------------------------------------------------------ input
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if self.menu.open:
                if not self.menu.handle_event(e):
                    return False
                continue
            if self.picker.open:
                if not self.picker.handle_event(e):
                    return False
                continue
            if self.teleop.handle_event(e):
                if self.auto_running and e.type == pygame.KEYDOWN:
                    self.stop_auto("taken over")
                continue
            if e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    return False
                elif k == pygame.K_g:
                    self.show_labels = not self.show_labels
                    self.toast(f"Labels {'on' if self.show_labels else 'off'}  (default: settings.yaml show_labels)")
                elif k == pygame.K_l:
                    self.menu.show(current=self.scene_name)
                elif k == pygame.K_TAB:
                    self.switch_robot()
                elif k == pygame.K_r:
                    if self.robot_options():
                        self.picker.show(self.robot_options(), current=self.sim.robot_name)
                    else:
                        self.toast("This scene has no robot choice" + ("  (Tab switches)" if len(self.sim.arms) > 1 else ""))
                elif k in (pygame.K_h, pygame.K_F1):
                    self.show_help = not self.show_help
                elif k == pygame.K_BACKSPACE:
                    self.reset()
                elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    i = self.scenes.index(self.scene_name) if self.scene_name in self.scenes else 0
                    i = (i + (1 if k == pygame.K_RIGHTBRACKET else -1)) % len(self.scenes)
                    self.replay_path = None
                    self.load_scene(self.scenes[i])
                elif k == pygame.K_RETURN:
                    self.toggle_recording()
                elif k == pygame.K_p:
                    self.stop_auto() if self.auto_running else self.start_auto()
                elif k == pygame.K_o:
                    names = self.auto_names
                    cur = names.index(self.auto_name) if self.auto_name in names else -1
                    self.auto_name = names[(cur + 1) % len(names)]
                    self.replay_path = None
                    self.toast(f"Autopilot policy: {self.auto_name}  (P to run)")
                elif k == pygame.K_c:
                    self.cam_i = (self.cam_i + 1) % len(self.cam_modes)
                    self.toast(f"Camera: {self.cam_modes[self.cam_i]}")
                elif k == pygame.K_v:
                    self.show_pip = not self.show_pip
                elif k == pygame.K_t:
                    self.follow = not self.follow
                    self.toast(f"Camera follow {'on' if self.follow else 'off'}")
                elif k == pygame.K_k:
                    self.contacts = not self.contacts
                    self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self.contacts
                    self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.contacts
                elif k == pygame.K_n:
                    self.sim.randomize = not self.sim.randomize
                    self.toast(f"Randomize on reset: {'on' if self.sim.randomize else 'off'}")
                elif k == pygame.K_F2:
                    self.paused = not self.paused
            elif e.type == getattr(pygame, "CONTROLLERBUTTONDOWN", -1):
                if e.button == pygame.CONTROLLER_BUTTON_START:
                    self.toggle_recording()
                elif e.button == pygame.CONTROLLER_BUTTON_Y:
                    self.switch_robot()
                elif e.button == pygame.CONTROLLER_BUTTON_BACK:
                    self.reset()
            elif e.type == pygame.MOUSEMOTION and self.cam_i == 0:
                dx, dy = e.rel
                if e.buttons[0] and not (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    self.cam.azimuth -= dx * 0.3
                    self.cam.elevation = float(np.clip(self.cam.elevation - dy * 0.3, -89, 5))
                elif e.buttons[2] or (e.buttons[0] and pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    self.follow = False
                    a, el = math.radians(self.cam.azimuth), math.radians(self.cam.elevation)
                    right = np.array([math.sin(a), -math.cos(a), 0])
                    up = np.array([-math.cos(a) * math.sin(el), -math.sin(a) * math.sin(el), math.cos(el)])
                    s = self.cam.distance * 0.0015
                    self.cam.lookat[:] += (-dx * right + dy * up) * s
            elif e.type == pygame.MOUSEWHEEL and self.cam_i == 0:
                self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** e.y), 0.3, 12))
        return True

    # ------------------------------------------------------------------ simulation
    def _follow_point(self):
        o, R, _ = self.sim.base_pose()
        return o + R @ np.array([0.3, 0, 0.45])

    def tick_sim(self, frame_dt):
        if self.paused or self.menu.open or self.picker.open:
            return
        self.accum = min(self.accum + frame_dt, 0.1)
        while self.accum >= self.sim.control_dt:
            self.accum -= self.sim.control_dt
            if self.auto_running and self.teleop.is_active_input():
                self.stop_auto("taken over")
            if self.auto_running:
                cmd = self.auto.act(self.obs, self.sim)
                if self.auto.done:
                    self.stop_auto(f"finished: {self.auto.status}")
                    if self.recorder.active:
                        self.stop_recording(save=True, success=self.auto.success)
            else:
                cmd = self.teleop.act(self.obs, self.sim)
            task = self.auto.task_info() if self.auto_running else None
            self.recorder.add(self.obs, cmd.to_array(), task, self.sim)
            self.obs = self.sim.step(cmd)
        for m in list(self.sim.messages):
            self.toast(m)
        self.sim.messages.clear()

    # ------------------------------------------------------------------ rendering
    def _add_marker(self, scene, pos, size, rgba, gtype=mujoco.mjtGeom.mjGEOM_SPHERE):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.array([size, size, size]), np.asarray(pos, float),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        scene.ngeom += 1

    def render(self):
        mode = self.cam_modes[self.cam_i]
        if mode == "free":
            if self.follow:
                self.cam.lookat[:] += 0.08 * (self._follow_point() - self.cam.lookat)
            self.main_r.update_scene(self.sim.data, camera=self.cam, scene_option=self.opt)
        else:
            self.main_r.update_scene(self.sim.data, camera=mode, scene_option=self.opt)
        # ghost marker at the commanded TCP target
        if not self.sim.homing:
            self._add_marker(self.main_r.scene, self.sim.to_world(self.sim.ee_target_pos), 0.012, (1, 0.45, 0.1, 0.45))
        img = self.main_r.render()
        self.screen.blit(pygame.image.frombuffer(img.tobytes(), (self.W, self.H), "RGB"), (0, 0))
        if self.show_labels:
            self._draw_labels()
        cam = self.sim.wrist_camera
        if self.show_pip and cam in self.sim.cameras and mode != cam:
            self.pip_r.update_scene(self.sim.data, camera=cam)
            pip = pygame.image.frombuffer(self.pip_r.render().tobytes(), (320, 240), "RGB")
            x, y = self.W - 336, self.H - 256
            pygame.draw.rect(self.screen, (20, 22, 26), (x - 3, y - 3, 326, 246), border_radius=6)
            self.screen.blit(pip, (x, y))
            self._text("wrist camera", x + 8, y + 6, (230, 230, 230))
        self.draw_hud()
        if self.picker.open:
            self.picker.draw(self.screen, self.font, self.font_b, self.font_big)
        if self.menu.open:
            self.menu.update()
            self.menu.draw(self.screen, self.font, self.font_b, self.font_big)
        pygame.display.flip()

    def _project(self, p):
        """World point -> screen pixel for the main view (None if behind the camera)."""
        c = self.main_r.scene.camera[0]
        fwd, up = np.array(c.forward), np.array(c.up)
        v = np.asarray(p, float) - np.array(c.pos)
        z = v @ fwd
        if z <= c.frustum_near:
            return None
        x, y = (v @ np.cross(fwd, up)) * c.frustum_near / z, (v @ up) * c.frustum_near / z
        half_h = (c.frustum_top - c.frustum_bottom) / 2
        half_w = half_h * self.W / self.H
        sx = self.W / 2 + (x - c.frustum_center) / half_w * self.W / 2
        sy = self.H / 2 - (y - (c.frustum_top + c.frustum_bottom) / 2) / half_h * self.H / 2
        return int(sx), int(sy)

    def _draw_labels(self):
        for lb in self.sim.spec.labels:
            p = np.asarray(lb["pos"], float)
            if lb.get("on_robot"):
                o, R, _ = self.sim.base_pose()
                p = o + R @ p
            text, col = lb["text"], (240, 240, 240)
            if lb.get("mount") is not None:          # robot name labels: hide absent robots, mark the active one
                live = [a for a in self.sim.arms if a.mount_index == lb["mount"]]
                if not live:
                    continue
                if live[0] is self.sim.arm and len(self.sim.arms) > 1:
                    text, col = text + "  (active)", (120, 220, 235)
            s = self._project(p)
            if s is None:
                continue
            surf = self.font_b.render(text, True, col)
            w, h = surf.get_width() + 16, surf.get_height() + 8
            x, y = s[0] - w // 2, s[1] - h
            self._panel(x, y, w, h, 175)
            pygame.draw.line(self.screen, (120, 220, 235), (s[0], y + h), (s[0], y + h + 10), 2)
            self.screen.blit(surf, (x + 8, y + 4))

    def _text(self, s, x, y, color=(235, 235, 235), font=None):
        surf = (font or self.font).render(s, True, color)
        self.screen.blit(surf, (x, y))
        return surf.get_height()

    def _panel(self, x, y, w, h, alpha=170):
        p = pygame.Surface((w, h), pygame.SRCALPHA)
        p.fill((14, 16, 20, alpha))
        self.screen.blit(p, (x, y))

    def toast(self, msg):
        self.toasts.append((time.time(), msg))
        self.toasts = self.toasts[-5:]

    def draw_hud(self):
        s, o = self.sim, self.obs
        title = f"COBOT SANDBOX  -  {scene_label(self.sim.spec.raw)}"
        robot = f"robot  {s.arm.title}"
        hint = "  [Tab]" if len(s.arms) > 1 else ("  [R]" if self.robot_options() else "")
        self._panel(10, 10, max(380, self.font_b.size(title)[0] + 30, self.font.size(robot + hint)[0] + 30), 190)
        y = 18
        y += self._text(title, 20, y, (120, 220, 235), self.font_b) + 6
        if self.auto_running:
            mode = f"AUTO  {self.auto.status}"[:48]
        else:
            mode = "MANUAL" + (f"   [{self.teleop.pad_name}]" if self.teleop.pad else "")
        y += self._text(mode, 20, y, (255, 210, 120) if self.auto_running else (235, 235, 235)) + 3
        y += self._text(robot + hint, 20, y, (90, 210, 230)) + 3
        g = "CLOSED" if s.grip_cmd > 0.5 else "open"
        y += self._text(f"grip   {g:6s}  width {o['gripper_width'][0] * 100:4.1f} cm", 20, y) + 3
        p = o["tcp_pos"]
        y += self._text(f"tcp    x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f}", 20, y) + 3
        f = np.linalg.norm(o["wrist_force"])
        y += self._text(f"force  {f:5.1f} N    t {o['time'][0]:6.1f} s   {self.clock.get_fps():4.0f} fps", 20, y) + 3
        flags = []
        if not self.sim.spec.recording:
            flags.append("no recording")
        if self.sim.randomize:
            flags.append("randomize")
        if self.paused:
            flags.append("PAUSED")
        flags.append(f"cam: {self.cam_modes[self.cam_i]}")
        self._text("  ".join(flags) + "    H = help", 20, y, (160, 165, 175))
        if self.recorder.active:
            self._panel(self.W - 230, 10, 220, 34)
            if int(time.time() * 2) % 2 == 0:
                pygame.draw.circle(self.screen, (235, 50, 50), (self.W - 212, 27), 7)
            self._text(f"REC  {self.recorder.length} steps", self.W - 196, 19, (255, 120, 120), self.font_b)
        tasks = s.tasks.status(s.arm) if s.tasks else []
        if tasks:                                   # task checklist (top right)
            lines = [("[x] " if x["achieved"] else "[ ] ") + x["text"] for x in tasks]
            w = max(self.font.size(l)[0] for l in lines) + 40
            y0 = 54 if self.recorder.active else 10
            self._panel(self.W - w - 10, y0, w, 36 + 20 * len(lines))
            done = sum(x["achieved"] for x in tasks)
            self._text(f"Tasks  {done}/{len(tasks)}", self.W - w + 6, y0 + 8,
                       (80, 235, 110) if done == len(tasks) else (120, 220, 235), self.font_b)
            for i, (l, x) in enumerate(zip(lines, tasks)):
                self._text(l, self.W - w + 6, y0 + 32 + 20 * i, (120, 235, 140) if x["achieved"] else (225, 225, 225))
        now = time.time()
        self.toasts = [(t, m) for t, m in self.toasts if now - t < 4.0]
        for i, (t, m) in enumerate(reversed(self.toasts)):
            surf = self.font.render(m, True, (240, 240, 240))
            w = surf.get_width() + 20
            yy = self.H - 40 - i * 30
            self._panel(16, yy, w, 26, int(170 * min(1, 4.0 - (now - t))))
            self.screen.blit(surf, (26, yy + 5))
        if self.show_help:
            w, h = 840, 30 + 24 * len(HELP)
            x0, y0 = (self.W - w) // 2, (self.H - h) // 2
            self._panel(x0, y0, w, h, 215)
            for i, (a, b) in enumerate(HELP):
                self._text(a, x0 + 20, y0 + 16 + 24 * i, (120, 220, 235), self.font_b)
                self._text(b, x0 + 140, y0 + 18 + 24 * i)

    # ------------------------------------------------------------------ main loop
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            running = self.handle_events()
            self.tick_sim(dt)
            self.render()
        self.stop_recording(save=True)
        self.sim.close()
        pygame.quit()
