"""Core simulation: scene loading, robot control stack, observations, tasks, randomization.

Control stack (per control tick, default 50 Hz), the same for every robot (FR3, UR5e):
    Command (TCP twist in the robot's base frame, gripper 0..1, home)
      -> TCP target pose (base frame = the robot's mount)
      -> differential IK -> joint position targets (rate-limited to max joint velocity)
      -> joint servo at physics rate (500 Hz): PD + gravity/Coriolis compensation -> torques
This mirrors how a real cobot is commanded (Cartesian velocity / servoJ over a torque loop), so
controllers and data written against `Sim` map cleanly onto hardware later.

A scene may hold several live robots (`robots: {select: all}`, the test bench). Commands go to the
*active* robot (`set_active` / `next_robot`); the others hold their pose and gripper.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import mujoco
import numpy as np

from . import robots as R
from .command import Command
from .ik import DiffIK
from .mathutil import clip_norm, mat_to_quat, quat_conj, quat_from_rotvec, quat_mul, quat_yaw
from .scene_builder import SceneSpec, build_model, build_xml, compile_scene, place_objects
from .tasks import TaskTracker


class Arm:
    """One robot: its model indices, controller state and observations."""

    def __init__(self, sim: "Sim", mount: dict):
        m, J, A = sim.model, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR
        self.sim, self.mount = sim, mount
        self.name, self.prefix, self.mount_index = mount["robot"], mount["prefix"], mount["index"]
        self.cfg = c = R.load_cfg(self.name)
        self.title = c.get("title", self.name)
        P = self.prefix
        idx = lambda kind, n: sim._id(kind, P + n)                     # noqa: E731
        arm_j = [idx(J, n) for n in c["arm_joints"]]
        self.arm_q = np.array([m.jnt_qposadr[j] for j in arm_j])
        self.arm_d = np.array([m.jnt_dofadr[j] for j in arm_j])
        self.arm_act = np.array([idx(A, n) for n in c["arm_actuators"]])
        self.kp, self.kd = np.array(c["kp"], float), np.array(c["kd"], float)
        self.tau_lim = np.array(c["torque_limits"], float)
        self.qd_max = np.array(c["max_joint_vel"], float)
        self.err_max = np.array(c.get("max_following_error", [np.inf] * len(arm_j)), float)
        self.tcp_lag = float(c.get("max_tcp_lag", np.inf))
        self.home = np.array(c["home"], float)
        self.lim = c["limits"]
        self.reach = tuple(c.get("reach", [0.28, 0.8]))
        self.base_body = sim._id(mujoco.mjtObj.mjOBJ_BODY, f"mount_{self.mount_index}")
        self.tcp = idx(mujoco.mjtObj.mjOBJ_SITE, c["tcp"].get("site", "tcp"))
        self.camera = P + (c.get("wrist_camera") or {}).get("name", "wrist_cam")
        g = c["gripper"]
        self.grip_act = idx(A, g["actuator"])
        self.grip_open, self.grip_closed = float(g["open"]), float(g["closed"])
        w = c["width"]
        if "joints" in w:
            self.width_jnt = [idx(J, n) for n in w["joints"]]
            self.width_q = np.array([m.jnt_qposadr[j] for j in self.width_jnt])
            self.width_scale, self.width_bodies = float(w.get("scale", 1.0)), None
        else:
            self.width_jnt, self.width_bodies = [], [idx(mujoco.mjtObj.mjOBJ_BODY, n) for n in w["bodies"]]
            self.width_offset = float(w.get("offset", 0.0))
        self.finger_bodies = [set(idx(mujoco.mjtObj.mjOBJ_BODY, n) for n in side) for side in c["finger_bodies"]]
        self.sensor_force, self.sensor_torque = P + "wrist_force", P + "wrist_torque"
        ik = c.get("ik", {})
        self.ik = DiffIK(m, self.arm_q, self.arm_d, self.tcp, self.home, damping=ik.get("damping", 0.05),
                         null_gain=ik.get("null_gain", 0.05), iters=ik.get("iters", 3),
                         rot_weight=ik.get("rot_weight", 0.5))
        self.q_des = self.home.copy()
        self.grip_cmd, self.homing = 0.0, False
        self.ee_target_pos, self.ee_target_quat = np.zeros(3), np.array([1.0, 0, 0, 0])

    # ------------------------------------------------------------------ frames
    def base_pose(self):
        """(origin xyz, rotation matrix, quat) of the robot's mount (its base frame) in world frame."""
        d, b = self.sim.data, self.base_body
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy(), d.xquat[b].copy()

    def tcp_pose(self, frame="base"):
        d = self.sim.data
        p, q = d.site_xpos[self.tcp].copy(), mat_to_quat(d.site_xmat[self.tcp])
        if frame == "world":
            return p, q
        o, Rm, bq = self.base_pose()
        return Rm.T @ (p - o), quat_mul(quat_conj(bq), q)

    def sync_target(self):
        p, q = self.ik.fk(self.sim.data.qpos, self.q_des)
        o, Rm, bq = self.base_pose()
        self.ee_target_pos = Rm.T @ (p - o)
        self.ee_target_quat = quat_mul(quat_conj(bq), q)

    # ------------------------------------------------------------------ control
    def reset(self):
        d, m = self.sim.data, self.sim.model
        d.qpos[self.arm_q] = self.home
        for j in self.width_jnt:                     # start with the gripper open
            d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j, 1] if self.grip_open > self.grip_closed else m.jnt_range[j, 0]
        self.q_des = self.home.copy()
        self.grip_cmd, self.homing = 0.0, False
        d.ctrl[self.grip_act] = self.grip_open

    def hold(self) -> Command:
        return Command(gripper=self.grip_cmd)

    def command(self, cmd: Command):
        """Control-tick update: Command -> TCP target -> IK -> rate-limited joint targets; gripper."""
        d, L, dt = self.sim.data, self.lim, self.sim.control_dt
        if cmd.home:
            self.homing = True
        if self.homing:
            q_goal = self.home
        else:
            self.ee_target_pos = self.ee_target_pos + clip_norm(np.asarray(cmd.ee_lin, float), L["ee_lin"]) * dt
            w = clip_norm(np.asarray(cmd.ee_ang, float), L["ee_ang"])
            self.ee_target_quat = quat_mul(quat_from_rotvec(w * dt), self.ee_target_quat)
            self.ee_target_quat /= np.linalg.norm(self.ee_target_quat)
            o, Rm, bq = self.base_pose()
            tcp_b = Rm.T @ (d.site_xpos[self.tcp] - o)
            lag = np.linalg.norm(self.ee_target_pos - tcp_b)
            if lag > self.tcp_lag:          # blocked: don't let the target run away from the real arm
                self.ee_target_pos = tcp_b + (self.ee_target_pos - tcp_b) * (self.tcp_lag / lag)
            q_goal, p_ach, q_ach, perr, rerr = self.ik.solve(
                d.qpos, self.q_des, o + Rm @ self.ee_target_pos, quat_mul(bq, self.ee_target_quat))
            if perr > 0.02:                 # keep the target from running away when it is unreachable
                p_ach_b = Rm.T @ (p_ach - o)
                self.ee_target_pos = p_ach_b + (self.ee_target_pos - p_ach_b) * (0.02 / perr)
            if rerr > 0.15:
                self.ee_target_quat = quat_mul(quat_conj(bq), q_ach)
        step = np.clip(q_goal - self.q_des, -self.qd_max * dt, self.qd_max * dt)
        self.q_des = self.q_des + step
        if self.homing:
            if np.max(np.abs(self.home - self.q_des)) < 1e-4:
                self.homing = False
            self.sync_target()
        self.grip_cmd = float(np.clip(cmd.gripper, 0.0, 1.0))
        d.ctrl[self.grip_act] = self.grip_open + (self.grip_closed - self.grip_open) * self.grip_cmd

    def servo(self):
        """Physics-rate joint servo: PD + gravity/Coriolis compensation, following error clipped."""
        d = self.sim.data
        q, qd = d.qpos[self.arm_q], d.qvel[self.arm_d]
        e = np.clip(self.q_des - q, -self.err_max, self.err_max)
        tau = self.kp * e - self.kd * qd + d.qfrc_bias[self.arm_d]
        d.ctrl[self.arm_act] = np.clip(tau, -self.tau_lim, self.tau_lim)

    # ------------------------------------------------------------------ observations
    def gripper_width(self) -> float:
        d = self.sim.data
        if self.width_bodies is None:
            return self.width_scale * float(np.sum(d.qpos[self.width_q]))
        a, b = (d.xpos[i] for i in self.width_bodies)
        return float(np.linalg.norm(a - b)) + self.width_offset

    def finger_forces(self) -> np.ndarray:
        """Total contact normal force (N) on each finger side."""
        m, d = self.sim.model, self.sim.data
        out, f6 = np.zeros(2), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for side, bodies in enumerate(self.finger_bodies):
                if (b1 in bodies) != (b2 in bodies):
                    mujoco.mj_contactForce(m, d, i, f6)
                    out[side] += abs(f6[0])
        return out

    def obs(self) -> dict:
        d, sim = self.sim.data, self.sim
        q, qd = d.qpos[self.arm_q].copy(), d.qvel[self.arm_d].copy()
        if sim.obs_noise:
            q += sim.rng.normal(0, sim.obs_noise, q.shape)
            qd += sim.rng.normal(0, sim.obs_noise * 10, qd.shape)
        tcp_b, tcpq_b = self.tcp_pose("base")
        tcp_w, tcpq_w = self.tcp_pose("world")
        o, _, bq = self.base_pose()
        return {
            "arm_qpos": q, "arm_qvel": qd, "arm_qdes": self.q_des.copy(), "arm_tau": d.ctrl[self.arm_act].copy(),
            "gripper_width": np.array([self.gripper_width()]), "gripper_cmd": np.array([self.grip_cmd]),
            "tcp_pos": tcp_b, "tcp_quat": tcpq_b, "tcp_pos_world": tcp_w, "tcp_quat_world": tcpq_w,
            "ee_target_pos": self.ee_target_pos.copy(), "ee_target_quat": self.ee_target_quat.copy(),
            "base_pose": np.array([o[0], o[1], quat_yaw(bq)]),
            "wrist_force": d.sensor(self.sensor_force).data.copy(),
            "wrist_torque": d.sensor(self.sensor_torque).data.copy(),
            "finger_touch": self.finger_forces(),
        }

    def get_state(self) -> dict:
        return {"q_des": self.q_des.copy(), "ee_target_pos": self.ee_target_pos.copy(),
                "ee_target_quat": self.ee_target_quat.copy(), "grip_cmd": np.array([self.grip_cmd]),
                "homing": np.array([float(self.homing)])}

    def set_state(self, st: dict):
        self.q_des = np.array(st["q_des"], float)
        self.ee_target_pos = np.array(st["ee_target_pos"], float)
        self.ee_target_quat = np.array(st["ee_target_quat"], float)
        self.grip_cmd = float(np.ravel(st["grip_cmd"])[0])
        self.homing = bool(np.ravel(st["homing"])[0] > 0.5)


def _active(attr, doc=""):
    """Sim attribute that reads/writes the active robot's attribute (keeps the single-robot API)."""
    return property(lambda self: getattr(self.arm, attr), lambda self, v: setattr(self.arm, attr, v), doc=doc)


class Sim:
    def __init__(self, scene="lift", seed: int = 0, control_hz: float = 50.0, randomize: bool = False,
                 robot: Optional[str] = None, obs_noise: float = 0.0, action_latency: int = 0):
        """
        scene:          scene name (cobot_sandbox/scenes/NN_<name>.yaml), a path, or a dict
        robot:          which robot to use in a `select: one` scene (default: the first available)
        randomize:      apply the scene's `randomize:` block on every reset
        obs_noise:      std-dev of Gaussian noise on joint position/velocity observations
        action_latency: commands take effect this many control ticks late (sim-to-real realism)
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.randomize, self.obs_noise, self.action_latency = randomize, obs_noise, int(action_latency)
        self.spec: SceneSpec = compile_scene(scene, seed=seed)
        self.mounts = R.resolve(self.spec.mounts, self.spec.robot_select, robot)
        self.xml = build_xml(self.spec)
        self.model = build_model(self.spec, self.xml, self.mounts)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.n_sub = max(1, int(round(1.0 / (control_hz * self.dt))))
        self.control_dt = self.n_sub * self.dt
        self.arms: List[Arm] = [Arm(self, mt) for mt in self.mounts]
        self.active = 0
        self._index_objects()
        self._renderers: Dict[tuple, mujoco.Renderer] = {}
        self.messages: deque = deque(maxlen=6)
        self.tasks: Optional[TaskTracker] = None           # created in reset()
        self.reset(seed=seed)

    # ------------------------------------------------------------------ robots
    @property
    def arm(self) -> Arm:
        return self.arms[self.active]

    @property
    def robot_name(self) -> str:
        return self.arm.name

    ee_target_pos = _active("ee_target_pos")
    ee_target_quat = _active("ee_target_quat")
    grip_cmd = _active("grip_cmd")
    homing = _active("homing")
    lim = property(lambda self: self.arm.lim)
    wrist_camera = property(lambda self: self.arm.camera)

    def set_active(self, i: int):
        self.active = int(i) % len(self.arms)
        self.messages.append(f"Controlling {self.arm.title}")

    def next_robot(self):
        self.set_active(self.active + 1)

    # ------------------------------------------------------------------ setup
    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            raise KeyError(f"{kind} '{name}' not found")
        return i

    def _index_objects(self):
        m = self.model
        self.objects = {}
        for o in self.spec.objects:
            b = self._id(mujoco.mjtObj.mjOBJ_BODY, o.name)
            gid = self._id(mujoco.mjtObj.mjOBJ_GEOM, o.name)
            self.objects[o.name] = dict(spec=o, body=b, geom=gid,
                                        qadr=m.jnt_qposadr[m.body_jntadr[b]], dadr=m.jnt_dofadr[m.body_jntadr[b]])
        self.cameras = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
        # nominal values for domain randomization
        self._nom = dict(friction=m.geom_friction.copy(), mass=m.body_mass.copy(),
                         inertia=m.body_inertia.copy(), mat_rgba=m.mat_rgba.copy(),
                         light=m.light_diffuse.copy())

    # ------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None, randomize: Optional[bool] = None) -> dict:
        if seed is None:                       # always reseed so every episode is reproducible
            seed = int(self.rng.integers(2 ** 31))
        self.rng = np.random.default_rng(seed)
        rnd = self.randomize if randomize is None else randomize
        self.last_seed, self.last_randomize = int(seed), bool(rnd)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._apply_randomization(rnd)
        for a in self.arms:
            a.reset()
        r = self.spec.randomize if rnd else {}
        poses = place_objects(self.spec, self.rng, randomize=rnd, xy_jitter=float(r.get("object_xy", 0.0)),
                              random_yaw=bool(r.get("object_yaw", False)))
        for name, (pos, yaw) in poses.items():
            o = self.objects[name]
            d.qpos[o["qadr"]:o["qadr"] + 3] = pos
            d.qpos[o["qadr"] + 3:o["qadr"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        for a in self.arms:
            a.sync_target()
        self._cmd_queue = deque([Command() for _ in range(self.action_latency)])
        self.tasks = None
        for _ in range(int(0.3 / self.control_dt)):      # let objects and grippers settle
            self._physics(self.arm.hold())
        self.t0 = d.time
        self.tasks = TaskTracker(self)
        return self.get_obs()

    def _apply_randomization(self, on: bool):
        m, n = self.model, self._nom
        m.geom_friction[:] = n["friction"]
        m.body_mass[:] = n["mass"]
        m.body_inertia[:] = n["inertia"]
        m.mat_rgba[:] = n["mat_rgba"]
        m.light_diffuse[:] = n["light"]
        if not on:
            return
        r, rng = self.spec.randomize, self.rng
        for o in self.objects.values():
            if "friction_scale" in r:
                m.geom_friction[o["geom"], 0] *= rng.uniform(*r["friction_scale"])
            if "mass_scale" in r:
                s = rng.uniform(*r["mass_scale"])
                m.body_mass[o["body"]] *= s
                m.body_inertia[o["body"]] *= s
            if r.get("object_colors"):
                mid = m.geom_matid[o["geom"]]
                if mid >= 0:
                    m.mat_rgba[mid, :3] = rng.uniform(0.1, 1.0, 3)
        if r.get("light_jitter"):
            j = float(r["light_jitter"])
            m.light_diffuse[:] *= rng.uniform(1 - j, 1 + j, (m.nlight, 1))

    # ------------------------------------------------------------------ full state (for exact replay)
    _STATE = mujoco.mjtState.mjSTATE_INTEGRATION

    def get_state(self) -> dict:
        n = mujoco.mj_stateSize(self.model, self._STATE)
        phys = np.empty(n)
        mujoco.mj_getState(self.model, self.data, phys, self._STATE)
        st = {"physics": phys, "active": np.array([self.active])}
        for i, a in enumerate(self.arms):
            st.update({f"arm{i}.{k}": v for k, v in a.get_state().items()})
        return st

    def set_state(self, st: dict):
        mujoco.mj_setState(self.model, self.data, np.asarray(st["physics"], float), self._STATE)
        self.active = int(np.ravel(st.get("active", [0]))[0])
        for i, a in enumerate(self.arms):
            a.set_state({k.split(".", 1)[1]: v for k, v in st.items() if k.startswith(f"arm{i}.")})
        self._cmd_queue = deque([Command() for _ in range(self.action_latency)])
        mujoco.mj_forward(self.model, self.data)
        self.t0 = self.data.time

    # ------------------------------------------------------------------ frames (active robot)
    def base_pose(self):
        return self.arm.base_pose()

    def to_base(self, p_w):
        o, Rm, _ = self.base_pose()
        return Rm.T @ (np.asarray(p_w) - o)

    def to_world(self, p_b):
        o, Rm, _ = self.base_pose()
        return o + Rm @ np.asarray(p_b)

    def tcp_pose(self, frame="base"):
        return self.arm.tcp_pose(frame)

    # ------------------------------------------------------------------ stepping
    def step(self, cmd: Optional[Command] = None) -> dict:
        # Round to the recorded precision (float32) so what runs is exactly what a recording stores.
        cmd = Command.from_array((cmd or Command()).to_array())
        if self.action_latency:
            self._cmd_queue.append(cmd)
            cmd = self._cmd_queue.popleft()
        self._physics(cmd)
        # MuJoCo auto-resets mjData on NaN/huge qpos, qvel or qacc; restore a consistent sandbox state.
        ws = [self.data.warning[k] for k in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
                                             mujoco.mjtWarning.mjWARN_BADQPOS)]
        if any(w.number for w in ws):
            self.unstable_count = getattr(self, "unstable_count", 0) + 1
            self.messages.append("Physics instability detected - scene reset")
            for w in ws:
                w.number = 0
            return self.reset(seed=self.last_seed)
        return self.get_obs()

    def _physics(self, cmd: Command):
        # The active robot follows cmd. The others keep their joint targets and gripper exactly
        # (no IK update, so a redundant arm's elbow doesn't drift while it waits).
        self.arm.command(cmd)
        for _ in range(self.n_sub):
            for a in self.arms:
                a.servo()
            mujoco.mj_step(self.model, self.data)
        # mj_step leaves positions/sensors describing the pre-integration state; refresh them so
        # observations are exactly the current state (and identical after set_state()).
        mujoco.mj_forward(self.model, self.data)
        if self.tasks is not None:
            self.tasks.update()

    # ------------------------------------------------------------------ observations
    def get_obs(self) -> dict:
        obs = {"time": np.array([self.data.time - getattr(self, "t0", 0.0)])}
        obs.update(self.arm.obs())
        o, Rm, _ = self.base_pose()
        for name, ob in self.objects.items():
            pw = self.data.xpos[ob["body"]].copy()
            obs[f"obj.{name}.pos"] = pw
            obs[f"obj.{name}.quat"] = self.data.xquat[ob["body"]].copy()
            obs[f"obj.{name}.pos_base"] = Rm.T @ (pw - o)
        return obs

    def object_pose(self, name, frame="world"):
        ob = self.objects[name]
        p, q = self.data.xpos[ob["body"]].copy(), self.data.xquat[ob["body"]].copy()
        if frame == "base":
            o, Rm, bq = self.base_pose()
            return Rm.T @ (p - o), quat_mul(quat_conj(bq), q)
        return p, q

    @property
    def success(self) -> bool:
        """All of the active robot's scene tasks have been achieved (see tasks.TaskTracker)."""
        return self.tasks.success(self.arm)

    # ------------------------------------------------------------------ rendering
    def renderer(self, width, height) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def render(self, camera="overview", width=320, height=240, scene_option=None) -> np.ndarray:
        """camera "wrist_cam" means the active robot's wrist camera."""
        if camera == "wrist_cam":
            camera = self.wrist_camera
        r = self.renderer(width, height)
        if scene_option is None:
            r.update_scene(self.data, camera=camera)
        else:
            r.update_scene(self.data, camera=camera, scene_option=scene_option)
        return r.render()

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
