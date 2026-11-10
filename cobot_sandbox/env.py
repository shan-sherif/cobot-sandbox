"""Gymnasium-style headless environment (no gymnasium dependency required).

    env = CobotEnv("lift", robot="franka_fr3", randomize=True, cameras=["wrist_cam"])
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space_sample())

Actions are the 8-D command (command.ACTION_LABELS): TCP linear / angular velocity normalized to
[-1, 1] and scaled by the robot's command limits, gripper in [0, 1], home > 0.5.
Default reward: 1.0 and terminated once all of the scene's tasks are achieved (sparse success, as in
robosuite); pass reward_fn(sim, obs) -> (reward, terminated) for your own shaping. If gymnasium is
installed, `CobotEnv.gym_spaces()` returns matching Box spaces.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .command import ACTION_DIM, GRIPPER, HOME, Command
from .sim import Sim


class CobotEnv:
    def __init__(self, scene="lift", randomize=True, cameras: Sequence[str] = (), image_size=(128, 128),
                 max_steps=1500, reward_fn: Optional[Callable] = None, seed=0, **sim_kw):
        self.sim = Sim(scene, seed=seed, randomize=randomize, **sim_kw)
        self.cameras, self.image_size = list(cameras), tuple(image_size)
        self.max_steps, self.reward_fn = max_steps, reward_fn
        L = self.sim.lim
        self.scale = np.array([L["ee_lin"]] * 3 + [L["ee_ang"]] * 3 + [1.0, 1.0])
        self._rng = np.random.default_rng(seed)

    def _obs(self, obs):
        for c in self.cameras:
            obs[f"img.{c}"] = self.sim.render(c, *self.image_size)
        return obs

    def reset(self, seed: Optional[int] = None, options=None):
        obs = self.sim.reset(seed=seed)
        self.t = 0
        return self._obs(obs), {"seed": self.sim.last_seed}

    def denormalize(self, action) -> Command:
        a = np.clip(np.asarray(action, float), -1, 1) * self.scale
        a[GRIPPER] = float(np.clip(action[GRIPPER], 0, 1))
        return Command.from_array(a)

    def step(self, action):
        cmd = action if isinstance(action, Command) else self.denormalize(action)
        obs = self.sim.step(cmd)
        self.t += 1
        if self.reward_fn:
            reward, terminated = self.reward_fn(self.sim, obs)
        else:
            reward = terminated = float(self.sim.success)
        truncated = self.t >= self.max_steps
        return self._obs(obs), float(reward), bool(terminated), truncated, {"success": self.sim.success, "robot": self.sim.robot_name}

    def action_space_sample(self):
        a = self._rng.uniform(-1, 1, ACTION_DIM)
        a[GRIPPER], a[HOME] = self._rng.uniform(0, 1), 0.0
        return a

    def gym_spaces(self):
        import gymnasium as gym
        lo = -np.ones(ACTION_DIM, np.float32)
        lo[GRIPPER] = lo[HOME] = 0
        act = gym.spaces.Box(lo, np.ones(ACTION_DIM, np.float32))
        obs = self.sim.get_obs()
        spaces = {k: gym.spaces.Box(-np.inf, np.inf, np.shape(v), np.float32) for k, v in obs.items()}
        for c in self.cameras:
            spaces[f"img.{c}"] = gym.spaces.Box(0, 255, (self.image_size[1], self.image_size[0], 3), np.uint8)
        return gym.spaces.Dict(spaces), act

    def close(self):
        self.sim.close()
