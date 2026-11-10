"""The single action interface shared by keyboard/gamepad teleop, scripts and learned policies.

Because every controller emits the same `Command`, demonstrations recorded from a human are
directly usable as training targets for a policy that is later run in the same loop.

As an array (recordings, CobotEnv, learned policies) a Command is 8 numbers, see ACTION_LABELS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTION_LABELS = ["ee_vx", "ee_vy", "ee_vz", "ee_wx", "ee_wy", "ee_wz", "gripper", "home"]
ACTION_DIM = len(ACTION_LABELS)
GRIPPER, HOME = 6, 7          # indices of the non-velocity entries


def _z3():
    return np.zeros(3)


@dataclass
class Command:
    ee_lin: np.ndarray = field(default_factory=_z3)    # TCP linear velocity, robot base frame (m/s)
    ee_ang: np.ndarray = field(default_factory=_z3)    # TCP angular velocity, robot base frame (rad/s)
    gripper: float = 0.0                               # 0 = open ... 1 = closed (absolute)
    home: bool = False                                 # rising edge: move arm to home posture

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.ee_lin, self.ee_ang, [self.gripper],
                               [1.0 if self.home else 0.0]]).astype(np.float32)

    @classmethod
    def from_array(cls, a) -> "Command":
        a = np.asarray(a, float)
        if len(a) != ACTION_DIM:
            raise ValueError(f"expected an action of length {ACTION_DIM} ({', '.join(ACTION_LABELS)}), got {len(a)}")
        return cls(ee_lin=a[0:3].copy(), ee_ang=a[3:6].copy(), gripper=float(a[GRIPPER]), home=bool(a[HOME] > 0.5))
