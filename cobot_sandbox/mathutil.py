"""Small quaternion / frame helpers (MuJoCo convention: quaternions are [w, x, y, z])."""
from __future__ import annotations

import math

import mujoco
import numpy as np


def quat_mul(a, b):
    r = np.zeros(4)
    mujoco.mju_mulQuat(r, np.asarray(a, float), np.asarray(b, float))
    return r


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_from_rotvec(v):
    v = np.asarray(v, float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.array([1.0, 0, 0, 0])
    r = np.zeros(4)
    mujoco.mju_axisAngle2Quat(r, v / ang, ang)
    return r


def quat_to_rotvec(q):
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    r = np.zeros(3)
    mujoco.mju_quat2Vel(r, q, 1.0)
    return r


def quat_err(target, current):
    """Rotation vector e such that target = exp(e) * current (fixed-frame error)."""
    return quat_to_rotvec(quat_mul(target, quat_conj(current)))


def mat_to_quat(m):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(m, float).reshape(9))
    return q


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))




def tool_down_quat(theta):
    """Gripper pointing straight down (tool z = -z), rotated by theta about the vertical.
    The finger closing axis (tool y) is then (sin theta, -cos theta, 0)."""
    return quat_mul(yaw_quat(theta), np.array([0.0, 1.0, 0.0, 0.0]))


def tool_yaw(q):
    """Inverse of tool_down_quat for a (roughly) downward-pointing tool."""
    y = quat_to_mat(q)[:, 1]
    return math.atan2(y[0], -y[1])


def clip_norm(v, limit):
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit else v
