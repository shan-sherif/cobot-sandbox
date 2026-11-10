"""Damped-least-squares differential IK with a null-space pull toward a home posture.

Runs on its own MjData scratch copy so it never disturbs the simulation state.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .mathutil import mat_to_quat, quat_err


class DiffIK:
    def __init__(self, model, qadr, dadr, site_id, home, damping=0.05, null_gain=0.05,
                 iters=3, rot_weight=0.5, max_step=0.2):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr, self.dadr = np.asarray(qadr), np.asarray(dadr)
        self.sid = site_id
        self.home = np.asarray(home, float)
        jnt = [int(np.where(model.jnt_qposadr == a)[0][0]) for a in self.qadr]
        self.lo = model.jnt_range[jnt, 0].copy()
        self.hi = model.jnt_range[jnt, 1].copy()
        self.lam2 = damping ** 2
        self.null_gain, self.iters, self.w, self.max_step = null_gain, iters, rot_weight, max_step
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        n = len(self.qadr)
        self.eye6, self.eyen = np.eye(6), np.eye(n)

    def fk(self, qpos_full, q_arm):
        d = self.d
        d.qpos[:] = qpos_full
        d.qpos[self.qadr] = q_arm
        mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.sid].copy(), mat_to_quat(d.site_xmat[self.sid])

    def solve(self, qpos_full, q_init, pos_w, quat_w):
        """Returns (q, achieved_pos_w, achieved_quat_w, pos_err, rot_err)."""
        m, d = self.m, self.d
        q = np.array(q_init, float)
        d.qpos[:] = qpos_full
        for _ in range(self.iters + 1):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.sid]
            cq = mat_to_quat(d.site_xmat[self.sid])
            ep, er = pos_w - p, quat_err(quat_w, cq)
            if _ == self.iters:
                break
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.sid)
            J = np.vstack([self.jacp[:, self.dadr], self.w * self.jacr[:, self.dadr]])
            err = np.concatenate([ep, self.w * er])
            A = J @ J.T + self.lam2 * self.eye6
            dq = J.T @ np.linalg.solve(A, err)
            Jpinv = J.T @ np.linalg.inv(A)
            dq += (self.eyen - Jpinv @ J) @ (self.null_gain * (self.home - q))
            dq = np.clip(dq, -self.max_step, self.max_step)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, p.copy(), cq, float(np.linalg.norm(ep)), float(np.linalg.norm(er))
