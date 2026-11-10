from __future__ import annotations

from typing import Dict

import numpy as np

from ..command import Command


class Policy:
    """Anything that maps (obs, sim) -> Command: teleop, scripts, learned models, replays."""
    name = "policy"

    def __init__(self):
        self.done = False
        self.success = None      # True / False once known
        self.status = ""         # one-line HUD text

    def reset(self, sim) -> None:
        self.done, self.success, self.status = False, None, ""

    def act(self, obs: dict, sim) -> Command:
        return Command()

    def task_info(self) -> Dict[str, np.ndarray]:
        """Extra per-step arrays saved alongside observations (e.g. the current goal)."""
        return {}


class Sequence(Policy):
    """Run sub-policies one after another. Stops at the first failure of a required step; steps with
    `optional = True` may fail without aborting (overall success = every required step succeeded)."""
    name = "sequence"

    def __init__(self, policies):
        super().__init__()
        self.policies = list(policies)

    def reset(self, sim):
        super().reset(sim)
        self.i = 0
        self.policies[0].reset(sim)

    def act(self, obs, sim):
        p = self.policies[self.i]
        cmd = p.act(obs, sim)
        self.status = f"[{self.i + 1}/{len(self.policies)}] {p.status}"
        if p.done:
            if p.success is False and not getattr(p, "optional", False):
                self.done, self.success = True, False
            elif self.i + 1 < len(self.policies):
                self.i += 1
                self.policies[self.i].reset(sim)
            else:
                self.done, self.success = True, True
        return cmd

    def task_info(self):
        return self.policies[self.i].task_info()
