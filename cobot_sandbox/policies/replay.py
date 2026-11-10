from __future__ import annotations

from ..command import Command
from ..recorder import load_episode
from .base import Policy


class Replay(Policy):
    """Open-loop playback of a recorded episode's actions (deterministic given the same seed)."""
    name = "replay"

    def __init__(self, path):
        super().__init__()
        ep = load_episode(path)
        self.actions, self.meta = ep["action"], ep["meta"]

    def reset(self, sim):
        super().reset(sim)
        self.i = 0

    def act(self, obs, sim):
        if self.i >= len(self.actions):
            self.done, self.success, self.status = True, self.meta.get("success"), "replay finished"
            return Command()
        a = self.actions[self.i]
        self.i += 1
        self.status = f"replay {self.i}/{len(self.actions)}"
        return Command.from_array(a)
