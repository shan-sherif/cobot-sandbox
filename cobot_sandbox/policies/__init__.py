from .base import Policy, Sequence
from .scripted import ClearTable, PickPlace, Push, TaskPolicy

SCRIPTED = ["task", "pick_place", "clear_table"]


def make_policy(name: str, sim, **kw) -> Policy:
    """Factory used by run.py / collect.py. 'teleop' and 'replay' are built by their callers.
    "task" solves the scene's own tasks (lift / place / stack / push / insert)."""
    if name == "task":
        return TaskPolicy(sim)
    if name == "pick_place":
        return PickPlace(**kw)
    if name == "clear_table":
        return ClearTable(**kw)
    raise ValueError(f"unknown policy '{name}'")
