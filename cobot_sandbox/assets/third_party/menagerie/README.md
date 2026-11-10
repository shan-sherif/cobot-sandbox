# Third-party robot models (MuJoCo Menagerie subset)

Source: https://github.com/google-deepmind/mujoco_menagerie.git
Commit: 98a70c569c4c1b6da467a20012f766ee1f8dc521

These folders are an **unmodified subset** of MuJoCo Menagerie: only the model and mesh files this
sandbox loads are included; other files (other variants, previews, demo scenes) were omitted.
Each folder keeps its own `LICENSE` (the license that applies to that folder) and `README.md`
(origin and credits). This sandbox's MIT license does not apply to these files.

| Folder | Used for | License |
|---|---|---|
| `franka_fr3/` | Franka Research 3 arm | Apache-2.0 |
| `franka_emika_panda/` | Franka Hand (`hand.xml`) used on the FR3 | Apache-2.0 |
| `universal_robots_ur5e/` | UR5e arm | BSD-3-Clause |
| `robotiq_2f85/` | Robotiq 2F-85 gripper used on the UR5e | BSD-2-Clause |

The models are adapted in memory at load time by `sandbox/robots.py` (gripper attached, position
servos replaced by torque motors, sites/camera/sensors added); the files themselves are unchanged.
Re-create this folder with `python scripts/fetch_models.py`.
