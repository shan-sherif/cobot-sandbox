# Third-party notices

This project's own code is licensed under the MIT License (see `LICENSE`). It also includes the
third-party files below, which remain under their own licenses. The MIT License does not apply
to them.

## MuJoCo Menagerie robot models

`cobot_sandbox/assets/third_party/menagerie/` contains an unmodified subset of
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (commit
`98a70c569c4c1b6da467a20012f766ee1f8dc521`): the model and mesh files this sandbox loads, with
each folder's original `LICENSE`, `README.md` and `CHANGELOG.md`. Unused files were omitted.

| Folder | Model | Derived from | License |
|---|---|---|---|
| `franka_fr3/` | Franka Research 3 | Franka Robotics' public `franka_description` URDF | Apache-2.0 (`franka_fr3/LICENSE`) |
| `franka_emika_panda/` | Franka Hand (`hand.xml` only) | Franka Robotics' public `franka_ros` description | Apache-2.0 (`franka_emika_panda/LICENSE`) |
| `universal_robots_ur5e/` | UR5e | ROS-Industrial `universal_robot` description, © 2018 ROS Industrial Consortium | BSD-3-Clause (`universal_robots_ur5e/LICENSE`) |
| `robotiq_2f85/` | Robotiq 2F-85 gripper | ROS-Industrial `robotiq` description, © 2013 ROS-Industrial | BSD-2-Clause (`robotiq_2f85/LICENSE`) |

The files are used unchanged; `cobot_sandbox/robots.py` adapts the models in memory when loading them
(attaches the gripper, replaces position servos with torque motors, adds sites, a camera and
sensors). `scripts/fetch_models.py` re-creates the folder from the pinned upstream commit.

## Trademarks

"Franka", "Franka Research 3", "Universal Robots", "UR5e" and "Robotiq" are trademarks of their
respective owners. They are used here only to identify the robot models; this project is not
affiliated with or endorsed by those companies.
