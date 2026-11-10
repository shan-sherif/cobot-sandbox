#!/usr/bin/env python
"""Write a scene's generated MJCF to build/<scene>.xml, e.g. to inspect it in MuJoCo's own viewer:

    python scripts/export_scene.py lift
    python -m mujoco.viewer --mjcf=build/lift.xml     # (macOS: use mjpython if it complains)

Note: in the native viewer the arm is unpowered (its servo runs in cobot_sandbox/sim.py) and will sag.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cobot_sandbox.paths import BUILD_DIR  # noqa: E402
from cobot_sandbox.scene_builder import build_spec, build_xml, compile_scene, list_scenes  # noqa: E402

if __name__ == "__main__":
    names = sys.argv[1:] or list_scenes()
    out = BUILD_DIR
    out.mkdir(parents=True, exist_ok=True)
    for n in names:
        spec = compile_scene(n)
        p = out / f"{spec.name}.xml"
        p.write_text(build_spec(spec, build_xml(spec)).to_xml())   # robots included (absolute mesh paths)
        print(p)
