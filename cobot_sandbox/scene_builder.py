"""Turn a scene YAML (environment, robot mounts, fixtures, objects, tasks, cameras) + robot models into one
MuJoCo model.

Robots are defined once (assets/robot/<name>.yaml, built by robots.py) and attached to the scene's
mounts, so anything you build or learn transfers between scenes: only the YAML changes. See
scenes/_template.yaml for the full schema.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np
import yaml

from .paths import ROBOT_DIR, SCENE_DIR, TEXTURE_DIR

PLATFORM_DIR = ROBOT_DIR / "platforms"
PLATFORM_MOUNT_Z = 0.28                    # robot mount height on a pedestal (platforms/pedestal.xml)
from .textures import ensure_textures

BIN_FLOOR = 0.02    # bin floor thickness (m); thin floors/walls let light falling objects tunnel or spin
BIN_WALL = 0.0075   # bin wall half-thickness (1.5 cm walls)

PALETTE = [
    (0.90, 0.22, 0.20), (0.15, 0.55, 0.95), (0.98, 0.72, 0.10), (0.20, 0.75, 0.40),
    (0.62, 0.35, 0.85), (0.98, 0.45, 0.15), (0.10, 0.75, 0.80), (0.95, 0.40, 0.65),
]


# --------------------------------------------------------------------------- scene spec
@dataclass
class Fixture:
    name: str
    type: str
    pos: np.ndarray            # world x, y (and base z for stacked fixtures)
    yaw: float
    size: List[float]
    material: Any
    top: float                 # height of the usable top surface
    base_z: float = 0.0
    height: float = 0.4        # table height
    hole: float = 0.016        # half-width of the hole (type "hole")

    def contains(self, x, y, margin=0.0) -> bool:
        lx, ly = self.to_local(x, y)
        return abs(lx) <= self.size[0] - margin and abs(ly) <= self.size[1] - margin

    def to_local(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x - self.pos[0], y - self.pos[1]
        return c * dx + s * dy, -s * dx + c * dy

    def to_world(self, lx, ly):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.pos[0] + c * lx - s * ly, self.pos[1] + s * lx + c * ly


@dataclass
class ObjectSpec:
    name: str
    type: str                  # cube | ball | cylinder | box
    size: List[float]          # MuJoCo geom size
    mass: float
    color: Any
    friction: List[float]
    pos: np.ndarray            # nominal world position (x, y, z)
    yaw: float
    region: Optional[str] = None   # fixture name objects are (re)sampled on
    margin: float = 0.06

    @property
    def half_height(self) -> float:
        if self.type == "ball":
            return self.size[0]
        if self.type == "cylinder":
            return self.size[1]
        return self.size[2]

    @property
    def footprint(self) -> float:
        if self.type in ("ball", "cylinder"):
            return self.size[0]
        return math.hypot(self.size[0], self.size[1])


@dataclass
class SceneSpec:
    name: str
    raw: Dict[str, Any]
    fixtures: List[Fixture] = field(default_factory=list)
    objects: List[ObjectSpec] = field(default_factory=list)
    cameras: List[Dict[str, Any]] = field(default_factory=list)
    mounts: List[Dict[str, Any]] = field(default_factory=list)     # robot mount points (pedestals)
    robot_select: str = "one"            # one: the user picks one robot; all: every robot is live
    tasks: List[Dict[str, Any]] = field(default_factory=list)      # success checks (cobot_sandbox/tasks.py)
    recording: bool = True                # False: no .npz recording in this scene (e.g. the test bench)
    randomize: Dict[str, Any] = field(default_factory=dict)
    labels: List[Dict[str, Any]] = field(default_factory=list)   # floating text drawn by the app

    def fixture(self, name: str) -> Fixture:
        for f in self.fixtures:
            if f.name == name:
                return f
        raise KeyError(f"unknown fixture '{name}'")

    def surface_height(self, x: float, y: float) -> float:
        h = 0.0
        for f in self.fixtures:
            if f.type in ("table", "box", "bin", "hole") and f.contains(x, y):
                h = max(h, f.top)
        return h

    def bins(self) -> List[Fixture]:
        return [f for f in self.fixtures if f.type == "bin"]


class _Loader(yaml.SafeLoader):
    """SafeLoader where only true/false are booleans (YAML 1.1 would turn the key `on:` into True)."""


_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:bool"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", __import__("re").compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"))


def read_yaml(path) -> Any:
    with open(path) as fh:
        return yaml.load(fh, Loader=_Loader)


_NUMBERED_FILE = __import__("re").compile(r"^(\d+)_(.+)$")  # "02_stack" -> number 2 (menu order), name "stack"


def _split_stem(stem: str):
    """File stem -> (scene name, number from the numeric prefix or None)."""
    m = _NUMBERED_FILE.match(stem)
    return (m.group(2), int(m.group(1))) if m else (stem, None)


def _scene_files() -> Dict[str, Path]:
    """Scene name -> file, for every scene in SCENE_DIR (files starting with "_" are skipped)."""
    return {_split_stem(p.stem)[0]: p for p in sorted(SCENE_DIR.glob("*.yaml")) if not p.stem.startswith("_")}


def scene_path(scene: str) -> Path:
    """Resolve a scene argument: a path to a .yaml, a scene name ("stack") or a file stem
    with its number prefix ("02_stack")."""
    p = Path(scene)
    if p.suffix:
        return p
    files = _scene_files()
    name = scene if scene in files else _split_stem(scene)[0]
    if name not in files:
        raise FileNotFoundError(f"unknown scene '{scene}' (available: {', '.join(list_scenes())})")
    return files[name]


def scene_info() -> List[Dict[str, Any]]:
    """Every scene in SCENE_DIR in menu order: by the file's numeric prefix (02_stack.yaml has
    number 2). Unnumbered
    scenes come last, alphabetically. `title:` is the display name (defaults to the scene name)."""
    out = []
    for name, p in _scene_files().items():
        raw = load_scene_yaml(p)
        rb = raw.get("robots") or {}
        out.append({"name": name, "number": raw.get("number"), "file": p.name,
                    "title": str(raw.get("title") or name), "description": str(raw.get("description") or ""),
                    "recording": bool(raw.get("recording", True)), "robot_select": rb.get("select", "one"),
                    "robots": [m["robot"] for m in rb.get("mounts", []) or []]})
    return sorted(out, key=lambda s: (s["number"] is None, s["number"] or 0, s["name"]))


def list_scenes() -> List[str]:
    """Scene names in menu order: what --scene and the app's [ / ] switcher use."""
    return [s["name"] for s in scene_info()]


def scene_label(raw: Dict[str, Any]) -> str:
    """Display name of a scene, e.g. 'Pick and place' (the file number only orders the menu)."""
    return str(raw.get("title") or raw.get("name", "scene"))


def load_scene_yaml(scene) -> Dict[str, Any]:
    if isinstance(scene, dict):
        raw = copy.deepcopy(scene)
        raw.setdefault("name", "custom")
        return raw
    p = scene_path(str(scene))
    raw = read_yaml(p) or {}
    name, number = _split_stem(p.stem)
    raw.setdefault("name", name)
    raw["number"] = number                # menu position from the file name (None: listed last)
    return raw


def _vec(v, n, default=0.0):
    v = list(v) if v is not None else []
    return np.array(v + [default] * (n - len(v)), dtype=float)[:n]


def compile_scene(scene, seed: int = 0) -> SceneSpec:
    raw = load_scene_yaml(scene)
    rng = np.random.default_rng(seed)
    spec = SceneSpec(name=raw["name"], raw=raw, randomize=raw.get("randomize") or {})

    # Fixtures may sit on top of other fixtures ("on: table_a").
    for f in raw.get("fixtures", []) or []:
        ftype = f["type"]
        pos = _vec(f.get("pos"), 2)
        base_z = spec.fixture(f["on"]).top if f.get("on") else 0.0
        size = list(f.get("size", [0.3, 0.3, 0.1]))
        if ftype == "table":
            top = base_z + float(f.get("height", 0.4))
        elif ftype == "box":
            top = base_z + 2 * size[2]
        elif ftype == "bin":
            top = base_z + BIN_FLOOR       # inner floor
        elif ftype == "hole":              # block with a square vertical hole (peg insertion)
            top = base_z + 2 * size[2]
        elif ftype == "target":            # flat visual marker on a surface (push goals etc.)
            top = base_z
        else:
            raise ValueError(f"unknown fixture type '{ftype}'")
        spec.fixtures.append(Fixture(
            name=f["name"], type=ftype, pos=pos, yaw=math.radians(f.get("yaw_deg", 0.0)),
            size=size, material=f.get("material", f.get("color")), top=top, base_z=base_z,
            height=float(f.get("height", 0.4)), hole=float(f.get("hole", 0.016))))

    color_i = 0

    def make_obj(o, name):
        nonlocal color_i
        otype = o.get("type", "cube")
        s = o.get("size", 0.025)
        if otype == "cube":
            size = [float(s)] * 3
        elif otype == "ball":
            size = [float(s if not isinstance(s, list) else s[0])]
        elif otype == "cylinder":
            size = list(s) if isinstance(s, list) else [float(s), float(s) * 1.4]
        elif otype == "box":
            size = list(s)
        else:
            raise ValueError(f"unknown object type '{otype}'")
        color = o.get("color")
        if color is None and not o.get("material"):
            color = PALETTE[color_i % len(PALETTE)]
            color_i += 1
        return ObjectSpec(
            name=name, type=otype, size=size, mass=float(o.get("mass", 0.08)),
            color=o.get("material") or list(color),
            friction=list(o.get("friction", [1.0, 0.01, 0.001])),
            pos=_vec(o.get("pos"), 3, default=np.nan), yaw=math.radians(o.get("yaw_deg", 0.0)),
            region=o.get("on"), margin=float(o.get("margin", 0.06)))

    for o in raw.get("objects", []) or []:
        spec.objects.append(make_obj(o, o["name"]))

    # Procedural spawners: N random objects on a fixture.
    for gi, g in enumerate(raw.get("spawn", []) or []):
        types = g.get("types", ["cube", "ball", "cylinder"])
        lo, hi = g.get("size_range", [0.02, 0.03])
        for i in range(int(g.get("count", 4))):
            t = types[int(rng.integers(len(types)))]
            s = float(rng.uniform(lo, hi))
            o = {"type": t, "size": s if t != "cylinder" else [s, s * float(rng.uniform(1.0, 1.8))],
                 "mass": float(rng.uniform(*g.get("mass_range", [0.04, 0.15]))),
                 "on": g["on"], "margin": g.get("margin", 0.08)}
            if g.get("colors"):
                o["color"] = g["colors"][i % len(g["colors"])]
            spec.objects.append(make_obj(o, f"{g.get('prefix', 'item')}{gi}_{i}"))

    place_objects(spec, rng, randomize=False)

    for c in raw.get("cameras", []) or []:
        spec.cameras.append({"name": c["name"], "pos": _vec(c["pos"], 3),
                             "lookat": _vec(c.get("lookat", [0, 0, 0.4]), 3),
                             "fovy": float(c.get("fovy", 45))})

    for lb in raw.get("labels", []) or []:
        spec.labels.append({"text": str(lb["text"]), "pos": _vec(lb.get("pos"), 3),
                            "on_robot": bool(lb.get("on_robot", False))})   # on_robot: pos is relative to the base

    rb = raw.get("robots") or {}
    spec.robot_select = str(rb.get("select", "one"))
    for m in rb.get("mounts", []) or [{"robot": "franka_fr3"}, {"robot": "ur5e"}]:
        p = _vec(m.get("pos"), 2)
        stand = m.get("stand", "pedestal")
        spec.mounts.append({"robot": m["robot"], "pos": p, "yaw": math.radians(float(m.get("yaw_deg", 0))),
                            "stand": stand, "z": float(m.get("z", PLATFORM_MOUNT_Z if stand else 0.0)),
                            "label": m.get("label")})
    for i, m in enumerate(spec.mounts):          # robot name labels (the app marks the active one)
        if m["label"]:
            spec.labels.append({"text": str(m["label"]), "pos": [m["pos"][0], m["pos"][1], m["z"] + 0.78], "mount": i})
    spec.tasks = [dict(tk) for tk in raw.get("tasks", []) or []]
    spec.recording = bool(raw.get("recording", True))
    return spec


def place_objects(spec: SceneSpec, rng, randomize: bool, xy_jitter: float = 0.0,
                  random_yaw: bool = False) -> Dict[str, tuple]:
    """Compute (pos, yaw) for every object. Objects with a region and no fixed pos (or all objects
    when randomize=True) are sampled on their fixture without overlaps."""
    placed: List[tuple] = []
    out = {}
    for o in spec.objects:
        fixed = not np.isnan(o.pos[0])
        yaw = float(rng.uniform(-math.pi, math.pi)) if (randomize and random_yaw) else o.yaw
        resample = (not fixed) or (randomize and o.region is not None)
        if not resample:
            x, y = o.pos[0], o.pos[1]
            if randomize and xy_jitter:
                x += rng.uniform(-xy_jitter, xy_jitter)
                y += rng.uniform(-xy_jitter, xy_jitter)
        else:
            if not o.region:
                raise ValueError(f"object '{o.name}' needs either pos or on: <fixture>")
            fx = spec.fixture(o.region)
            m = o.margin + o.footprint
            # things standing on this fixture (bins, boxes) are keep-out zones
            blockers = [f for f in spec.fixtures if f is not fx and f.base_z > 0 and abs(f.base_z - fx.top) < 1e-6]
            for _ in range(300):
                lx = rng.uniform(-fx.size[0] + m, fx.size[0] - m)
                ly = rng.uniform(-fx.size[1] + m, fx.size[1] - m)
                x, y = fx.to_world(lx, ly)
                if any(f.contains(x, y, margin=-(o.footprint + 0.02)) for f in blockers):
                    continue
                if all(math.hypot(x - px, y - py) > o.footprint + pr + 0.02 for px, py, pr in placed):
                    break
        if fixed and not np.isnan(o.pos[2]) and not randomize:
            z = o.pos[2]
        else:
            z = spec.surface_height(x, y) + o.half_height + 0.002
        placed.append((x, y, o.footprint))
        out[o.name] = (np.array([x, y, z]), yaw)
        if not randomize:
            o.pos = np.array([x, y, z])
    return out


# --------------------------------------------------------------------------- XML generation
def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: (v if isinstance(v, str) else _f(v)) for k, v in attrs.items()
                                       if v is not None})


class _Assets:
    BUILTIN = {
        "concrete": dict(texture="concrete", texrepeat=(0.5, 0.5), reflectance=0.04, specular=0.15, shininess=0.2),
        "wood": dict(texture="wood", texrepeat=(1, 1), reflectance=0.04, specular=0.3, shininess=0.4),
        "metal": dict(texture="brushed_metal", texrepeat=(2, 2), reflectance=0.1, specular=0.9, shininess=0.8),
    }

    def __init__(self, asset_el, textures):
        self.el = asset_el
        self.textures = textures
        self.done = set()
        self.tex_done = set()
        self.n = 0

    def _texture(self, stem):
        if stem not in self.textures:
            raise FileNotFoundError(f"texture '{stem}.png' not found in {TEXTURE_DIR}")
        if stem not in self.tex_done:
            _sub(self.el, "texture", name=f"tex_{stem}", type="2d", file=f"{stem}.png")
            self.tex_done.add(stem)
        return f"tex_{stem}"

    def material(self, spec, hint="mat") -> str:
        """spec: builtin name | existing texture stem | [r,g,b(,a)] | {texture, repeat, rgba, ...}."""
        if spec is None:
            spec = [0.7, 0.7, 0.7]
        if isinstance(spec, str):
            if spec in self.BUILTIN:
                name = f"m_{spec}"
                if name not in self.done:
                    b = dict(self.BUILTIN[spec])
                    tex = self._texture(b.pop("texture"))
                    _sub(self.el, "material", name=name, texture=tex, texuniform="true", **b)
                    self.done.add(name)
                return name
            return self.material({"texture": spec}, hint)
        if isinstance(spec, (list, tuple)):
            rgba = list(spec) + [1.0] * (4 - len(spec))
            self.n += 1
            name = f"m_{hint}_{self.n}"
            _sub(self.el, "material", name=name, rgba=rgba, specular=0.45, shininess=0.55, reflectance=0.03)
            return name
        if isinstance(spec, dict):
            self.n += 1
            name = f"m_{hint}_{self.n}"
            attrs = {"rgba": spec.get("rgba"), "specular": spec.get("specular", 0.4),
                     "shininess": spec.get("shininess", 0.5), "reflectance": spec.get("reflectance", 0.03),
                     "emission": spec.get("emission")}
            if spec.get("texture"):
                attrs["texture"] = self._texture(spec["texture"])
                attrs["texrepeat"] = spec.get("repeat", (1, 1))
                attrs["texuniform"] = "true" if spec.get("uniform", True) else "false"
            _sub(self.el, "material", name=name, **attrs)
            return name
        raise ValueError(f"bad material spec {spec!r}")


def _lookat_xyaxes(pos, lookat):
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(-f, x)
    return np.concatenate([x, y])


def _fragment_bodies(root, path: Path, loaded: set, prefix: str = ""):
    """Bodies of a visual fragment (platforms/*.xml), adding its <asset> entries to `root` once."""
    frag = ET.parse(path).getroot()
    if path not in loaded:
        for a in frag.findall("asset/*"):
            root.find("asset").append(a)
        loaded.add(path)
    out = []
    for b in frag.findall("worldbody/body"):
        b = copy.deepcopy(b)
        if prefix:
            for el in b.iter():
                if "name" in el.attrib:
                    el.set("name", prefix + el.get("name"))
        out.append(b)
    return out


def build_model(spec: SceneSpec, xml: str, robots: Optional[List[dict]] = None) -> mujoco.MjModel:
    return build_spec(spec, xml, robots).compile()


def build_spec(spec: SceneSpec, xml: str, robots: Optional[List[dict]] = None) -> mujoco.MjSpec:
    """Compile a scene: the XML from build_xml plus the robots, attached to their `mount_<i>` body
    (default: robots.resolve() of the scene's mounts)."""
    from . import robots as R
    if robots is None:
        robots = R.resolve(spec.mounts, spec.robot_select)
    s = mujoco.MjSpec.from_string(xml)
    for r in robots:
        body = s.body(f"mount_{r['index']}")
        child = R.build(r["robot"])
        R.match_options(s, child)
        s.attach(child, frame=body.add_frame(), prefix=r["prefix"])
    return s


def build_xml(spec: SceneSpec) -> str:
    """Scene without robots (build_model attaches them to the `mount_<i>` bodies made here)."""
    raw = spec.raw
    env = raw.get("environment", {}) or {}
    textures = ensure_textures()

    root = ET.Element("mujoco", model=spec.name)
    _sub(root, "compiler", angle="radian", autolimits="true", texturedir=str(TEXTURE_DIR))
    opt = raw.get("physics", {}) or {}
    # 1 ms physics: at 2 ms, light objects piling up in a bin occasionally blew up with the Menagerie
    # grippers (measured: FR3 pick-and-place 8/10 at 2 ms, 10/10 at 1 ms).
    _sub(root, "option", timestep=opt.get("timestep", 0.001), integrator="implicitfast",
         cone="elliptic", impratio=opt.get("impratio", 1))
    vis = _sub(root, "visual")
    _sub(vis, "global", offwidth="1920", offheight="1080", azimuth="140", elevation="-25")
    _sub(vis, "quality", shadowsize="4096", offsamples="8")
    _sub(vis, "headlight", ambient=(0.25, 0.25, 0.28), diffuse=(0.25, 0.25, 0.25), specular=(0.1, 0.1, 0.1))
    _sub(vis, "map", znear="0.01", zfar="60", haze="0.35", shadowclip="3", shadowscale="0.5")
    _sub(vis, "rgba", haze=env.get("haze", [0.16, 0.19, 0.24, 1]))

    default = _sub(root, "default")
    asset = _sub(root, "asset")
    assets = _Assets(asset, textures)
    # Visual fragments (platforms/*.xml: pedestals) share materials and a visual-only default class
    # (platforms/_common.xml); copy those in (robots themselves are attached separately).
    common = ET.parse(PLATFORM_DIR / "_common.xml").getroot()
    for mat in common.findall("asset/material"):
        asset.append(copy.deepcopy(mat))
    for cls in common.findall("default/default"):
        default.append(copy.deepcopy(cls))
    sky = env.get("sky", {}) or {}
    _sub(asset, "texture", name="skybox", type="skybox", builtin="gradient",
         rgb1=sky.get("top", [0.52, 0.62, 0.76]), rgb2=sky.get("bottom", [0.08, 0.1, 0.13]),
         width="512", height="3072")
    _sub(root, "worldbody")
    for tag in ("contact", "equality", "actuator", "sensor"):
        _sub(root, tag)

    loaded = set()
    wb = root.find("worldbody")
    # ---- environment
    floor = env.get("floor", {}) or {}
    fs = floor.get("size", [6, 6])
    _sub(wb, "geom", name="floor", type="plane", size=[fs[0], fs[1], 0.05],
         material=assets.material(floor.get("material", "concrete"), "floor"), condim="3")
    walls = env.get("walls")
    if walls:
        wh, wt = float(walls.get("height", 1.0)), 0.05
        wm = assets.material(walls.get("material", [0.75, 0.76, 0.78]), "wall")
        for i, (px, py, sx, sy) in enumerate([(fs[0], 0, wt, fs[1]), (-fs[0], 0, wt, fs[1]),
                                              (0, fs[1], fs[0], wt), (0, -fs[1], fs[0], wt)]):
            _sub(wb, "geom", name=f"wall{i}", type="box", pos=[px, py, wh / 2], size=[sx, sy, wh / 2], material=wm)

    lights = env.get("lights") or [
        {"pos": [2.5, -2.0, 4.0], "dir": [-0.45, 0.35, -1], "diffuse": [0.75, 0.72, 0.68], "shadow": True},
        {"pos": [-3.0, 2.5, 3.5], "dir": [0.6, -0.4, -1], "diffuse": [0.25, 0.28, 0.34], "shadow": False},
    ]
    for i, l in enumerate(lights):
        _sub(wb, "light", name=f"light{i}", pos=l["pos"], dir=l["dir"], directional="true",
             diffuse=l.get("diffuse", [0.5, 0.5, 0.5]), specular=l.get("specular", [0.3, 0.3, 0.3]),
             castshadow="true" if l.get("shadow") else "false")

    for c in spec.cameras:
        _sub(wb, "camera", name=c["name"], pos=c["pos"], xyaxes=_lookat_xyaxes(c["pos"], c["lookat"]),
             fovy=c["fovy"])

    # ---- robot mounts: a stand (visual, e.g. pedestal) + an empty `mount_<i>` body the robot attaches to
    for i, m in enumerate(spec.mounts):
        if m["stand"]:
            stand = _sub(wb, "body", name=f"stand_{i}", pos=[m["pos"][0], m["pos"][1], 0], euler=[0, 0, m["yaw"]])
            for b in _fragment_bodies(root, PLATFORM_DIR / f"{m['stand']}.xml", loaded, prefix=f"stand{i}_"):
                stand.append(b)
            if m["stand"] == "pedestal":       # collision for the pedestal column
                _sub(stand, "geom", name=f"stand{i}_col", type="cylinder", size=[0.09, m["z"] / 2],
                     pos=[0, 0, m["z"] / 2], rgba=[0, 0, 0, 0])
        _sub(wb, "body", name=f"mount_{i}", pos=[m["pos"][0], m["pos"][1], m["z"]], euler=[0, 0, m["yaw"]])

    # ---- fixtures
    for fx in spec.fixtures:
        b = _sub(wb, "body", name=fx.name, pos=[fx.pos[0], fx.pos[1], fx.base_z], euler=[0, 0, fx.yaw])
        sx, sy = fx.size[0], fx.size[1]
        if fx.type == "table":
            ht = fx.height
            _sub(b, "geom", name=f"{fx.name}_top", type="box", pos=[0, 0, ht - 0.015], size=[sx, sy, 0.015],
                 material=assets.material(fx.material or "wood", fx.name))
            leg_m = assets.material("metal")
            for i, (lx, ly) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                _sub(b, "geom", name=f"{fx.name}_leg{i}", type="box",
                     pos=[lx * (sx - 0.04), ly * (sy - 0.04), (ht - 0.03) / 2],
                     size=[0.02, 0.02, (ht - 0.03) / 2], material=leg_m)
        elif fx.type == "box":
            _sub(b, "geom", name=f"{fx.name}_geom", type="box", pos=[0, 0, fx.size[2]], size=fx.size,
                 material=assets.material(fx.material or [0.55, 0.45, 0.35], fx.name))
        elif fx.type == "hole":
            m = assets.material(fx.material or [0.55, 0.58, 0.62], fx.name)
            h, H = fx.hole, fx.size[2]
            a, w = (h + sx) / 2, (sx - h) / 2
            for i, (p, s) in enumerate([([a, 0, H], [w, sy, H]), ([-a, 0, H], [w, sy, H]),
                                        ([0, (h + sy) / 2, H], [h, (sy - h) / 2, H]),
                                        ([0, -(h + sy) / 2, H], [h, (sy - h) / 2, H])]):
                _sub(b, "geom", name=f"{fx.name}_wall{i}", type="box", pos=p, size=s, material=m)
            _sub(b, "geom", name=f"{fx.name}_bottom", type="box", pos=[0, 0, 0.004], size=[h, h, 0.004], material=m)
        elif fx.type == "target":
            _sub(b, "geom", name=f"{fx.name}_marker", type="cylinder", size=[sx, 0.0008], pos=[0, 0, 0.0009],
                 rgba=fx.material or [0.2, 0.9, 0.4, 0.55], contype="0", conaffinity="0", group="1")
        elif fx.type == "bin":
            m = assets.material(fx.material or [0.2, 0.45, 0.75], fx.name)
            t, H = BIN_WALL, fx.size[2]
            _sub(b, "geom", name=f"{fx.name}_floor", type="box", pos=[0, 0, BIN_FLOOR / 2], size=[sx, sy, BIN_FLOOR / 2], material=m)
            for i, (p, s) in enumerate([([sx - t, 0, H / 2], [t, sy, H / 2]), ([-sx + t, 0, H / 2], [t, sy, H / 2]),
                                        ([0, sy - t, H / 2], [sx, t, H / 2]), ([0, -sy + t, H / 2], [sx, t, H / 2])]):
                _sub(b, "geom", name=f"{fx.name}_wall{i}", type="box", pos=p, size=s, material=m)

    # ---- objects
    for ob in spec.objects:
        b = _sub(wb, "body", name=ob.name, pos=ob.pos, euler=[0, 0, ob.yaw])
        _sub(b, "freejoint", name=f"{ob.name}_free")
        gtype = {"cube": "box", "ball": "sphere", "cylinder": "cylinder", "box": "box"}[ob.type]
        cond = "6" if ob.type == "ball" else "4"
        fr = list(ob.friction)[:3]
        if ob.type == "ball":
            fr[2] = max(fr[2], 0.0005)       # rolling resistance so balls come to rest
        _sub(b, "geom", name=ob.name, type=gtype, size=ob.size, mass=ob.mass, condim=cond,
             friction=fr, material=assets.material(ob.color, ob.name))
        _sub(b, "site", name=f"{ob.name}_site", size="0.004", group="4")

    ET.indent(root) if hasattr(ET, "indent") else None
    return ET.tostring(root, encoding="unicode")
