"""Scene menu overlay for the app: a card per scene (preview image, number, title, description).

Opens at startup when no scene was given on the command line, and with L at any time.
Keys: Up/Down (or W/S) select, 0-9 jump to that scene number, Enter/Space load, Esc close. Mouse: hover + click.
Gamepad: D-pad select, A load, B close.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import mujoco
import pygame

from .scene_builder import scene_info

THUMB_W, THUMB_H = 192, 108
ROW_H = THUMB_H + 16
ACCENT = (120, 220, 235)


def render_thumbnail(name: str) -> Optional[pygame.Surface]:
    """Preview of a scene from its `overview` camera (or a default free camera)."""
    from .sim import Sim
    try:
        sim = Sim(name, seed=0)
    except Exception:            # a broken scene file shouldn't take the menu down
        return None
    try:
        r = sim.renderer(THUMB_W * 2, THUMB_H * 2)   # supersample, then smooth-scale down
        if "overview" in sim.cameras:
            r.update_scene(sim.data, camera="overview")
        else:
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation, cam.distance = 145.0, -28.0, 3.2
            o, R, _ = sim.base_pose()
            cam.lookat[:] = o + R @ [0.3, 0, 0.45]
            r.update_scene(sim.data, camera=cam)
        img = r.render()
        surf = pygame.image.frombuffer(img.tobytes(), (THUMB_W * 2, THUMB_H * 2), "RGB")
        return pygame.transform.smoothscale(surf, (THUMB_W, THUMB_H))
    finally:
        sim.close()


class SceneMenu:
    def __init__(self, on_select: Callable[[str], None]):
        self.on_select = on_select
        self.open = False
        self.scenes: List[dict] = []
        self.sel = 0
        self.thumbs: Dict[str, Optional[pygame.Surface]] = {}
        self.rows: List[pygame.Rect] = []    # hit boxes of the visible cards, filled by draw()
        self.first = 0                       # first visible row (scrolling)
        self.closable = True

    def show(self, current: Optional[str] = None, closable: bool = True):
        self.scenes = scene_info()
        names = [s["name"] for s in self.scenes]
        self.sel = names.index(current) if current in names else 0
        self.current, self.closable, self.open = current, closable, True

    def close(self):
        if self.closable:
            self.open = False

    def _choose(self, i: int):
        if 0 <= i < len(self.scenes):
            self.open = False
            self.on_select(self.scenes[i]["name"])

    def update(self):
        """Render at most one missing preview per frame so the menu stays responsive."""
        for s in self.scenes:
            if s["name"] not in self.thumbs:
                self.thumbs[s["name"]] = render_thumbnail(s["name"])
                return

    # ------------------------------------------------------------------ input
    def handle_event(self, e) -> bool:
        """Returns False if the app should quit; every other event is swallowed while open."""
        n = len(self.scenes)
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            k = e.key
            if k in (pygame.K_UP, pygame.K_w):
                self.sel = (self.sel - 1) % n
            elif k in (pygame.K_DOWN, pygame.K_s):
                self.sel = (self.sel + 1) % n
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._choose(self.sel)
            elif k in (pygame.K_ESCAPE, pygame.K_l):
                if not self.closable and k == pygame.K_ESCAPE:
                    return False                      # startup menu: Esc quits, like elsewhere
                self.close()
            elif pygame.K_0 <= k <= pygame.K_9:         # jump to that scene number
                nums = [s["number"] for s in self.scenes]
                if k - pygame.K_0 in nums:
                    self._choose(nums.index(k - pygame.K_0))
        elif e.type == pygame.MOUSEMOTION:
            for i, r in self.rows:
                if r.collidepoint(e.pos):
                    self.sel = i
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for i, r in self.rows:
                if r.collidepoint(e.pos):
                    self._choose(i)
        elif e.type == pygame.MOUSEWHEEL:
            self.sel = (self.sel - e.y) % n
        elif e.type == getattr(pygame, "CONTROLLERBUTTONDOWN", -1):
            if e.button == pygame.CONTROLLER_BUTTON_DPAD_UP:
                self.sel = (self.sel - 1) % n
            elif e.button == pygame.CONTROLLER_BUTTON_DPAD_DOWN:
                self.sel = (self.sel + 1) % n
            elif e.button == pygame.CONTROLLER_BUTTON_A:
                self._choose(self.sel)
            elif e.button == pygame.CONTROLLER_BUTTON_B:
                self.close()
        return True

    # ------------------------------------------------------------------ drawing
    def draw(self, screen: pygame.Surface, font: pygame.font.Font, font_b: pygame.font.Font,
             font_big: pygame.font.Font):
        W, H = screen.get_size()
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((8, 10, 14, 200))
        screen.blit(shade, (0, 0))

        pw = min(860, W - 40)
        visible = max(1, min(len(self.scenes), (H - 170) // ROW_H))
        ph = 110 + visible * ROW_H
        x0, y0 = (W - pw) // 2, (H - ph) // 2
        pygame.draw.rect(screen, (18, 21, 27), (x0, y0, pw, ph), border_radius=12)
        pygame.draw.rect(screen, (45, 52, 62), (x0, y0, pw, ph), width=1, border_radius=12)
        screen.blit(font_big.render("Select a scene", True, ACCENT), (x0 + 24, y0 + 18))

        # keep the selection in view
        if self.sel < self.first:
            self.first = self.sel
        elif self.sel >= self.first + visible:
            self.first = self.sel - visible + 1
        self.rows = []
        y = y0 + 64
        for i in range(self.first, min(len(self.scenes), self.first + visible)):
            s = self.scenes[i]
            card = pygame.Rect(x0 + 16, y, pw - 32, ROW_H - 8)
            self.rows.append((i, card))
            selected = i == self.sel
            pygame.draw.rect(screen, (30, 38, 48) if selected else (24, 27, 33), card, border_radius=8)
            if selected:
                pygame.draw.rect(screen, ACCENT, card, width=2, border_radius=8)
            tx, ty = card.x + 8, card.y + (card.h - THUMB_H) // 2
            thumb = self.thumbs.get(s["name"])
            if thumb is not None:
                screen.blit(thumb, (tx, ty))
            else:
                pygame.draw.rect(screen, (36, 40, 48), (tx, ty, THUMB_W, THUMB_H))
                msg = "loading..." if s["name"] not in self.thumbs else "no preview"
                screen.blit(font.render(msg, True, (130, 135, 145)), (tx + 12, ty + THUMB_H // 2 - 8))
            lx = tx + THUMB_W + 18
            tag = f"Scene {s['number']}" if s["number"] is not None else "Extra"
            if s["name"] == self.current:
                tag += "   (current)"
            screen.blit(font.render(tag, True, ACCENT if selected else (140, 150, 160)), (lx, card.y + 14))
            screen.blit(font_big.render(s["title"], True, (240, 240, 240)), (lx, card.y + 34))
            maxw = card.right - lx - 16
            for j, line in enumerate(_wrap(s["description"], font, maxw)[:2]):
                screen.blit(font.render(line, True, (185, 190, 200)), (lx, card.y + 62 + 16 * j))
            robots = {"franka_fr3": "FR3", "ur5e": "UR5e"}
            names = ", ".join(robots.get(r, r) for r in dict.fromkeys(s.get("robots", [])))
            meta = (("live: " if s.get("robot_select") == "all" else "pick: ") + names) if names else ""
            if not s.get("recording", True):
                meta += ("   |   " if meta else "") + "No recording (.npz) in this scene"
            if meta:
                screen.blit(font.render(meta, True, (255, 200, 110) if not s.get("recording", True) else (130, 200, 215)),
                            (lx, card.bottom - 19))
            y += ROW_H

        hint = "Up/Down or mouse select   Enter load   0-9 scene no."
        hint += "   Esc close" if self.closable else "   Esc quit"
        if len(self.scenes) > visible:
            hint += f"   ({self.first + 1}-{self.first + visible} of {len(self.scenes)})"
        screen.blit(font.render(hint, True, (140, 145, 155)), (x0 + 24, y0 + ph - 30))


def _wrap(text: str, font: pygame.font.Font, width: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.size(trial)[0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + ([cur] if cur else [])
