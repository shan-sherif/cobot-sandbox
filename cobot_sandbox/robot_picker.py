"""Robot picker overlay: shown when a single-robot scene loads (and with R). Lists the robots the
scene offers; a robot whose model files are missing is shown greyed out.

Keys: Up/Down (or W/S) select, 1-9 jump, Enter/Space choose, Esc keep the current robot.
Mouse: hover + click. Gamepad: D-pad, A choose, B close.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import pygame

from . import robots as R

ACCENT = (120, 220, 235)
ROW_H = 64


class RobotPicker:
    def __init__(self, on_select: Callable[[str], None]):
        self.on_select = on_select
        self.open = False
        self.items: List[dict] = []
        self.sel = 0
        self.rows = []

    def show(self, names: List[str], current: Optional[str] = None):
        self.items = []
        for n in dict.fromkeys(names):                         # unique, keep order
            cfg = R.load_cfg(n)
            self.items.append({"name": n, "title": cfg.get("title", n), "ok": R.available(n),
                               "dof": len(cfg["arm_joints"]), "source": cfg.get("source", "xml")})
        ok = [i for i, it in enumerate(self.items) if it["ok"]]
        names_ok = [self.items[i]["name"] for i in ok]
        self.sel = names_ok.index(current) if current in names_ok else 0
        self.sel = ok[self.sel] if ok else 0
        self.current, self.open = current, True

    def _choose(self, i):
        if 0 <= i < len(self.items) and self.items[i]["ok"]:
            self.open = False
            self.on_select(self.items[i]["name"])

    def _move(self, step):
        n = len(self.items)
        for _ in range(n):
            self.sel = (self.sel + step) % n
            if self.items[self.sel]["ok"]:
                return

    def handle_event(self, e) -> bool:
        """Returns False if the app should quit; every other event is swallowed while open."""
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            k = e.key
            if k in (pygame.K_UP, pygame.K_w):
                self._move(-1)
            elif k in (pygame.K_DOWN, pygame.K_s):
                self._move(1)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._choose(self.sel)
            elif k in (pygame.K_ESCAPE, pygame.K_r):
                self.open = False
            elif pygame.K_1 <= k <= pygame.K_9:
                self._choose(k - pygame.K_1)
        elif e.type == pygame.MOUSEMOTION:
            for i, r in self.rows:
                if r.collidepoint(e.pos) and self.items[i]["ok"]:
                    self.sel = i
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for i, r in self.rows:
                if r.collidepoint(e.pos):
                    self._choose(i)
        elif e.type == getattr(pygame, "CONTROLLERBUTTONDOWN", -1):
            if e.button == pygame.CONTROLLER_BUTTON_DPAD_UP:
                self._move(-1)
            elif e.button == pygame.CONTROLLER_BUTTON_DPAD_DOWN:
                self._move(1)
            elif e.button == pygame.CONTROLLER_BUTTON_A:
                self._choose(self.sel)
            elif e.button == pygame.CONTROLLER_BUTTON_B:
                self.open = False
        return True

    def draw(self, screen, font, font_b, font_big):
        W, H = screen.get_size()
        shade = pygame.Surface((W, H), pygame.SRCALPHA)
        shade.fill((8, 10, 14, 170))
        screen.blit(shade, (0, 0))
        pw = min(640, W - 40)
        ph = 110 + ROW_H * len(self.items)
        x0, y0 = (W - pw) // 2, (H - ph) // 2
        pygame.draw.rect(screen, (18, 21, 27), (x0, y0, pw, ph), border_radius=12)
        pygame.draw.rect(screen, (45, 52, 62), (x0, y0, pw, ph), width=1, border_radius=12)
        screen.blit(font_big.render("Choose a robot", True, ACCENT), (x0 + 24, y0 + 18))
        self.rows, y = [], y0 + 62
        for i, it in enumerate(self.items):
            card = pygame.Rect(x0 + 16, y, pw - 32, ROW_H - 8)
            self.rows.append((i, card))
            sel = i == self.sel
            pygame.draw.rect(screen, (30, 38, 48) if sel else (24, 27, 33), card, border_radius=8)
            if sel:
                pygame.draw.rect(screen, ACCENT, card, width=2, border_radius=8)
            col = (240, 240, 240) if it["ok"] else (110, 115, 125)
            tag = f"{i + 1}" + ("   (current)" if it["name"] == self.current else "")
            screen.blit(font.render(tag, True, ACCENT if sel else (140, 150, 160)), (card.x + 14, card.y + 8))
            screen.blit(font_b.render(it["title"], True, col), (card.x + 14, card.y + 27))
            note = f"{it['dof']} joints" + ("" if it["source"] == "xml" else ", MuJoCo Menagerie model")
            if not it["ok"]:
                note = "model files missing (see THIRD_PARTY_NOTICES.md)"
            nw = font.size(note)[0]
            screen.blit(font.render(note, True, (150, 155, 165)), (card.right - nw - 14, card.y + 8))
            y += ROW_H
        hint = "Up/Down select   Enter choose   Esc keep current   (remembered in settings.yaml)"
        screen.blit(font.render(hint, True, (140, 145, 155)), (x0 + 24, y0 + ph - 30))
