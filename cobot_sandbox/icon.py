"""Window / Dock icon for the app, drawn in code (no image file). Research look: a white cobot arm
over a faint measurement grid on a blue lab tile, its gripper tracing a sampled trajectory curve
(data collection / learning). Preview: python -m cobot_sandbox.icon out.png 256"""
from __future__ import annotations

import math

import pygame

TOP, BOTTOM = (44, 82, 132), (24, 46, 82)        # tile gradient: mid "lab" blue, not near-black
GRID = (255, 255, 255, 28)                       # faint grid lines
AXIS = (255, 255, 255, 70)
TRACE = (255, 176, 60)                           # trajectory (orange)
SAMPLE = (255, 222, 160)
WHITE = (246, 246, 242)
LINE = (18, 32, 52)                              # outlines / joint caps / gripper
CYAN = (40, 205, 230)
S = 512                                          # drawing size; scaled down with smoothing
OUT = 8                                          # outline width at drawing size


def _tile() -> pygame.Surface:
    grad = pygame.Surface((S, S), pygame.SRCALPHA)
    for y in range(S):
        t = y / (S - 1)
        pygame.draw.line(grad, [round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM)] + [255], (0, y), (S, y))
    grid = pygame.Surface((S, S), pygame.SRCALPHA)       # measurement grid + plot axes
    for k in range(64, S, 48):
        pygame.draw.line(grid, GRID, (k, 24), (k, S - 24), 2)
        pygame.draw.line(grid, GRID, (24, k), (S - 24, k), 2)
    pygame.draw.line(grid, AXIS, (64, 448), (448, 448), 4)
    pygame.draw.line(grid, AXIS, (64, 448), (64, 64), 4)
    grad.blit(grid, (0, 0))
    mask = pygame.Surface((S, S), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (8, 8, S - 16, S - 16), border_radius=104)
    grad.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return grad


def _capsule(s, a, b, w, color):
    pygame.draw.line(s, color, a, b, w)
    for p in (a, b):
        pygame.draw.circle(s, color, p, w // 2)


def _trajectory():
    """Sampled curve the gripper has traced: rises from the lower right and ends at the gripper tip."""
    pts = []
    for i in range(9):
        t = i / 8
        x = 420 - 60 * t + 26 * math.sin(t * math.pi * 1.4)
        y = 418 - 150 * t + 34 * math.sin(t * math.pi * 2.2)
        pts.append((round(x), round(y)))
    return pts


def make_icon(size: int = 128) -> pygame.Surface:
    s = _tile()

    pts = _trajectory()                                   # trajectory: smooth line + sample dots
    pygame.draw.lines(s, TRACE, False, pts, 9)
    for p in pts[:-1]:
        pygame.draw.circle(s, LINE, p, 13)
        pygame.draw.circle(s, SAMPLE, p, 9)

    # fixed mount + arm, ending at the last trajectory sample
    pygame.draw.rect(s, LINE, (92 - OUT, 392 - OUT, 120 + 2 * OUT, 44 + 2 * OUT), border_radius=12 + OUT)
    pygame.draw.rect(s, WHITE, (92, 392, 120, 44), border_radius=12)
    pygame.draw.rect(s, CYAN, (92, 408, 120, 10))
    shoulder, elbow, wrist = (152, 350), (178, 190), (330, 168)
    for a, b, w in ((shoulder, elbow, 56), (elbow, wrist, 46)):
        _capsule(s, a, b, w + 2 * OUT, LINE)
    pygame.draw.rect(s, LINE, (126 - OUT, 340, 52 + 2 * OUT, 60), border_radius=10)
    pygame.draw.rect(s, WHITE, (126, 346, 52, 50), border_radius=8)
    _capsule(s, shoulder, elbow, 56, WHITE)
    _capsule(s, elbow, wrist, 46, WHITE)
    for p, r in ((shoulder, 36), (elbow, 32), (wrist, 28)):
        pygame.draw.circle(s, LINE, p, r + OUT)
        pygame.draw.circle(s, WHITE, p, r)
        pygame.draw.circle(s, LINE, p, int(r * 0.55))
    # gripper pointing down from the wrist toward the trajectory's end point
    tip = pts[-1]
    pygame.draw.rect(s, LINE, (wrist[0] - 12, wrist[1] + 22, 24, 22))
    pygame.draw.rect(s, LINE, (wrist[0] - 34, wrist[1] + 40, 68, 20), border_radius=6)
    pygame.draw.rect(s, LINE, (wrist[0] - 34, wrist[1] + 56, 14, 40), border_radius=5)
    pygame.draw.rect(s, LINE, (wrist[0] + 20, wrist[1] + 56, 14, 40), border_radius=5)
    pygame.draw.circle(s, LINE, tip, 14)                   # current sample, highlighted
    pygame.draw.circle(s, TRACE, tip, 10)
    return pygame.transform.smoothscale(s, (size, size))


if __name__ == "__main__":
    import sys
    pygame.init()
    pygame.image.save(make_icon(int(sys.argv[2]) if len(sys.argv) > 2 else 256), sys.argv[1] if len(sys.argv) > 1 else "icon.png")
