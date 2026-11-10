"""Keyboard + gamepad teleoperation (pygame). Produces the same Command as every other policy.

Keyboard
  W / S     TCP forward / back           Space   gripper open/close
  A / D     TCP left / right             Z       arm home
  R / F     TCP up / down                Shift   precision (slow) mode
  Q / E     rotate tool (yaw)
  Up/Down   tilt tool (pitch)            (Tab, switching robots, is handled by the app)
  Lt/Right  roll tool

Gamepad (SDL game-controller layout: Xbox / PlayStation / Switch Pro)
  Left stick   TCP x/y                          A / Cross     gripper
  Right stick  TCP z + tool yaw                 X / Square    home
  LB / RB      roll      D-pad up/down  pitch   LT (hold)     precision mode
"""
from __future__ import annotations

import numpy as np
import pygame

from .command import Command
from .policies.base import Policy

try:
    from pygame._sdl2 import controller as sdl_controller
except Exception:  # pragma: no cover
    sdl_controller = None

ARM_LIN, ARM_ANG = 0.15, 0.8        # m/s, rad/s at full deflection
PRECISION = 0.3
DEADZONE = 0.15


class Teleop(Policy):
    name = "teleop"

    def __init__(self):
        super().__init__()
        self.grip = 0.0
        self._home = False
        self.pad = None
        if sdl_controller is not None:
            try:
                sdl_controller.init()
                self._connect_pad()
            except Exception:
                pass

    # ------------------------------------------------------------------ devices
    def _connect_pad(self):
        if sdl_controller is None or self.pad is not None:
            return
        for i in range(sdl_controller.get_count()):
            if sdl_controller.is_controller(i):
                self.pad = sdl_controller.Controller(i)
                return

    @property
    def pad_name(self):
        if self.pad is None:
            return None
        try:
            return self.pad.name
        except Exception:
            return "gamepad"

    def _axis(self, a):
        v = self.pad.get_axis(a) / 32767.0
        return 0.0 if abs(v) < DEADZONE else float(np.clip((abs(v) - DEADZONE) / (1 - DEADZONE), 0, 1) * np.sign(v))

    # ------------------------------------------------------------------ events
    def toggle_gripper(self):
        self.grip = 0.0 if self.grip > 0.5 else 1.0

    def handle_event(self, e) -> bool:
        """Edge-triggered inputs. Returns True if the event was consumed."""
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                self.toggle_gripper()
            elif e.key == pygame.K_z:
                self._home = True
            else:
                return False
            return True
        if e.type in (getattr(pygame, "CONTROLLERDEVICEADDED", -1),):
            self._connect_pad()
            return True
        if e.type in (getattr(pygame, "CONTROLLERDEVICEREMOVED", -1),):
            self.pad = None
            self._connect_pad()
            return True
        if e.type == getattr(pygame, "CONTROLLERBUTTONDOWN", -1):
            b = e.button
            if b == pygame.CONTROLLER_BUTTON_A:
                self.toggle_gripper()
            elif b == pygame.CONTROLLER_BUTTON_X:
                self._home = True
            else:
                return False
            return True
        return False

    def is_active_input(self) -> bool:
        """True while the human is pressing a motion input (used to take over from autopilot)."""
        return bool(np.any(self._motion()[0]) or np.any(self._motion()[1]))

    # ------------------------------------------------------------------ continuous input
    def _motion(self):
        k = pygame.key.get_pressed()
        ax = lambda pos, neg: float(k[pos]) - float(k[neg])
        lin = np.array([ax(pygame.K_w, pygame.K_s), ax(pygame.K_a, pygame.K_d), ax(pygame.K_r, pygame.K_f)])
        ang = np.array([ax(pygame.K_RIGHT, pygame.K_LEFT), ax(pygame.K_DOWN, pygame.K_UP), ax(pygame.K_q, pygame.K_e)])
        slow = bool(k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT])
        if self.pad is not None:
            try:
                A = pygame
                lin += [-self._axis(A.CONTROLLER_AXIS_LEFTY), -self._axis(A.CONTROLLER_AXIS_LEFTX),
                        -self._axis(A.CONTROLLER_AXIS_RIGHTY)]
                ang += [float(self.pad.get_button(A.CONTROLLER_BUTTON_RIGHTSHOULDER))
                        - float(self.pad.get_button(A.CONTROLLER_BUTTON_LEFTSHOULDER)),
                        float(self.pad.get_button(A.CONTROLLER_BUTTON_DPAD_DOWN))
                        - float(self.pad.get_button(A.CONTROLLER_BUTTON_DPAD_UP)),
                        -self._axis(A.CONTROLLER_AXIS_RIGHTX)]
                slow = slow or self.pad.get_axis(A.CONTROLLER_AXIS_TRIGGERLEFT) > 16000
            except Exception:
                self.pad = None
        return np.clip(lin, -1, 1), np.clip(ang, -1, 1), slow

    def act(self, obs, sim) -> Command:
        lin, ang, slow = self._motion()
        s = PRECISION if slow else 1.0
        cmd = Command(ee_lin=lin * ARM_LIN * s, ee_ang=ang * ARM_ANG * s, gripper=self.grip, home=self._home)
        self._home = False
        self.status = "manual" + (" (precision)" if slow else "")
        return cmd
