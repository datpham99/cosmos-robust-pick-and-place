"""Keyboard teleop device that treats a key as held while presses keep arriving.

Isaac Lab's Se3Keyboard adds a delta on KEY_PRESS and removes it on KEY_RELEASE. Over a remote desktop
session (TeamViewer on X11) a held key arrives as rapid PRESS/RELEASE pairs with no KEY_REPEAT, so the
delta cancels before the env reads it and the arm never moves. Here a movement key counts as held for
``hold_s`` seconds after its most recent PRESS or REPEAT; releases are ignored. Works with a physical
keyboard too, since a real hold emits KEY_REPEAT events.
"""

from __future__ import annotations

import time

import carb.input
import numpy as np
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.configclass import configclass

_POS_KEYS = ("W", "S", "A", "D", "Q", "E")
_ROT_KEYS = ("Z", "X", "T", "G", "C", "V")


class PulseKeyboard(Se3Keyboard):
    def __init__(self, cfg: PulseKeyboardCfg):
        self._hold_s = cfg.hold_s
        self._last_press: dict[str, float] = {}
        super().__init__(cfg)

    def reset(self):
        super().reset()
        self._last_press = {}

    def _on_keyboard_event(self, event, *args, **kwargs):
        et = carb.input.KeyboardEventType
        if event.type in (et.KEY_PRESS, et.KEY_REPEAT):
            name = event.input.name
            if name in _POS_KEYS or name in _ROT_KEYS:
                self._last_press[name] = time.monotonic()
            if event.type == et.KEY_PRESS:
                if name == "L":
                    self.reset()
                elif name == "K":
                    self._close_gripper = not self._close_gripper
                if name in self._additional_callbacks:
                    self._additional_callbacks[name]()
        return True

    def advance(self) -> torch.Tensor:
        now = time.monotonic()
        held = [k for k, t in self._last_press.items() if now - t < self._hold_s]
        self._delta_pos = sum((self._INPUT_KEY_MAPPING[k] for k in held if k in _POS_KEYS), np.zeros(3))
        self._delta_rot = sum((self._INPUT_KEY_MAPPING[k] for k in held if k in _ROT_KEYS), np.zeros(3))
        return super().advance()


@configclass
class PulseKeyboardCfg(Se3KeyboardCfg):
    hold_s: float = 0.15
    class_type: type = PulseKeyboard
