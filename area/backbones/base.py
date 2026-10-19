"""What every backbone adapter shares: step tracking, the schedule and the per-step state."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from ..multi_gpu import to_device
from ..schedule import Schedule, build_schedule


@dataclass
class StepState:
    """The current denoising step, read by every attention processor of the transformer."""

    phase: Optional[str] = None          # None: plain attention, "coarse": Phase I, "fine": Phase II
    step: int = 0
    size: int = 0                        # pooling factor f (Phase I) or window side w (Phase II)
    h: int = 0
    w: int = 0
    txt_len: int = 0
    data: Dict[str, Any] = field(default_factory=dict)
    _copies: Dict[torch.device, Dict[str, Any]] = field(default_factory=dict, repr=False)

    def clear(self) -> None:
        """Start a forward pass: plain attention until the backbone sets a phase."""
        self.phase = None
        self.data = {}
        self._copies = {}

    def on(self, device: torch.device) -> Dict[str, Any]:
        """``data`` with its tensors on ``device``, copied once per forward pass for blocks placed on
        another GPU (see ``area.multi_gpu``). The backbone fills ``data`` before any block runs."""
        if device not in self._copies:
            self._copies[device] = to_device(self.data, device)
        return self._copies[device]


class Backbone:
    """Base adapter. A subclass wires AReA into one model family and sets its constants."""

    native: int                          # native training extent, in patches per side
    switch_sigma: float                  # default sigma*, the noise level where Phase II takes over

    def __init__(self, pipe, config):
        self.pipe = pipe
        self.config = config
        self.state = StepState()
        self.step_offset = 0
        self._schedule_key = None
        self._schedule: Optional[Schedule] = None

    @classmethod
    def supports(cls, pipe) -> bool:
        raise NotImplementedError

    def attach(self) -> None:
        raise NotImplementedError

    def detach(self) -> None:
        raise NotImplementedError

    def locate(self, timestep: torch.Tensor) -> Tuple[int, int]:
        """(step index, total steps) of the running pipeline call."""
        scheduler = self.pipe.scheduler
        current = getattr(self.pipe, "_current_timestep", None)
        if current is not None:
            ref, value = scheduler.timesteps.float().cpu(), float(current)
        else:
            ref, value = scheduler.sigmas[:-1].float().cpu(), float(timestep.flatten()[0])
        step = int(torch.argmin((ref - value).abs()))
        return step + self.step_offset, len(scheduler.timesteps) + self.step_offset

    def phase1_steps(self) -> int:
        """S_c: the steps of the running call whose noise level is above sigma*. Under AReA-Bootstrap
        the scheduler holds only the Phase II steps, and ``step_offset`` counts the replaced ones back in."""
        sigma = self.config.switch_sigma if self.config.switch_sigma is not None else self.switch_sigma
        scheduler = self.pipe.scheduler
        sigmas = scheduler.sigmas[: len(scheduler.timesteps)].float().cpu()
        return self.step_offset + int((sigmas > sigma).sum())

    def schedule(self, h: int, w: int, steps: int, txt_len: int = 0) -> Optional[Schedule]:
        """The schedule for this canvas, or None at or below the native extent (plain attention)."""
        if max(h, w) <= self.native:
            return None
        phase1 = self.phase1_steps()
        key = (h, w, steps, phase1, txt_len)
        if key != self._schedule_key:
            self._schedule = build_schedule(h, w, steps, phase1, self.native, txt_len=txt_len)
            self._schedule_key = key
        return self._schedule
