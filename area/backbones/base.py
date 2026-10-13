"""What every backbone adapter shares: step tracking, the schedule and the per-step state."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

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


class Backbone:
    """Base adapter. A subclass wires AReA into one model family and sets its constants."""

    native: int                          # native training extent, in patches per side
    phase1_steps: int                    # default number of Phase I steps

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

    def schedule(self, h: int, w: int, steps: int, txt_len: int = 0) -> Optional[Schedule]:
        """The schedule for this canvas, or None at or below the native extent (plain attention)."""
        if max(h, w) <= self.native:
            return None
        phase1 = self.config.phase1_steps if self.config.phase1_steps is not None else self.phase1_steps
        key = (h, w, steps, phase1, txt_len)
        if key != self._schedule_key:
            self._schedule = build_schedule(h, w, steps, phase1, self.native, txt_len=txt_len)
            self._schedule_key = key
        return self._schedule
