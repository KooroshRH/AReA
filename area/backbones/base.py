"""What every backbone adapter shares: step tracking, the schedule and the per-step state."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from .. import attention, tiling
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
    window_floor: Optional[int] = None   # smallest Phase II window w_min; None: the native extent

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
            self._schedule = build_schedule(h, w, steps, phase1, self.native, self.window_floor, txt_len=txt_len)
            self._schedule_key = key
        return self._schedule

    def fine_layout(self, tokens: torch.Tensor, h: int, w: int, step: int, size: int, txt_len: int):
        """The Phase II layout of one step: windows of side ``size`` split by the DWT detail map of
        the embedded image ``tokens``, their Halo widths and the bridge count."""
        cfg = self.config
        w_min = min(self.native, max(h, w))
        energy = tiling.detail_map(tokens, h, w)
        windows = tiling.adaptive_tiling(
            energy, h, w, size, w_min, txt_len, rng=tiling.step_rng(cfg.seed, step, 0x5EA1))
        registers = max(1, int(round(size * cfg.bridge_ratio))) ** 2 if cfg.bridge_ratio > 0 else 0
        widths = tiling.halo_widths(windows, h, w, cfg.halo, tiling.step_rng(cfg.seed, step, 0x4A11))
        # never cost more than a full-attention step: cut bridge tokens, then the Halo
        registers, widths = attention.fit_extra_keys(windows, h, w, txt_len, registers, widths)
        return attention.build_fine_layout(
            windows, h, w, widths, cfg.halo, registers, cfg.seed, step, tokens.dtype, tokens.device)


class TrueCfgBackbone(Backbone):
    """Adapter for pipelines with true classifier-free guidance, which run the transformer twice per
    step (prompt and negative prompt) on the same latents tensor. The second pass reuses the schedule,
    the Phase II layout and the token selections of the first, so both passes run the same attention
    operator and the guidance difference reflects only the prompt.

    A subclass calls ``begin_forward`` from a forward pre-hook on the transformer and registers
    ``after_embedding`` as a forward hook on its image embedding."""

    def __init__(self, pipe, config):
        super().__init__(pipe, config)
        self.forget_passes()

    def begin_forward(self, latents: torch.Tensor, timestep: torch.Tensor, h: int, w: int, txt_len: int) -> None:
        """Set the step state for one transformer forward on an ``h x w`` patch grid."""
        st = self.state
        step, total = self.locate(timestep)
        if self._pass is None or latents is not self._pass[0] or step != self._pass[1]:
            self.forget_passes()
            self._pass = (latents, step)
            schedule = self.schedule(h, w, total, txt_len)
            st.clear()
            if schedule is None:
                return
            st.phase, st.size = schedule.at(step)
            st.step, st.h, st.w = step, h, w
        else:
            phase = st.phase
            st.clear()
            st.phase = phase
            if phase is None:
                return
        st.txt_len = txt_len
        if st.phase == "fine":
            st.data["layout"] = self._step_layout

    def after_embedding(self, module, args, output) -> None:
        """The first pass of a Phase II step builds the layout from the embedded image tokens."""
        st = self.state
        if st.phase == "fine" and self._step_layout is None:
            self._step_layout = st.data["layout"] = self.fine_layout(output, st.h, st.w, st.step, st.size, st.txt_len)

    def memo(self, block: int) -> dict:
        """Token selections of one block in the current step, recorded by the first pass."""
        return self._memo.setdefault(block, {})

    def forget_passes(self) -> None:
        self._pass, self._memo, self._step_layout = None, {}, None


def call_arg(args, kwargs, name: str, index: int):
    """Argument ``name`` of a forward call, passed by keyword or at position ``index``."""
    return kwargs[name] if name in kwargs else (args[index] if len(args) > index else None)


def shifted_scheduler(scheduler, shift):
    """A copy of ``scheduler`` that sets its timesteps with ``mu = shift(mu_pipeline)``, where
    ``mu_pipeline`` is the shift the pipeline computed. For pipelines that offer no ``mu`` argument."""
    base = type(scheduler)

    class Shifted(base):
        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
            return super().set_timesteps(num_inference_steps, device, sigmas=sigmas,
                                         mu=None if mu is None else shift(mu), timesteps=timesteps)

    # named like the stock class, so the pipeline config, which records each component's library and
    # class, is unchanged while AReA is attached
    Shifted.__module__, Shifted.__name__, Shifted.__qualname__ = base.__module__, base.__name__, base.__qualname__
    return Shifted.from_config(scheduler.config)
