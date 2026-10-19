"""AReA for Ideogram 4 (``diffusers.Ideogram4Pipeline``)."""

from typing import Optional, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_ideogram4 import IMAGE_POSITION_OFFSET, Ideogram4Transformer2DModel

from .. import attention, tiling
from ..pipelines.ideogram4 import SlimIdeogram4Transformer2DModel
from ..schedule import Schedule, build_schedule
from .base import Backbone

NATIVE_PATCHES = 128         # 2048 px / 16 px per patch
NATIVE_UP_TO_4K = 64         # canvases up to 4096 px anchor on 1024 px
SWITCH_SIGMA = 0.92          # Phase II from step 20 of 48 at 4096^2


def _rotate(x: torch.Tensor, rope: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Ideogram 4's half-split rotation of ``[B, L, H, D]`` by ``[B, L, D]`` cos/sin."""
    cos, sin = rope[0].to(x.dtype).unsqueeze(2), rope[1].to(x.dtype).unsqueeze(2)
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


class IdeogramAReAProcessor:
    """Attention processor for every Ideogram 4 block; reads the step state of its backbone."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, backbone: "IdeogramBackbone", block: int):
        self.backbone = backbone
        self.block = block

    def __call__(self, attn, hidden_states, attention_mask, image_rotary_emb):
        # the slim pipeline trims the text padding, so every token shares one segment and there is no mask
        st = self.backbone.state
        query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim)))
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim)))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        t = st.txt_len
        data = st.on(query.device)

        if st.phase is None:
            out = dispatch_attention_fn(_rotate(query, image_rotary_emb), _rotate(key, image_rotary_emb), value,
                                        attn_mask=None, backend=self._attention_backend)
        elif st.phase == "coarse":
            key, value = attention.pool_keys_values(
                key, value, t, st.h, st.w, st.size, self.backbone.config.pooling,
                seed=self.backbone.config.seed, step=st.step, block=self.block,
            )
            out = dispatch_attention_fn(_rotate(query, data["rope_q"]), _rotate(key, data["rope_kv"]), value,
                                        attn_mask=None, backend=self._attention_backend)
        else:
            q = _rotate(query, image_rotary_emb)
            k = _rotate(key, image_rotary_emb)
            out_txt, out_img = attention.windowed_attention(
                q[:, :t], k[:, :t], value[:, :t], q[:, t:], k[:, t:], value[:, t:], key[:, t:],
                data["layout"], backend=self._attention_backend,
            )
            out = torch.cat([out_txt, out_img], dim=1)
        return attn.to_out[0](out.flatten(2, 3).to(query.dtype))


class IdeogramBackbone(Backbone):
    """Ideogram 4 packs ``[text | image]`` into one sequence and runs a second, image-only
    transformer for the unconditional branch; both get AReA, and both share one window layout
    per step. Needs ``area.pipelines.SlimIdeogram4Pipeline``.

    The last step of Ideogram 4's schedule starts from a high noise level and carries a large
    share of the denoising, so it runs with the start window ``w_start`` of Phase II, without DWT
    splits, instead of the last rung: small windows there settle their brightness independently
    and leave window-shaped blocks."""

    native = NATIVE_PATCHES
    switch_sigma = SWITCH_SIGMA

    @classmethod
    def supports(cls, pipe) -> bool:
        return isinstance(getattr(pipe, "transformer", None), Ideogram4Transformer2DModel)

    def attach(self) -> None:
        self._transformers = [t for t in (self.pipe.transformer, getattr(self.pipe, "unconditional_transformer", None))
                              if t is not None]
        if not all(isinstance(t, SlimIdeogram4Transformer2DModel) for t in self._transformers):
            raise ValueError("AReA on Ideogram 4 needs area.pipelines.SlimIdeogram4Pipeline (same checkpoints); the "
                             "stock pipeline's text features span the whole image sequence and do not fit in memory "
                             "at high resolution")
        self._saved_processors = [t.attn_processors for t in self._transformers]
        self._hooks = []
        for tr in self._transformers:
            for i, block in enumerate(tr.layers):
                block.attention.set_processor(IdeogramAReAProcessor(self, i))
            self._hooks.append(tr.register_forward_pre_hook(self._begin_forward, with_kwargs=True))
        self._text_len = 0
        self._last = False
        self._layout_key = None
        self._ropes = {}

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        for tr, saved in zip(self._transformers, self._saved_processors):
            tr.set_attn_processor(saved)
        self._ropes = {}
        self._layout_key = None
        self.state.clear()

    # ---- geometry ----------------------------------------------------------

    def anchor(self, h: int, w: int) -> Optional[int]:
        """Extent the schedule anchors on: 1024 px up to 4096 px canvases, the 2048 px native above,
        and None (plain attention) at or below native."""
        extent = max(h, w)
        if extent <= NATIVE_PATCHES:
            return None
        return NATIVE_UP_TO_4K if extent <= 4 * NATIVE_UP_TO_4K else NATIVE_PATCHES

    def schedule(self, h: int, w: int, steps: int, txt_len: int = 0) -> Optional[Schedule]:
        anchor = self.anchor(h, w)
        if anchor is None:
            return None
        phase1 = self.phase1_steps()
        key = (h, w, steps, phase1, txt_len)
        if key != self._schedule_key:
            self._schedule = build_schedule(h, w, steps, phase1, anchor, txt_len=txt_len)
            self._schedule_key = key
        return self._schedule

    def locate(self, timestep: torch.Tensor) -> Tuple[int, int]:
        scheduler = self.pipe.scheduler
        return int(scheduler.step_index or 0), len(scheduler.timesteps)

    # ---- per forward -------------------------------------------------------

    def _begin_forward(self, module, args, kwargs):
        if args:
            raise ValueError("AReA expects the Ideogram 4 transformer to be called with keyword arguments")
        segments = kwargs["segment_ids"]
        if not bool((segments == segments[:, :1]).all()):
            raise ValueError("AReA on Ideogram 4 needs prompts of equal token length in one call; "
                             "generate one prompt at a time")

        positions = kwargs["position_ids"]
        image = positions[0, :, 0] == IMAGE_POSITION_OFFSET
        txt_len = int(image.to(torch.long).argmax())
        img_pos = positions[0, txt_len:]
        h = int(img_pos[:, 1].max()) - IMAGE_POSITION_OFFSET + 1
        w = int(img_pos[:, 2].max()) - IMAGE_POSITION_OFFSET + 1
        if module is self.pipe.transformer:
            self._text_len = txt_len

        step, total = self.locate(kwargs["timestep"])
        schedule = self.schedule(h, w, total, self._text_len)
        st = self.state
        st.clear()
        if schedule is None:
            return
        phase, size = schedule.at(step)
        self._last = phase == "fine" and step == total - 1
        if self._last:
            size = schedule.windows[0]
        st.phase, st.size = phase, size
        st.step, st.h, st.w, st.txt_len = step, h, w, txt_len
        if phase == "coarse":
            st.data["rope_q"], st.data["rope_kv"] = self._coarse_ropes(
                module, positions, h, w, size, txt_len, kwargs["hidden_states"].dtype)
        else:
            st.data["layout"] = self._layout(kwargs["hidden_states"][:, txt_len:], h, w, step, size)

    def _layout(self, latents: torch.Tensor, h: int, w: int, step: int, size: int):
        """One window layout per step, built from the noisy latents both transformers receive."""
        key = (step, h, w, size, self._text_len)
        if self._layout_key == key:
            return self._layout_cache
        cfg = self.config
        rng = tiling.step_rng(cfg.seed, step, 0x5EA1)
        if self._last:
            # the last step keeps every start window whole; a DWT split would hand the smooth regions,
            # where window brightness shows most, back to small windows
            rows, cols = tiling.jittered_starts(h, size, rng), tiling.jittered_starts(w, size, rng)
            windows = [tiling.Window(r, c, min(h - 1, r + size - 1), min(w - 1, c + size - 1))
                       for r in rows for c in cols]
        else:
            w_min = min(self.anchor(h, w), max(h, w))
            energy = tiling.detail_map(latents, h, w)
            windows = tiling.adaptive_tiling(energy, h, w, size, w_min, self._text_len, rng=rng)
        registers = max(1, int(round(size * cfg.bridge_ratio))) ** 2 if cfg.bridge_ratio > 0 else 0
        widths = tiling.halo_widths(windows, h, w, cfg.halo, tiling.step_rng(cfg.seed, step, 0x4A11))
        registers, widths = attention.fit_extra_keys(windows, h, w, self._text_len, registers, widths)
        self._layout_cache = attention.build_fine_layout(
            windows, h, w, widths, cfg.halo, registers, cfg.seed, step, latents.dtype, latents.device)
        self._layout_key = key
        return self._layout_cache

    # ---- rotary embeddings -------------------------------------------------

    def _coarse_ropes(self, module, positions, h, w, f, txt_len, dtype):
        """Queries keep full resolution and share the coarse position of their block; pooled keys
        sit on the ``(h/f) x (w/f)`` grid. Text keeps its own positions."""
        key = (h, w, f, txt_len, positions.device, dtype)
        if key not in self._ropes:
            self._ropes = {k: v for k, v in self._ropes.items() if k[2] == f}
            device = positions.device
            text = positions[:1, :txt_len]
            pos = torch.arange(h * w, device=device)
            nh, nw = h // f, w // f
            coarse = torch.arange(nh * nw, device=device)

            def image(rows, cols):
                ids = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1) + IMAGE_POSITION_OFFSET
                return torch.cat([text, ids.unsqueeze(0).to(text.dtype)], dim=1)

            cast = lambda rope: tuple(x.to(dtype) for x in rope)
            self._ropes[key] = (cast(module.rotary_emb(image(pos // w // f, pos % w // f))),
                                cast(module.rotary_emb(image(coarse // nw, coarse % nw))))
        return self._ropes[key]
