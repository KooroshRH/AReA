"""AReA for FLUX.1 (``diffusers.FluxPipeline``)."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

from .. import attention, tiling
from .base import Backbone

NATIVE_PATCHES = 64          # 1024 px / 16 px per patch
SWITCH_SIGMA = 0.89          # Phase II from step 8 of 28
BOOTSTRAP_SIDE = 1024


def _qkv(attn, hidden_states, encoder_hidden_states):
    """Normalised, unrotated Q/K/V as ``[B, S, heads, head_dim]``, text first on dual-stream blocks."""
    if getattr(attn, "fused_projections", False):
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    else:
        query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
    query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim)))
    key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim)))
    value = value.unflatten(-1, (-1, attn.head_dim))
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        if getattr(attn, "fused_projections", False):
            eq, ek, ev = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
        else:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
        eq = attn.norm_added_q(eq.unflatten(-1, (-1, attn.head_dim)))
        ek = attn.norm_added_k(ek.unflatten(-1, (-1, attn.head_dim)))
        ev = ev.unflatten(-1, (-1, attn.head_dim))
        query, key, value = torch.cat([eq, query], 1), torch.cat([ek, key], 1), torch.cat([ev, value], 1)
    return query, key, value


class FluxAReAProcessor:
    """Attention processor for every FLUX.1 block; reads the step state of its backbone."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, backbone: "FluxBackbone", block: int):
        self.backbone = backbone
        self.block = block
        self.plain = FluxAttnProcessor()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        st = self.backbone.state
        if st.phase is None:
            return self.plain(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        query, key, value = _qkv(attn, hidden_states, encoder_hidden_states)
        data = st.on(query.device)
        t = st.txt_len
        if st.phase == "coarse":
            key, value = attention.pool_keys_values(
                key, value, t, st.h, st.w, st.size, self.backbone.config.pooling,
                seed=self.backbone.config.seed, step=st.step, block=self.block,
            )
            query = apply_rotary_emb(query, data["rope_q"], sequence_dim=1)
            key = apply_rotary_emb(key, data["rope_kv"], sequence_dim=1)
            out = dispatch_attention_fn(query, key, value, attn_mask=None, backend=self._attention_backend)
        else:
            q = apply_rotary_emb(query, data["rope"], sequence_dim=1)
            k = apply_rotary_emb(key, data["rope"], sequence_dim=1)
            out_txt, out_img = attention.windowed_attention(
                q[:, :t], k[:, :t], value[:, :t], q[:, t:], k[:, t:], value[:, t:], key[:, t:],
                data["layout"], backend=self._attention_backend,
            )
            out = torch.cat([out_txt, out_img], dim=1)
        out = out.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            txt_out, img_out = out.split_with_sizes([t, out.shape[1] - t], dim=1)
            img_out = attn.to_out[1](attn.to_out[0](img_out.contiguous()))
            return img_out, attn.to_add_out(txt_out.contiguous())
        return out


def _grid(img_ids: torch.Tensor) -> Tuple[int, int]:
    """Patch grid ``(h, w)`` of row-major image ids. Rows 0 and 1 are exact in any dtype, so the
    width is where the row id first changes."""
    rows = (img_ids[0] if img_ids.ndim == 3 else img_ids)[:, 1]
    n = int(rows.numel())
    changed = torch.nonzero(rows != rows[0])
    w = int(changed[0]) if changed.numel() else n
    return n // w, w


class FluxBackbone(Backbone):
    native = NATIVE_PATCHES
    switch_sigma = SWITCH_SIGMA

    @classmethod
    def supports(cls, pipe) -> bool:
        return type(getattr(pipe, "transformer", None)).__name__ == "FluxTransformer2DModel"

    def attach(self) -> None:
        tr = self.pipe.transformer
        self._saved_processors = tr.attn_processors
        blocks = list(tr.transformer_blocks) + list(tr.single_transformer_blocks)
        for i, block in enumerate(blocks):
            block.attn.set_processor(FluxAReAProcessor(self, i))
        self._hooks = [
            tr.register_forward_pre_hook(self._begin_step, with_kwargs=True),
            tr.x_embedder.register_forward_hook(self._after_embedding),
        ]
        self._scheduler = self.pipe.scheduler
        self.pipe.scheduler = type(self._scheduler).from_config(
            self._scheduler.config, base_shift=self._scheduler.config.max_shift)
        self._ropes = {}

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self.pipe.transformer.set_attn_processor(self._saved_processors)
        self.pipe.scheduler = self._scheduler
        self._ropes = {}
        self.state.clear()

    # ---- per step ----------------------------------------------------------

    def _begin_step(self, module, args, kwargs):
        timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
        img_ids = kwargs.get("img_ids", args[4] if len(args) > 4 else None)
        text = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        h, w = _grid(img_ids)
        step, total = self.locate(timestep)
        schedule = self.schedule(h, w, total, int(text.shape[1]))
        st = self.state
        st.clear()
        if schedule is None:
            return
        st.phase, st.size = schedule.at(step)
        st.step, st.h, st.w, st.txt_len = step, h, w, int(text.shape[1])
        device = img_ids.device
        if st.phase == "coarse":
            st.data["rope_q"], st.data["rope_kv"] = self._coarse_ropes(h, w, st.size, st.txt_len, device)
        else:
            st.data["rope"] = self._grid_rope(h, w, st.txt_len, device)
            st.data["layout"] = None

    def _after_embedding(self, module, args, output):
        st = self.state
        if st.phase != "fine":
            return
        cfg = self.config
        w_min = min(self.native, max(st.h, st.w))
        energy = tiling.detail_map(output, st.h, st.w)
        windows = tiling.adaptive_tiling(
            energy, st.h, st.w, st.size, w_min, st.txt_len, rng=tiling.step_rng(cfg.seed, st.step, 0x5EA1))
        registers = max(1, int(round(st.size * cfg.bridge_ratio))) ** 2 if cfg.bridge_ratio > 0 else 0
        widths = tiling.halo_widths(windows, st.h, st.w, cfg.halo, tiling.step_rng(cfg.seed, st.step, 0x4A11))
        # never cost more than a full-attention step: cut bridge tokens, then the Halo
        registers, widths = attention.fit_extra_keys(windows, st.h, st.w, st.txt_len, registers, widths)
        st.data["layout"] = attention.build_fine_layout(
            windows, st.h, st.w, widths, cfg.halo, registers, cfg.seed, st.step, output.dtype, output.device)

    # ---- rotary embeddings -------------------------------------------------

    def _rope(self, txt_len: int, rows: torch.Tensor, cols: torch.Tensor):
        ids = torch.zeros(txt_len + rows.numel(), 3, device=rows.device, dtype=torch.float32)
        ids[txt_len:, 1] = rows.float()
        ids[txt_len:, 2] = cols.float()
        return self.pipe.transformer.pos_embed(ids)

    def _grid_rope(self, h, w, txt_len, device):
        key = ("grid", h, w, txt_len, device)
        if self._ropes.get("grid", (None,))[0] != key:
            pos = torch.arange(h * w, device=device)
            self._ropes["grid"] = (key, self._rope(txt_len, pos // w, pos % w))
        return self._ropes["grid"][1]

    def _coarse_ropes(self, h, w, f, txt_len, device):
        """Queries keep full resolution and share the coarse position of their block; pooled keys
        sit on the ``(h/f) x (w/f)`` grid, so both are measured in the same units."""
        key = ("coarse", h, w, f, txt_len, device)
        if self._ropes.get("coarse", (None,))[0] != key:
            pos = torch.arange(h * w, device=device)
            coarse = torch.arange((h // f) * (w // f), device=device)
            rope_q = self._rope(txt_len, pos // w // f, pos % w // f)
            rope_kv = self._rope(txt_len, coarse // (w // f), coarse % (w // f))
            self._ropes["coarse"] = (key, (rope_q, rope_kv))
        return self._ropes["coarse"][1]

    # ---- bootstrap ---------------------------------------------------------

    @torch.no_grad()
    def bootstrap(self, prompt, height: int, width: int, num_inference_steps: int = 28,
                  guidance_scale: float = 4.5, seed: int = 0, **kwargs):
        """AReA-Bootstrap (Appendix F): Phase I is replaced by a generation at the native resolution
        whose latent is upsampled to the target size and re-noised to the level where Phase II
        begins; Phase II then runs unchanged."""
        pipe = self.pipe
        device = pipe._execution_device
        base_sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        mu = pipe.scheduler.config.max_shift
        pipe.scheduler.set_timesteps(sigmas=base_sigmas, mu=mu, device=device)
        phase1 = self.phase1_steps()
        if not 0 < phase1 < num_inference_steps:
            raise ValueError("bootstrap needs a switch_sigma with denoising steps on both sides of it")
        dtype = pipe.transformer.dtype
        scale = min(1.0, BOOTSTRAP_SIDE / max(height, width))
        m = pipe.vae_scale_factor * 2
        low_h = max(m, int(round(height * scale)) // m * m)
        low_w = max(m, int(round(width * scale)) // m * m)

        generator = torch.Generator(device).manual_seed(seed)
        low = pipe(prompt, height=low_h, width=low_w, num_inference_steps=num_inference_steps,
                   guidance_scale=guidance_scale, generator=generator, output_type="latent",
                   return_dict=False, **kwargs)[0]
        latents = pipe._unpack_latents(low, low_h, low_w, pipe.vae_scale_factor)
        lat_h, lat_w = 2 * (height // m), 2 * (width // m)
        latents = F.interpolate(latents, size=(lat_h, lat_w), mode="bicubic", align_corners=False)
        clean = pipe._pack_latents(latents, latents.shape[0], latents.shape[1], lat_h, lat_w)
        clean = clean.to(device=device, dtype=dtype)

        pipe.scheduler.set_timesteps(sigmas=base_sigmas, mu=mu, device=device)  # the low-resolution run reset it
        start = pipe.scheduler.timesteps[phase1].reshape(1)
        generator.manual_seed(seed)
        noise = torch.randn(clean.shape, generator=generator, device=device, dtype=dtype)
        noisy = pipe.scheduler.scale_noise(clean, start, noise)

        self.step_offset = phase1
        try:
            return pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps - phase1,
                        sigmas=list(base_sigmas[phase1:]), latents=noisy, guidance_scale=guidance_scale,
                        generator=generator, **kwargs)
        finally:
            self.step_offset = 0
