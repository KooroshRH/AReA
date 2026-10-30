"""AReA for Qwen-Image (``diffusers.QwenImagePipeline``)."""

import math

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenImageTransformer2DModel,
    apply_rotary_emb_qwen,
)

from .. import attention
from .base import TrueCfgBackbone, call_arg, shifted_scheduler

NATIVE_PATCHES = 83          # 1328 px / 16 px per patch
WINDOW_FLOOR = 64            # Phase II windows shrink to 1024 px
SWITCH_SIGMA = 0.87          # Phase II from step 20 of 40 at 4096^2
SHIFT = 2.0                  # flow-matching shift mu up to 4096^2 ...
SHIFT_TOKENS = 256 * 256     # ... and above, mu grows by 0.5 ln(tokens / SHIFT_TOKENS)


def _rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    return apply_rotary_emb_qwen(x, freqs, use_real=False)


class QwenAReAProcessor:
    """Attention processor for every Qwen-Image block; reads the step state of its backbone."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, backbone: "QwenBackbone", block: int):
        self.backbone = backbone
        self.block = block
        self.plain = QwenDoubleStreamAttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                 attention_mask=None, image_rotary_emb=None):
        st = self.backbone.state
        if st.phase is None:
            return self.plain(attn, hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                              attention_mask, image_rotary_emb)

        head_dim = attn.inner_dim // attn.heads
        query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (-1, head_dim)))
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (-1, head_dim)))
        value = attn.to_v(hidden_states).unflatten(-1, (-1, head_dim))
        tq = attn.norm_added_q(attn.add_q_proj(encoder_hidden_states).unflatten(-1, (-1, head_dim)))
        tk = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (-1, head_dim)))
        tv = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (-1, head_dim))
        query, key, value = torch.cat([tq, query], 1), torch.cat([tk, key], 1), torch.cat([tv, value], 1)

        t = st.txt_len
        data = st.on(query.device)
        memo = self.backbone.memo(self.block)
        if st.phase == "coarse":
            key, value = attention.pool_keys_values(
                key, value, t, st.h, st.w, st.size, self.backbone.config.pooling,
                seed=self.backbone.config.seed, step=st.step, block=self.block, memo=memo,
            )
            out = dispatch_attention_fn(_rotate(query, data["rope_q"]), _rotate(key, data["rope_kv"]), value,
                                        attn_mask=None, backend=self._attention_backend)
        else:
            img_freqs, txt_freqs = image_rotary_emb
            rope = torch.cat([txt_freqs, img_freqs], dim=0)
            q, k = _rotate(query, rope), _rotate(key, rope)
            out_txt, out_img = attention.windowed_attention(
                q[:, :t], k[:, :t], value[:, :t], q[:, t:], k[:, t:], value[:, t:], key[:, t:],
                data["layout"], backend=self._attention_backend, memo=memo,
            )
            out = torch.cat([out_txt, out_img], dim=1)
        out = out.flatten(2, 3).to(query.dtype)

        img_out = attn.to_out[0](out[:, t:].contiguous())
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        return img_out, attn.to_add_out(out[:, :t].contiguous())


class QwenBackbone(TrueCfgBackbone):
    """Qwen-Image runs true classifier-free guidance, so its two passes per step share the step's
    window layout and token selections (``TrueCfgBackbone``).

    Phase I pools onto a grid that fits the 1328 px native extent; Phase II windows shrink to
    1024 px. The noise schedule is shifted with mu = 2.0 up to 4096 x 4096 and by
    0.5 ln(tokens / 256^2) more above."""

    native = NATIVE_PATCHES
    switch_sigma = SWITCH_SIGMA
    window_floor = WINDOW_FLOOR

    @classmethod
    def supports(cls, pipe) -> bool:
        return isinstance(getattr(pipe, "transformer", None), QwenImageTransformer2DModel)

    def attach(self) -> None:
        tr = self.pipe.transformer
        self._saved_processors = tr.attn_processors
        for i, block in enumerate(tr.transformer_blocks):
            block.attn.set_processor(QwenAReAProcessor(self, i))
        self._hooks = [
            tr.register_forward_pre_hook(self._begin_forward, with_kwargs=True),
            tr.img_in.register_forward_hook(self.after_embedding),
        ]
        self._scheduler = self.pipe.scheduler
        self.pipe.scheduler = shifted_scheduler(self._scheduler, self._shift)
        self._ropes = {}

    def detach(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self.pipe.transformer.set_attn_processor(self._saved_processors)
        self.pipe.scheduler = self._scheduler
        self._ropes = {}
        self.forget_passes()
        self.state.clear()

    def _shift(self, mu: float) -> float:
        """mu for the canvas, from the pipeline's own mu, which is linear in the image token count."""
        cfg = self._scheduler.config
        slope = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len)
        tokens = round(cfg.base_image_seq_len + (mu - cfg.base_shift) / slope)
        return SHIFT + max(0.0, 0.5 * math.log(tokens / SHIFT_TOKENS))

    # ---- per forward -------------------------------------------------------

    def _begin_forward(self, module, args, kwargs):
        latents = call_arg(args, kwargs, "hidden_states", 0)
        text = call_arg(args, kwargs, "encoder_hidden_states", 1)
        mask = call_arg(args, kwargs, "encoder_hidden_states_mask", 2)
        shapes = call_arg(args, kwargs, "img_shapes", 4)
        if mask is not None and not bool(mask.all()):
            raise ValueError("AReA on Qwen-Image needs prompts of equal token length in one call; "
                             "generate one prompt at a time")
        frames, h, w = shapes[0][0]
        if len(shapes[0]) != 1 or frames != 1 or latents.shape[1] != h * w:
            raise ValueError("AReA on Qwen-Image supports text-to-image generation only")
        txt_len = int(text.shape[1])

        self.begin_forward(latents, call_arg(args, kwargs, "timestep", 3), h, w, txt_len)
        st = self.state
        if st.phase == "coarse":
            st.data["rope_q"], st.data["rope_kv"] = self._coarse_ropes(module, h, w, st.size, txt_len, latents.device)

    # ---- rotary embeddings -------------------------------------------------

    def _coarse_ropes(self, module, h, w, f, txt_len, device):
        """Queries keep full resolution and share the coarse position of their block; pooled keys
        sit on the ``(h/f) x (w/f)`` grid, and text follows that grid as it follows the image in
        Qwen-Image's own positions."""
        key = (h, w, f, txt_len, device)
        if key not in self._ropes:
            self._ropes = {k: v for k, v in self._ropes.items() if k[2] == f}
            nh, nw = h // f, w // f
            img, txt = module.pos_embed([(1, nh, nw)], device=device, max_txt_seq_len=txt_len)
            pos = torch.arange(h * w, device=device)
            block = (pos // w // f) * nw + pos % w // f
            self._ropes[key] = (torch.cat([txt, img[block]], dim=0), torch.cat([txt, img], dim=0))
        return self._ropes[key]
