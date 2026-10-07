"""Backbone-independent parts of the two AReA attention phases.

Tensors follow the diffusers layout ``[batch, sequence, heads, head_dim]`` with the
sequence ordered ``[text | image]`` and image tokens row-major on an ``h x w`` grid.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn

from . import bridge, tiling

POOLING = ("subsample", "mean", "medoid", "random")
_MEDOID_CHUNK_ELEMS = 8_000_000


# ---------------------------------------------------------------------------
# Phase I: cross-scale coarse attention
# ---------------------------------------------------------------------------


def _block_selection(k_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
                     seed: int, step: int, block: int) -> Optional[torch.Tensor]:
    """Index in ``[0, f*f)`` of the token that represents each block, for the selecting poolings."""
    b, _, heads, dim = k_img.shape
    nh, nw = h // f, w // f
    if pooling == "random":
        mix = (int(seed) & 0x7FFFFFFF) + int(step) * 2_654_435_761 + int(block) * 40_503 + 0xC0DA
        sel = np.random.default_rng(mix).integers(0, f * f, size=(nh, nw), dtype=np.int64)
        return torch.from_numpy(sel).to(k_img.device).unsqueeze(0).expand(b, nh, nw).contiguous()
    if pooling == "medoid":
        grid = k_img.reshape(b, h, w, heads, dim)
        rows_per_chunk = max(1, _MEDOID_CHUNK_ELEMS // max(1, nw * f * f * heads * dim))
        parts = []
        for r0 in range(0, nh, rows_per_chunk):
            r1 = min(nh, r0 + rows_per_chunk)
            kb = (grid[:, r0 * f : r1 * f]
                  .reshape(b, r1 - r0, f, nw, f, heads, dim)
                  .permute(0, 1, 3, 2, 4, 5, 6)
                  .reshape(b, r1 - r0, nw, f * f, heads, dim)
                  .float())
            parts.append((kb - kb.mean(dim=3, keepdim=True)).pow(2).sum(dim=(-2, -1)).argmin(dim=3))
        return torch.cat(parts, dim=1)
    return None


def _pool(x_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
          sel: Optional[torch.Tensor]) -> torch.Tensor:
    b, _, heads, dim = x_img.shape
    nh, nw = h // f, w // f
    grid = x_img.reshape(b, h, w, heads, dim)
    if pooling == "subsample":
        off = f // 2
        pooled = grid[:, off::f, off::f][:, :nh, :nw]
    elif pooling == "mean":
        pooled = grid.reshape(b, nh, f, nw, f, heads, dim).mean(dim=(2, 4))
    else:
        blocks = (grid.reshape(b, nh, f, nw, f, heads, dim)
                  .permute(0, 1, 3, 2, 4, 5, 6)
                  .reshape(b, nh, nw, f * f, heads, dim))
        idx = sel[:, :, :, None, None, None].expand(b, nh, nw, 1, heads, dim)
        pooled = blocks.gather(3, idx).squeeze(3)
    return pooled.reshape(b, nh * nw, heads, dim)


def pool_keys_values(key: torch.Tensor, value: torch.Tensor, txt_len: int, h: int, w: int, f: int,
                     pooling: str, seed: int = 0, step: int = 0, block: int = 0
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep one token per ``f x f`` block of the image keys and values (text is never pooled).

    ``subsample`` keeps the block centre, ``mean`` averages the block, ``medoid`` keeps the token
    whose key is closest to the block-mean key, and ``random`` keeps a token drawn per
    (seed, step, block). Keys must be unrotated; K and V always come from the same token.
    """
    k_img, v_img = key[:, txt_len:], value[:, txt_len:]
    sel = _block_selection(k_img, h, w, f, pooling, seed, step, block)
    k = torch.cat([key[:, :txt_len], _pool(k_img, h, w, f, pooling, sel)], dim=1)
    v = torch.cat([value[:, :txt_len], _pool(v_img, h, w, f, pooling, sel)], dim=1)
    return k, v


# ---------------------------------------------------------------------------
# Phase II: adaptive windowed fine attention
# ---------------------------------------------------------------------------


@dataclass
class WindowGroup:
    """Windows of one step that share a token count and a Halo size, run as one batched call."""

    tokens: torch.Tensor                 # [G, n]
    halo: Optional[torch.Tensor]         # [G, n_halo]
    weights: torch.Tensor                # [G * n] merge weights


@dataclass
class FineLayout:
    """Everything Phase II needs for one denoising step, shared by all transformer blocks."""

    h: int
    w: int
    windows: List[tiling.Window]
    groups: List[WindowGroup]
    coverage: torch.Tensor               # [N] number of windows covering each token
    registers: int                       # R bridge tokens per window (0 = no bridge)


def build_fine_layout(windows: List[tiling.Window], h: int, w: int, halo: int, registers: int,
                      seed: int, step: int, dtype: torch.dtype, device) -> FineLayout:
    n_img = h * w
    tokens = [tiling.window_tokens(win, w, device) for win in windows]
    coverage = torch.zeros(n_img, dtype=dtype, device=device)
    for idx in tokens:
        coverage[idx] += 1.0
    rng = tiling.step_rng(seed, step, 0x4A10) if halo > 0 else None
    rings = [tiling.halo_ring(win, int(rng.integers(0, halo + 1)), h, w, device) if rng is not None else None
             for win in windows]
    weights = tiling.merge_weights(windows, tokens, n_img, dtype, device)
    by_key: dict = {}
    for i, idx in enumerate(tokens):
        ring = rings[i]
        by_key.setdefault((int(idx.numel()), 0 if ring is None else int(ring.numel())), []).append(i)
    groups = [
        WindowGroup(
            tokens=torch.stack([tokens[i] for i in members]),
            halo=torch.stack([rings[i] for i in members]) if ring_len else None,
            weights=torch.cat([weights[i] for i in members]),
        )
        for (_, ring_len), members in by_key.items()
    ]
    return FineLayout(h, w, windows, groups, coverage, registers)


def windowed_attention(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_img_unrotated,
                       layout: FineLayout, backend=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention inside each window, its keys extended by the Halo ring and the bridge tokens.

    Queries and keys are already rotated with their true grid positions; ``k_img_unrotated``
    ranks the bridge tokens. Window outputs are merged with the centre weights; the text
    output is the mean over windows.
    """
    b, t, heads, dim = q_txt.shape
    n_img = q_img.shape[1]
    out_img = torch.zeros(b, n_img, heads, dim, dtype=q_img.dtype, device=q_img.device)
    out_txt = torch.zeros(b, t, heads, dim, dtype=q_img.dtype, device=q_img.device)
    score = bridge.token_importance(k_img_unrotated) if layout.registers > 0 else None

    for group in layout.groups:
        g, n = group.tokens.shape
        flat = group.tokens.reshape(-1)

        def gather(x_txt, x_img):
            local = x_img[:, flat].reshape(b, g, n, heads, dim)
            return torch.cat([x_txt.unsqueeze(1).expand(b, g, t, heads, dim), local], dim=2)

        extra_k, extra_v = [], []
        if score is not None:
            reg = bridge.select_registers(score, group.tokens, layout.registers, layout.h, layout.w).reshape(-1)
            extra_k.append(k_img[:, reg].reshape(b, g, -1, heads, dim))
            extra_v.append(v_img[:, reg].reshape(b, g, -1, heads, dim))
        if group.halo is not None:
            ring = group.halo.reshape(-1)
            extra_k.append(k_img[:, ring].reshape(b, g, -1, heads, dim))
            extra_v.append(v_img[:, ring].reshape(b, g, -1, heads, dim))

        q = gather(q_txt, q_img).reshape(b * g, t + n, heads, dim)
        k = torch.cat([gather(k_txt, k_img)] + extra_k, dim=2).reshape(b * g, -1, heads, dim)
        v = torch.cat([gather(v_txt, v_img)] + extra_v, dim=2).reshape(b * g, -1, heads, dim)
        out = dispatch_attention_fn(q, k, v, attn_mask=None, backend=backend).reshape(b, g, t + n, heads, dim)

        out_txt += out[:, :, :t].sum(dim=1)
        out_img.index_add_(1, flat, out[:, :, t:].reshape(b, g * n, heads, dim) * group.weights.view(1, -1, 1, 1))

    out_img = out_img / layout.coverage.clamp(min=1.0).view(1, n_img, 1, 1)
    out_txt = out_txt / max(float(len(layout.windows)), 1.0)
    return out_txt, out_img
