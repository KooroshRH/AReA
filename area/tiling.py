"""Phase II window tiling: base tiling, boundary jitter, DWT splits, Halo and merge weights."""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .schedule import even_starts

DWT_LEVELS = 3
DWT_PAD = 8
GATE_LO, GATE_HI = 0.05, 0.25
SPLIT_QUANTILE = 0.5
DEDUPE_COVER = 0.8
BLEND_FLOOR = 0.05
_CHUNK_TOKENS = 8192


@dataclass
class Window:
    """Inclusive box on the patch grid."""

    r0: int
    c0: int
    r1: int
    c1: int
    split: bool = False

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width


def step_rng(seed: int, step: int, salt: int) -> np.random.Generator:
    return np.random.default_rng((int(seed) & 0x7FFFFFFF) + int(step) * 2_654_435_761 + salt)


def jittered_starts(total: int, size: int, rng: np.random.Generator) -> List[int]:
    """A uniformly random valid placement of the base windows on one axis.

    Same count and size as ``even_starts`` with both end windows pinned, so the axis stays
    covered. Valid placements are the integer compositions of ``total - size`` into gaps in
    ``[1, size]``; one is drawn uniformly with an exact counting table, since drawing each
    start in turn would pile the slack at the far edge.
    """
    base = even_starts(total, size)
    T, m = int(total) - int(size), len(base) - 1
    if m <= 1 or T <= 0:
        return base
    counts = np.zeros((m + 1, T + 1), dtype=np.float64)
    counts[0, 0] = 1.0
    idx = np.arange(T + 1)
    for i in range(1, m + 1):
        pre = np.cumsum(counts[i - 1])
        a, b = idx - 1, idx - size - 1
        counts[i] = (np.where(a >= 0, pre[np.clip(a, 0, T)], 0.0)
                     - np.where(b >= 0, pre[np.clip(b, 0, T)], 0.0))
    if counts[m, T] <= 0.0:
        return base
    starts, pos, rem = [0], 0, T
    for i in range(m, 0, -1):
        cand = np.arange(1, min(size, rem - (i - 1)) + 1)
        weight = counts[i - 1][rem - cand]
        total_w = float(weight.sum())
        gap = int(cand[-1]) if total_w <= 0.0 else int(rng.choice(cand, p=weight / total_w))
        pos += gap
        rem -= gap
        starts.append(pos)
    return starts


@torch.no_grad()
def detail_map(hidden_states: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Normalised wavelet detail map E (Eqs. 12 and 13) from image tokens ``[B, h*w, C]``."""
    n = h * w
    s = torch.empty(n, device=hidden_states.device, dtype=torch.float32)
    for i0 in range(0, n, _CHUNK_TOKENS):
        i1 = min(n, i0 + _CHUNK_TOKENS)
        s[i0:i1] = hidden_states[:, i0:i1].float().std(dim=-1).mean(dim=0)
    s = s.reshape(h, w)
    x = (s - s.mean()).cpu().numpy()

    multiple = (2 ** DWT_LEVELS) * DWT_PAD
    x = np.pad(x, ((0, (-h) % multiple), (0, (-w) % multiple)), mode="edge")
    energy = np.zeros_like(x, dtype=np.float32)
    cur = x.astype(np.float64)
    for level in range(DWT_LEVELS):
        ch, cw = cur.shape
        if ch < 2 or cw < 2:
            break
        cur = cur[: ch - ch % 2, : cw - cw % 2]
        a, b, c, d = cur[0::2, 0::2], cur[0::2, 1::2], cur[1::2, 0::2], cur[1::2, 1::2]
        detail = (np.abs((a + b - c - d) * 0.25) + np.abs((a - b + c - d) * 0.25)
                  + np.abs((a - b - c + d) * 0.25))
        up = detail.astype(np.float64)
        while up.shape[0] < x.shape[0] or up.shape[1] < x.shape[1]:
            up = np.repeat(np.repeat(up, 2, axis=0), 2, axis=1)
        energy += (up[: x.shape[0], : x.shape[1]].astype(np.float32) * (0.5 ** level)).astype(np.float32)
        cur = (a + b + c + d) * 0.25
    energy = energy[:h, :w]
    if energy.max() > 0:
        energy = energy / (energy.max() + 1e-8)
    return energy.astype(np.float32)


def _mean_energy(energy: np.ndarray, win: Window) -> float:
    block = energy[win.r0 : win.r1 + 1, win.c0 : win.c1 + 1]
    return float(block.mean()) if block.size else 0.0


def _quadrants(win: Window) -> List[Window]:
    rm = win.r0 + win.height // 2 - 1
    cm = win.c0 + win.width // 2 - 1
    return [Window(r0, c0, r1, c1, split=True)
            for r0, r1 in ((win.r0, rm), (rm + 1, win.r1))
            for c0, c1 in ((win.c0, cm), (cm + 1, win.c1))]


def _cover(a: Window, b: Window) -> float:
    r0, c0, r1, c1 = max(a.r0, b.r0), max(a.c0, b.c0), min(a.r1, b.r1), min(a.c1, b.c1)
    if r1 < r0 or c1 < c0:
        return 0.0
    return (r1 - r0 + 1) * (c1 - c0 + 1) / min(a.area, b.area)


def _dedupe(windows: List[Window]) -> List[Window]:
    """Replace split windows that cover each other by at least ``DEDUPE_COVER`` with their union."""
    split = [win for win in windows if win.split]
    kept = [win for win in windows if not win.split]
    while len(split) >= 2:
        best, pair = DEDUPE_COVER, None
        for i in range(len(split)):
            for j in range(i + 1, len(split)):
                f = _cover(split[i], split[j])
                if f >= best:
                    best, pair = f, (i, j)
        if pair is None:
            break
        b = split.pop(pair[1])
        a = split.pop(pair[0])
        split.append(Window(min(a.r0, b.r0), min(a.c0, b.c0), max(a.r1, b.r1), max(a.c1, b.c1), split=True))
    return kept + split


def adaptive_tiling(
    energy: np.ndarray, h: int, w: int, size: int, w_min: int, txt_len: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Window]:
    """Base tiling of side ``size`` with the smooth windows split into quadrants (Appendix C)."""
    rows = jittered_starts(h, size, rng) if rng is not None else even_starts(h, size)
    cols = jittered_starts(w, size, rng) if rng is not None else even_starts(w, size)
    boxes = [Window(r, c, min(h - 1, r + size - 1), min(w - 1, c + size - 1)) for r in rows for c in cols]
    e = np.array([_mean_energy(energy, box) for box in boxes], dtype=np.float64).reshape(len(rows), len(cols))

    tile = max(1, size // 2)
    th, tw = energy.shape[0] // tile, energy.shape[1] // tile
    tiles = (energy[: th * tile, : tw * tile].reshape(th, tile, tw, tile).mean(axis=(1, 3))
             if th * tw >= 4 else e)
    t_mean = float(tiles.mean())
    u = float(tiles.std() / t_mean) if t_mean > 1e-12 else 0.0
    alpha = min(1.0, max(0.0, (u - GATE_LO) / max(1e-6, GATE_HI - GATE_LO)))
    max_splits = int(alpha * len(boxes))
    if alpha <= 0.0 or len(boxes) < 2 or max_splits == 0:
        return boxes

    def cost(win: Window) -> float:
        # text tokens are queries and keys of every window
        return float(txt_len + win.area) ** 2

    budget = total = sum(cost(box) for box in boxes)
    smooth = -e
    threshold = float(np.quantile(smooth, SPLIT_QUANTILE))
    mean = float(smooth.mean())
    split = [False] * len(boxes)
    n_split = 0
    for k in sorted(range(len(boxes)), key=lambda k: -smooth.flat[k]):
        if n_split >= max_splits or smooth.flat[k] < threshold or smooth.flat[k] <= mean:
            break
        box = boxes[k]
        if min(box.height // 2, box.width // 2) < max(1, w_min // 2):
            continue
        delta = sum(cost(q) for q in _quadrants(box)) - cost(box)
        if total + delta > budget:
            continue
        split[k] = True
        total += delta
        n_split += 1

    windows: List[Window] = []
    for box, is_split in zip(boxes, split):
        windows.extend(_quadrants(box) if is_split else [box])
    return _dedupe(windows)


def window_tokens(win: Window, w: int, device) -> torch.Tensor:
    rows = torch.arange(win.r0, win.r1 + 1, device=device)
    cols = torch.arange(win.c0, win.c1 + 1, device=device)
    return (rows[:, None] * w + cols[None, :]).reshape(-1)


def _grown_box(win: Window, depth: Union[int, Sequence[int]], h: int, w: int) -> Tuple[int, int, int, int]:
    """The window's box grown by ``depth`` and clipped to the grid, as inclusive ``(r0, r1, c0, c1)``.

    ``depth`` is one width for all four sides or ``(top, bottom, left, right)``.
    """
    top, bottom, left, right = (depth,) * 4 if isinstance(depth, int) else depth
    return (max(0, win.r0 - top), min(h - 1, win.r1 + bottom),
            max(0, win.c0 - left), min(w - 1, win.c1 + right))


def ring_size(win: Window, depth: Union[int, Sequence[int]], h: int, w: int) -> int:
    """Number of tokens in ``halo_ring(win, depth, ...)``."""
    r0, r1, c0, c1 = _grown_box(win, depth, h, w)
    return (r1 - r0 + 1) * (c1 - c0 + 1) - win.area


def halo_ring(win: Window, depth: Union[int, Sequence[int]], h: int, w: int, device) -> Optional[torch.Tensor]:
    """Tokens of the window's box grown by ``depth`` and clipped to the grid, minus the window.

    ``depth`` is one width for all four sides or ``(top, bottom, left, right)``.
    """
    if ring_size(win, depth, h, w) <= 0:
        return None
    r0, r1, c0, c1 = _grown_box(win, depth, h, w)
    rows = torch.arange(r0, r1 + 1, device=device)
    cols = torch.arange(c0, c1 + 1, device=device)
    rr, cc = rows[:, None], cols[None, :]
    outside = (rr < win.r0) | (rr > win.r1) | (cc < win.c0) | (cc > win.c1)
    return (rr * w + cc)[outside]


def _side_profile(total: int, length: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """``length`` random reaches in ``[0, cap]`` that sum to ``total``."""
    depth = rng.integers(0, cap + 1, size=length)
    gap = total - int(depth.sum())
    while gap != 0:
        room = np.flatnonzero(depth < cap) if gap > 0 else np.flatnonzero(depth > 0)
        pick = rng.choice(room, size=min(abs(gap), room.size), replace=False)
        depth[pick] += 1 if gap > 0 else -1
        gap -= int(np.sign(gap)) * pick.size
    return depth


def jagged_halo_ring(win: Window, depth: Sequence[int], halo: int, h: int, w: int,
                     rng: np.random.Generator, device) -> Optional[torch.Tensor]:
    """``halo_ring`` with an uneven edge: every row (left, right) or column (top, bottom) of a side
    reaches its own random distance in ``[0, halo]``, and the reaches of a side sum to what the
    straight edge of ``depth`` would hold. The ring keeps ``ring_size(win, depth, ...)`` tokens, and
    its corners stay the blocks between the two sides' widths.
    """
    if ring_size(win, depth, h, w) <= 0:
        return None
    r0, r1, c0, c1 = _grown_box(win, depth, h, w)
    top, bottom, left, right = win.r0 - r0, r1 - win.r1, win.c0 - c0, c1 - win.c1   # widths after clipping
    rooms = (win.r0, h - 1 - win.r1, win.c0, w - 1 - win.c1)
    rows_in = np.arange(win.r0, win.r1 + 1)
    cols_in = np.arange(win.c0, win.c1 + 1)
    parts = []
    for width, room, along, side in ((top, rooms[0], cols_in, "top"), (bottom, rooms[1], cols_in, "bottom"),
                                     (left, rooms[2], rows_in, "left"), (right, rooms[3], rows_in, "right")):
        if width <= 0:
            continue
        cap = min(halo, room)
        reach = _side_profile(width * along.size, along.size, cap, rng)
        k = np.arange(1, cap + 1)[:, None]
        keep = k <= reach[None, :]
        if side == "top":
            rr, cc = win.r0 - k, along[None, :]
        elif side == "bottom":
            rr, cc = win.r1 + k, along[None, :]
        elif side == "left":
            rr, cc = along[None, :], win.c0 - k
        else:
            rr, cc = along[None, :], win.c1 + k
        parts.append(np.broadcast_to(rr * w + cc, keep.shape)[keep])
    for rows, cols in ((np.arange(r0, win.r0), np.arange(c0, win.c0)), (np.arange(r0, win.r0), np.arange(win.c1 + 1, c1 + 1)),
                       (np.arange(win.r1 + 1, r1 + 1), np.arange(c0, win.c0)),
                       (np.arange(win.r1 + 1, r1 + 1), np.arange(win.c1 + 1, c1 + 1))):
        parts.append((rows[:, None] * w + cols[None, :]).reshape(-1))
    return torch.from_numpy(np.concatenate(parts)).to(device)


def merge_weights(windows: List[Window], tokens: List[torch.Tensor], n_img: int,
                  dtype: torch.dtype, device) -> List[torch.Tensor]:
    """Centre-weighted merge: a separable raised cosine per window, rescaled so the weights
    covering a token sum to its coverage count (a token in one window keeps weight 1)."""
    raw = []
    for win in windows:
        r = torch.arange(win.height, device=device, dtype=torch.float32)
        c = torch.arange(win.width, device=device, dtype=torch.float32)
        fr = torch.sin(math.pi * (r + 0.5) / win.height) ** 2
        fc = torch.sin(math.pi * (c + 0.5) / win.width) ** 2
        raw.append((BLEND_FLOOR + (1.0 - BLEND_FLOOR) * fr[:, None] * fc[None, :]).reshape(-1))
    total = torch.zeros(n_img, dtype=torch.float32, device=device)
    count = torch.zeros(n_img, dtype=torch.float32, device=device)
    for idx, weight in zip(tokens, raw):
        total.index_add_(0, idx, weight)
        count[idx] += 1.0
    scale = count / total.clamp(min=1e-8)
    return [(weight * scale[idx]).to(dtype) for idx, weight in zip(tokens, raw)]
