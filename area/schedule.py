"""Per-step schedule of AReA: the Phase I pooling ramp and the Phase II window sizes."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

F_MIN = 2


def even_starts(total: int, size: int) -> List[int]:
    """Starts of the fewest ``size``-wide windows covering ``total``, spaced evenly."""
    total, size = int(total), int(size)
    if total <= size:
        return [0]
    n = (total - 1) // size + 1
    return [round(i * (total - size) / (n - 1)) for i in range(n)]


def window_count(h: int, w: int, size: int) -> int:
    """N(w) of Eq. 6."""
    return len(even_starts(h, size)) * len(even_starts(w, size))


def window_step_cost(h: int, w: int, size: int, txt_len: int = 0) -> float:
    """C(w) of Eq. 7: one windowed step in units of one full-attention pass. Every window runs
    its ``txt_len`` text tokens as queries and keys, so it costs ``(size^2 + T)^2`` against
    ``(h*w + T)^2`` for full attention."""
    t = float(txt_len)
    return window_count(h, w, size) * (float(size) ** 2 + t) ** 2 / (float(h * w) + t) ** 2


def grid_divisors(h: int, w: int) -> List[int]:
    g = math.gcd(int(h), int(w))
    return [d for d in range(1, g + 1) if g % d == 0]


def initial_factor(h: int, w: int, native: int) -> int:
    """f_0 of Eq. 5: the smallest grid divisor >= 2 whose pooled grid fits the native extent."""
    divisors = [d for d in grid_divisors(h, w) if d >= 2]
    if not divisors:
        raise ValueError(
            f"the patch grid {h}x{w} has no common divisor >= 2; choose a height and width "
            f"whose patch counts share a factor"
        )
    fits = [d for d in divisors if max(h, w) / d <= native]
    return min(fits) if fits else max(divisors)


def pooling_ramp(f0: int, steps: int, h: int, w: int, f_min: int = F_MIN) -> List[int]:
    """Eqs. 2 and 4: f decreases linearly from f_0 to f_min, snapped to non-increasing grid divisors."""
    divisors = [d for d in grid_divisors(h, w) if d <= f0]
    if f_min not in divisors:
        f_min = min(divisors, key=lambda d: (abs(d - f_min), -d))
    if steps <= 1:
        return [f0]
    plan, prev = [], f0
    for target in np.linspace(float(f0), float(f_min), steps):
        candidates = [d for d in divisors if d <= prev]
        plan.append(min(candidates, key=lambda d: (abs(d - target), -d)))
        prev = plan[-1]
    plan[0], plan[-1] = f0, f_min
    return plan


def start_window(h: int, w: int, lo: int, txt_len: int = 0) -> int:
    """w_start of Eq. 8: the largest window whose step is cheaper than full attention."""
    for size in range(max(h, w), lo - 1, -1):
        if window_step_cost(h, w, size, txt_len) < 1.0 and window_count(h, w, size) > 1:
            return size
    return lo


def _geometric_sizes(steps: int, start: int, floor: int) -> List[int]:
    if steps == 1:
        return [start]
    sizes = [
        max(floor, int(round(np.exp(np.log(start) + float(t) * (np.log(floor) - np.log(start))))))
        for t in np.linspace(0.0, 1.0, steps)
    ]
    sizes[0], sizes[-1] = start, floor
    return sizes


def window_sizes(h: int, w: int, steps: int, start: int, floor: int, txt_len: int = 0) -> List[int]:
    """Eqs. 10 and 11: geometric shrink from w_start to w_min, each rung snapped down so it
    never costs more than the first one and sizes strictly decrease."""
    if steps <= 0:
        return []
    ideal = _geometric_sizes(steps, start, floor)
    if start - floor < steps - 1:
        return ideal
    cap = window_step_cost(h, w, start, txt_len)
    sizes: List[int] = []
    for i, target in enumerate(ideal):
        lo = floor + steps - i - 1
        hi = start if not sizes else min(int(target), sizes[-1] - 1)
        hi = max(hi, lo)
        pick = next((s for s in range(hi, lo - 1, -1) if window_step_cost(h, w, s, txt_len) <= cap + 1e-12),
                    None)
        if pick is None:
            pick = min(range(lo, hi + 1), key=lambda s: window_step_cost(h, w, s, txt_len))
        sizes.append(pick)
    return sizes


@dataclass(frozen=True)
class Schedule:
    """What every denoising step runs: Phase I with pooling factor f, or Phase II with window side w."""

    grid: Tuple[int, int]
    factors: Tuple[int, ...]
    windows: Tuple[int, ...]

    @property
    def steps(self) -> int:
        return len(self.factors) + len(self.windows)

    def at(self, step: int) -> Tuple[str, int]:
        if step < len(self.factors):
            return "coarse", self.factors[step]
        return "fine", self.windows[min(step - len(self.factors), len(self.windows) - 1)]


def build_schedule(
    h: int, w: int, steps: int, phase1_steps: int, native: int, w_min: Optional[int] = None,
    txt_len: int = 0,
) -> Schedule:
    """Schedule for an ``h x w`` patch grid over ``steps`` denoising steps."""
    phase1_steps = max(0, min(int(phase1_steps), int(steps)))
    extent = max(h, w)
    w_min = min(int(w_min or native), extent)
    factors = pooling_ramp(initial_factor(h, w, native), phase1_steps, h, w) if phase1_steps else []
    phase2_steps = steps - phase1_steps
    windows: List[int] = []
    if phase2_steps:
        handoff = extent // (factors[-1] if factors else 1)
        lo = max(w_min, min(handoff, extent))
        windows = window_sizes(h, w, phase2_steps, start_window(h, w, lo, txt_len), w_min, txt_len)
    return Schedule((h, w), tuple(factors), tuple(windows))
