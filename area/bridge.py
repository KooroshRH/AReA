"""Global-KV bridge: which tokens outside a window join its keys and values (Appendix E)."""

import torch

CELLS = 8
_CHUNK_ELEMS = 8_000_000


@torch.no_grad()
def token_importance(k_img: torch.Tensor) -> torch.Tensor:
    """Eq. 17: squared distance of each (unrotated) image key ``[B, N, H, D]`` from the canvas-mean key."""
    n = int(k_img.shape[1])
    c = int(k_img.shape[2]) * int(k_img.shape[3])
    chunk = max(1, _CHUNK_ELEMS // c)
    total = torch.zeros(c, device=k_img.device, dtype=torch.float32)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        total += k_img[:, i0:i1].float().sum(0).reshape(i1 - i0, c).sum(0)
    mean = total / float(n)
    score = torch.empty(n, device=k_img.device, dtype=torch.float32)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        score[i0:i1] = (k_img[:, i0:i1].float().sum(0).reshape(i1 - i0, c) - mean).pow(2).sum(dim=1)
    return score


@torch.no_grad()
def select_registers(score: torch.Tensor, windows: torch.Tensor, count: int, h: int, w: int) -> torch.Tensor:
    """``[G, count]`` bridge tokens for ``G`` windows given their token indices ``[G, n]``.

    The canvas is split into a ``CELLS x CELLS`` grid. Each cell gets a quota proportional to
    how many of its tokens lie outside the window (leftovers go to the cells with the most
    headroom), and is filled with its highest-scoring outside tokens.
    """
    dev = score.device
    n = int(score.shape[0])
    pos = torch.arange(n, device=dev)
    cr = ((pos // w).float() * CELLS / max(h, 1)).long().clamp_(0, CELLS - 1)
    cc = ((pos % w).float() * CELLS / max(w, 1)).long().clamp_(0, CELLS - 1)
    cell = cr * CELLS + cc
    out = torch.empty(int(windows.shape[0]), count, dtype=torch.long, device=dev)
    for g in range(int(windows.shape[0])):
        sg = score.index_fill(0, windows[g], float("-inf"))
        avail = torch.isfinite(sg)
        cnt = torch.bincount(cell[avail], minlength=CELLS * CELLS).float()
        quota = torch.minimum(torch.floor(cnt / cnt.sum().clamp_min(1.0) * count).long(), cnt.long())
        short = count - int(quota.sum())
        if short > 0:
            order = torch.argsort(cnt - quota.float(), descending=True)
            head = (cnt.long() - quota)[order]
            take = torch.minimum(torch.clamp(short - (torch.cumsum(head, 0) - head), min=0), head)
            quota[order] += take
        ai = torch.nonzero(avail, as_tuple=False).squeeze(1)
        if ai.numel():
            ranked = ai[torch.sort(sg[ai], descending=True, stable=True)[1]]
            ranked = ranked[torch.sort(cell[ranked], stable=True)[1]]
            start = torch.cumsum(cnt.long(), 0) - cnt.long()
            rank_in_cell = torch.arange(ranked.numel(), device=dev) - start[cell[ranked]]
            sel = ranked[rank_in_cell < quota[cell[ranked]]]
        else:
            sel = torch.empty(0, dtype=torch.long, device=dev)
        if sel.numel() < count:
            extra = sg.topk(min(n, count)).indices
            sel = torch.cat([sel, extra[~torch.isin(extra, sel)][: count - sel.numel()]])
        out[g] = sel[:count]
    return out
