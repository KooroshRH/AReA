"""Model parallelism: a pipeline's transformer blocks spread over several GPUs.

It adds memory, not speed: the blocks still run one after another. Each block moves its inputs
to its own GPU, and the last block of each block list returns its output to the first GPU, where
everything outside the blocks (text encoders, VAE, embeddings, final layers) stays.
"""

import itertools
from dataclasses import fields, is_dataclass, replace
from typing import List, Optional, Sequence

import torch


def to_device(obj, device):
    """``obj`` with every tensor in it, also inside lists, tuples, dicts and dataclasses, on ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if type(obj) in (list, tuple):
        return type(obj)(to_device(x, device) for x in obj)
    if type(obj) is dict:
        return {k: to_device(v, device) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **{f.name: to_device(getattr(obj, f.name), device) for f in fields(obj) if f.init})
    return obj


def _block_lists(transformer: torch.nn.Module) -> List[torch.nn.ModuleList]:
    """The transformer's block lists, in the order they run (diffusers registers them that way)."""
    return [child for child in transformer.children() if isinstance(child, torch.nn.ModuleList)]


def spread_blocks(pipe, devices: Optional[Sequence] = None) -> None:
    """Spread the blocks of every transformer of ``pipe`` (each component whose name contains
    "transformer", diffusers' name for the denoiser) over ``devices`` (default: every visible GPU) in
    contiguous runs of about equal weight size, and put the rest of the pipeline on the first device."""
    devices = [torch.device(d) for d in (devices if devices is not None else range(torch.cuda.device_count()))]
    if len(devices) < 2:
        raise ValueError(f"multi-GPU needs at least two GPUs, found {len(devices)}")
    for name, component in pipe.components.items():
        if not isinstance(component, torch.nn.Module):
            continue
        if "transformer" in name:
            _spread_transformer(component, devices)
        else:
            component.to(devices[0])


def _spread_transformer(transformer: torch.nn.Module, devices: List[torch.device]) -> None:
    main = devices[0]
    lists = _block_lists(transformer)
    if not lists:
        raise ValueError(f"{type(transformer).__name__} has no block list to spread")
    for child in transformer.children():
        if not isinstance(child, torch.nn.ModuleList):
            child.to(main)
    for param in transformer.parameters(recurse=False):
        param.data = param.data.to(main)
    for name, buffer in transformer.named_buffers(recurse=False):
        setattr(transformer, name, buffer.to(main))

    blocks = [block for blocks in lists for block in blocks]
    sizes = [sum(t.numel() * t.element_size() for t in itertools.chain(b.parameters(), b.buffers())) for b in blocks]
    total, done = float(sum(sizes)), 0.0
    placed = {}
    for block, size in zip(blocks, sizes):
        device = devices[min(int((done + size / 2) / total * len(devices)), len(devices) - 1)]
        done += size
        block.to(device)
        placed[block] = device
        if device != main:
            block.register_forward_pre_hook(
                lambda module, args, kwargs, d=device: (to_device(args, d), to_device(kwargs, d)), with_kwargs=True)
    for blocks in lists:
        if placed[blocks[-1]] != main:
            blocks[-1].register_forward_hook(lambda module, args, output: to_device(output, main))
