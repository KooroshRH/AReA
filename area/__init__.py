"""AReA: Adaptive Receptive-Field Attention for efficient high-resolution diffusion transformers.

    import torch
    from diffusers import FluxPipeline
    from area import AReA

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16).to("cuda")
    AReA(pipe)
    image = pipe("a prompt", height=4096, width=4096, num_inference_steps=28, guidance_scale=4.5).images[0]
"""

from dataclasses import dataclass
from typing import Optional

from .attention import POOLING
from .backbones import find_backbone

__all__ = ["AReA", "AReAConfig", "POOLING"]


@dataclass
class AReAConfig:
    switch_sigma: Optional[float] = None  # sigma*: Phase II from the first step at or below this noise level;
                                          # None uses the backbone default
    pooling: str = "subsample"            # Phase I: one of POOLING
    halo: int = 8                         # Halo width delta_max, in patches: sides draw delta_max/2 to delta_max,
                                          # widened where windows barely overlap
    bridge_ratio: float = 0.5             # rho: each window gets (rho * w)^2 bridge tokens
    seed: int = 0                         # draws of boundary jitter, Halo widths and random pooling

    def __post_init__(self):
        if self.pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}, got {self.pooling!r}")
        if self.halo < 0 or self.bridge_ratio < 0:
            raise ValueError("halo and bridge_ratio must be non-negative")
        if self.switch_sigma is not None and not 0.0 < self.switch_sigma < 1.0:
            raise ValueError(f"switch_sigma must lie in (0, 1), got {self.switch_sigma}")


class AReA:
    """Runs a diffusers pipeline with AReA attention.

    Attaching swaps the transformer's attention processors and adds two hooks; the pipeline is
    then called as usual. Canvases at or below the model's native resolution keep plain attention.
    """

    def __init__(self, pipe, config: Optional[AReAConfig] = None, **kwargs):
        self.config = config or AReAConfig(**kwargs)
        self.pipe = pipe
        self.backbone = find_backbone(pipe)(pipe, self.config)
        self.backbone.attach()

    def bootstrap(self, prompt, **kwargs):
        """AReA-Bootstrap: Phase I is replaced by a native-resolution generation (FLUX.1 only)."""
        if not hasattr(self.backbone, "bootstrap"):
            raise NotImplementedError(f"bootstrap is not available for {type(self.backbone).__name__}")
        return self.backbone.bootstrap(prompt, **kwargs)

    def remove(self) -> None:
        """Restore the pipeline's original attention."""
        self.backbone.detach()
