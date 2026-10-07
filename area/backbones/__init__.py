from .flux import FluxBackbone

BACKBONES = (FluxBackbone,)


def find_backbone(pipe):
    for backbone in BACKBONES:
        if backbone.supports(pipe):
            return backbone
    supported = ", ".join(b.__name__.replace("Backbone", "") for b in BACKBONES)
    raise ValueError(f"no AReA backbone for {type(pipe).__name__}; supported: {supported}")
