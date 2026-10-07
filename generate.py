"""High-resolution text-to-image generation with AReA."""

import argparse
import json
import re
from pathlib import Path

import torch
from diffusers import FluxPipeline

from area import AReA, POOLING

OFFLOAD_BELOW_GB = 60

MODELS = {
    "flux1": dict(pipeline=FluxPipeline, checkpoint="black-forest-labs/FLUX.1-Krea-dev", steps=28, guidance=4.5),
}


def read_prompts(path: Path):
    if path.suffix == ".jsonl":
        return [json.loads(line)["prompt"] for line in path.read_text().splitlines() if line.strip()]
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def slug(prompt: str, words: int = 6) -> str:
    return "_".join(re.sub(r"[^a-z0-9 ]", "", prompt.lower()).split()[:words]) or "image"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="text prompt")
    source.add_argument("--prompt-file", type=Path,
                        help="one prompt per line (.txt) or a 'prompt' field per line (.jsonl), e.g. HRGen-1K/prompts.jsonl")
    parser.add_argument("--model", choices=sorted(MODELS), default="flux1", help="model family")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint of that family: a Hugging Face repo id or a local path (default: the model's own)")
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=None, help="denoising steps (default: the model's own)")
    parser.add_argument("--guidance", type=float, default=None, help="guidance scale (default: the model's own)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pooling", choices=POOLING, default="subsample",
                        help="Phase I: the token that represents each f x f block")
    parser.add_argument("--phase1-steps", type=int, default=None, help="steps of Phase I (default: the model's own)")
    parser.add_argument("--halo", type=int, default=8, help="largest Halo width delta_max, in patches")
    parser.add_argument("--bridge-ratio", type=float, default=0.5, help="global-KV bridge ratio rho")
    parser.add_argument("--bootstrap", action="store_true",
                        help="AReA-Bootstrap: replace Phase I with a native-resolution generation (FLUX.1 only)")
    parser.add_argument("--offload", action="store_true",
                        help=f"offload idle models to CPU (on by default below {OFFLOAD_BELOW_GB} GB of GPU memory)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    model = MODELS[args.model]

    pipe = model["pipeline"].from_pretrained(args.checkpoint or model["checkpoint"], torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if args.offload or gpu_gb < OFFLOAD_BELOW_GB:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    area = AReA(pipe, pooling=args.pooling, phase1_steps=args.phase1_steps, halo=args.halo,
                bridge_ratio=args.bridge_ratio, seed=args.seed)

    prompts = [args.prompt] if args.prompt else read_prompts(args.prompt_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        settings = dict(height=args.height, width=args.width, num_inference_steps=args.steps or model["steps"],
                        guidance_scale=args.guidance if args.guidance is not None else model["guidance"])
        if args.bootstrap:
            image = area.bootstrap(prompt, seed=args.seed, **settings).images[0]
        else:
            generator = torch.Generator(pipe._execution_device).manual_seed(args.seed)
            image = pipe(prompt, generator=generator, **settings).images[0]
        name = f"{slug(prompt)}_{args.width}x{args.height}_seed{args.seed}.png"
        if args.prompt_file:
            name = f"{i:04d}_{name}"
        image.save(args.output_dir / name)
        print(f"[{i + 1}/{len(prompts)}] {args.output_dir / name}")


if __name__ == "__main__":
    main()
