"""High-resolution text-to-image generation with AReA."""

import argparse
import json
import re
from pathlib import Path

import torch
from diffusers import FluxPipeline, Ideogram4PromptEnhancerHead, QwenImagePipeline

from area import AReA, POOLING
from area.multi_gpu import spread_blocks
from area.pipelines import SlimIdeogram4Pipeline

OFFLOAD_BELOW_GB = 60
IDEOGRAM_ENHANCER = "diffusers/qwen3-vl-8b-instruct-lm-head"
# Qwen-Image's own quality suffix, extended with detail terms that name no content, light or style
QWEN_SUFFIX = (", Ultra HD, 4K, cinematic composition, faithful to the described content, quantities, style and "
               "medium, coherent fine detail, consistent scale, proportion and perspective, believable light and "
               "shadow, smooth tonal gradations.")

# guidance None: the pipeline's own guidance schedule; vae_tiling "low_memory": tile only when offloading;
# guidance_arg: the pipeline argument that takes the guidance scale; negative_prompt: turns on true CFG;
# prompt_suffix: appended to every prompt
MODELS = {
    "flux1": dict(pipeline=FluxPipeline, checkpoint="black-forest-labs/FLUX.1-Krea-dev", steps=28, guidance=4.5,
                  vae_tiling="always"),
    "qwen-image": dict(pipeline=QwenImagePipeline, checkpoint="Qwen/Qwen-Image", steps=40, guidance=4.0,
                       guidance_arg="true_cfg_scale", negative_prompt="", vae_tiling="always",
                       prompt_suffix=QWEN_SUFFIX),
    "ideogram4": dict(pipeline=SlimIdeogram4Pipeline, checkpoint="ideogram-ai/ideogram-4-nf4-diffusers", steps=48,
                      guidance=None, fallback_guidance=7.0, vae_tiling="low_memory"),
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
    parser.add_argument("--switch-sigma", type=float, default=None,
                        help="noise level sigma* where Phase II takes over (default: the model's own)")
    parser.add_argument("--halo", type=int, default=8,
                        help="Halo width delta_max, in patches (sides draw delta_max/2 to delta_max, widened "
                             "where windows barely overlap)")
    parser.add_argument("--bridge-ratio", type=float, default=0.5, help="global-KV bridge ratio rho")
    parser.add_argument("--bootstrap", action="store_true",
                        help="AReA-Bootstrap: replace Phase I with a native-resolution generation (FLUX.1 only)")
    parser.add_argument("--offload", action="store_true",
                        help=f"offload idle models to CPU (on by default below {OFFLOAD_BELOW_GB} GB of GPU memory)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="spread the transformer blocks over all visible GPUs, for canvases that do not fit on "
                             "one (adds memory, not speed)")
    parser.add_argument("--prompt-suffix", action=argparse.BooleanOptionalAction, default=True,
                        help="append the model's quality suffix to every prompt (Qwen-Image only, QWEN_SUFFIX)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.multi_gpu and args.offload:
        parser.error("--multi-gpu and --offload cannot be combined")
    model = MODELS[args.model]

    upsample = args.model == "ideogram4"  # Ideogram 4 always rewrites the prompt into its structured caption
    extra = {}
    if upsample:
        extra["prompt_enhancer_head"] = Ideogram4PromptEnhancerHead.from_pretrained(
            IDEOGRAM_ENHANCER, torch_dtype=torch.bfloat16)
    pipe = model["pipeline"].from_pretrained(args.checkpoint or model["checkpoint"], torch_dtype=torch.bfloat16,
                                             **extra)
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    offload = not args.multi_gpu and (args.offload or gpu_gb < OFFLOAD_BELOW_GB)
    if model["vae_tiling"] == "always" or offload:
        pipe.vae.enable_tiling()
    if args.multi_gpu:
        spread_blocks(pipe)
    elif offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    area = AReA(pipe, pooling=args.pooling, switch_sigma=args.switch_sigma, halo=args.halo,
                bridge_ratio=args.bridge_ratio, seed=args.seed)

    prompts = [args.prompt] if args.prompt else read_prompts(args.prompt_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        steps = args.steps or model["steps"]
        settings = dict(height=args.height, width=args.width, num_inference_steps=steps)
        guidance = args.guidance if args.guidance is not None else model["guidance"]
        if guidance is None and steps != model["steps"]:
            guidance = model["fallback_guidance"]
        if guidance is not None:
            settings[model.get("guidance_arg", "guidance_scale")] = guidance
            if args.model == "ideogram4":
                settings["guidance_schedule"] = None
        if "negative_prompt" in model:
            settings["negative_prompt"] = model["negative_prompt"]
        if upsample:
            settings["prompt_upsampling"] = True
        text = prompt + (model.get("prompt_suffix", "") if args.prompt_suffix else "")
        if args.bootstrap:
            image = area.bootstrap(text, seed=args.seed, **settings).images[0]
        else:
            generator = torch.Generator(pipe._execution_device).manual_seed(args.seed)
            image = pipe(text, generator=generator, **settings).images[0]
        name = f"{slug(prompt)}_{args.width}x{args.height}_seed{args.seed}.png"
        if args.prompt_file:
            name = f"{i:04d}_{name}"
        image.save(args.output_dir / name)
        print(f"[{i + 1}/{len(prompts)}] {args.output_dir / name}")


if __name__ == "__main__":
    main()
