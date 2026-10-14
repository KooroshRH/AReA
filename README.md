<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/title-dark.png">
    <img src="assets/title-light.png" alt="AReA: Adaptive Receptive-Field Attention for Efficient High-Resolution Diffusion Transformers" width="640">
  </picture>
</p>

<p align="center">
  <a href="https://rajabi2001.github.io/area/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2610.17227"><img src="https://img.shields.io/badge/arXiv-2610.17227-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/kooroshrh/AReA"><img src="https://img.shields.io/badge/Open%20in%20Spaces-white?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5NSIgaGVpZ2h0PSI4OCIgZmlsbD0ibm9uZSI+PHBhdGggZmlsbD0iI0ZGRDIxRSIgZD0iTTQ3LjIxIDc2LjVhMzQuNzUgMzQuNzUgMCAxIDAgMC02OS41IDM0Ljc1IDM0Ljc1IDAgMCAwIDAgNjkuNVoiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNODEuOTYgNDEuNzVhMzQuNzUgMzQuNzUgMCAxIDAtNjkuNSAwIDM0Ljc1IDM0Ljc1IDAgMCAwIDY5LjUgMFptLTczLjUgMGEzOC43NSAzOC43NSAwIDEgMSA3Ny41IDAgMzguNzUgMzguNzUgMCAwIDEtNzcuNSAwWiIvPjxwYXRoIGZpbGw9IiMzQTNCNDUiIGQ9Ik01OC41IDMyLjNjMS4yOC40NCAxLjc4IDMuMDYgMy4wNyAyLjM4YTUgNSAwIDEgMC02Ljc2LTIuMDdjLjYxIDEuMTUgMi41NS0uNzIgMy43LS4zMlpNMzQuOTUgMzIuM2MtMS4yOC40NC0xLjc5IDMuMDYtMy4wNyAyLjM4YTUgNSAwIDEgMSA2Ljc2LTIuMDdjLS42MSAxLjE1LTIuNTYtLjcyLTMuNy0uMzJaIi8+PHBhdGggZmlsbD0iI0ZGMzIzRCIgZD0iTTQ2Ljk2IDU2LjI5YzkuODMgMCAxMy04Ljc2IDEzLTEzLjI2IDAtMi4zNC0xLjU3LTEuNi00LjA5LS4zNi0yLjMzIDEuMTUtNS40NiAyLjc0LTguOSAyLjc0LTcuMTkgMC0xMy02Ljg4LTEzLTIuMzhzMy4xNiAxMy4yNiAxMyAxMy4yNloiLz48cGF0aCBmaWxsPSIjM0EzQjQ1IiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0zOS40MyA1NGE4LjcgOC43IDAgMCAxIDUuMy00LjQ5Yy40LS4xMi44MS41NyAxLjI0IDEuMjguNC42OC44MiAxLjM3IDEuMjQgMS4zNy40NSAwIC45LS42OCAxLjMzLTEuMzUuNDUtLjcuODktMS4zOCAxLjMyLTEuMjVhOC42MSA4LjYxIDAgMCAxIDUgNC4xN2MzLjczLTIuOTQgNS4xLTcuNzQgNS4xLTEwLjcgMC0yLjM0LTEuNTctMS42LTQuMDktLjM2bC0uMTQuMDdjLTIuMzEgMS4xNS01LjM5IDIuNjctOC43NyAyLjY3cy02LjQ1LTEuNTItOC43Ny0yLjY3Yy0yLjYtMS4yOS00LjIzLTIuMS00LjIzLjI5IDAgMy4wNSAxLjQ2IDguMDYgNS40NyAxMC45N1oiIGNsaXAtcnVsZT0iZXZlbm9kZCIvPjxwYXRoIGZpbGw9IiNGRjlEMEIiIGQ9Ik03MC43MSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0yNC4yMSAzN2EzLjI1IDMuMjUgMCAxIDAgMC02LjUgMy4yNSAzLjI1IDAgMCAwIDAgNi41Wk0xNy41MiA0OGMtMS42MiAwLTMuMDYuNjYtNC4wNyAxLjg3YTUuOTcgNS45NyAwIDAgMC0xLjMzIDMuNzYgNy4xIDcuMSAwIDAgMC0xLjk0LS4zYy0xLjU1IDAtMi45NS41OS0zLjk0IDEuNjZhNS44IDUuOCAwIDAgMC0uOCA3IDUuMyA1LjMgMCAwIDAtMS43OSAyLjgyYy0uMjQuOS0uNDggMi44LjggNC43NGE1LjIyIDUuMjIgMCAwIDAtLjM3IDUuMDJjMS4wMiAyLjMyIDMuNTcgNC4xNCA4LjUyIDYuMSAzLjA3IDEuMjIgNS44OSAyIDUuOTEgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMCAxMC45MyAxLjZjNS44NiAwIDEwLjA1LTEuOCAxMi40Ni01LjM0IDMuODgtNS42OSAzLjMzLTEwLjktMS43LTE1LjkyLTIuNzctMi43OC00LjYyLTYuODctNS03Ljc3LS43OC0yLjY2LTIuODQtNS42Mi02LjI1LTUuNjJhNS43IDUuNyAwIDAgMC00LjYgMi40NmMtMS0xLjI2LTEuOTgtMi4yNS0yLjg2LTIuODJBNy40IDcuNCAwIDAgMCAxNy41MiA0OFptMCA0Yy41MSAwIDEuMTQuMjIgMS44Mi42NSAyLjE0IDEuMzYgNi4yNSA4LjQzIDcuNzYgMTEuMTguNS45MiAxLjM3IDEuMzEgMi4xNCAxLjMxIDEuNTUgMCAyLjc1LTEuNTMuMTUtMy40OC0zLjkyLTIuOTMtMi41NS03LjcyLS42OC04LjAxLjA4LS4wMi4xNy0uMDIuMjQtLjAyIDEuNyAwIDIuNDUgMi45MyAyLjQ1IDIuOTNzMi4yIDUuNTIgNS45OCA5LjNjMy43NyAzLjc3IDMuOTcgNi44IDEuMjIgMTAuODMtMS44OCAyLjc1LTUuNDcgMy41OC05LjE2IDMuNTgtMy44MSAwLTcuNzMtLjktOS45Mi0xLjQ2LS4xMS0uMDMtMTMuNDUtMy44LTExLjc2LTcgLjI4LS41NC43NS0uNzYgMS4zNC0uNzYgMi4zOCAwIDYuNyAzLjU0IDguNTcgMy41NC40MSAwIC43LS4xNy44My0uNi43OS0yLjg1LTEyLjA2LTQuMDUtMTAuOTgtOC4xNy4yLS43My43MS0xLjAyIDEuNDQtMS4wMiAzLjE0IDAgMTAuMiA1LjUzIDExLjY4IDUuNTMuMTEgMCAuMi0uMDMuMjQtLjEuNzQtMS4yLjMzLTIuMDQtNC45LTUuMi01LjIxLTMuMTYtOC44OC01LjA2LTYuOC03LjMzLjI0LS4yNi41OC0uMzggMS0uMzggMy4xNyAwIDEwLjY2IDYuODIgMTAuNjYgNi44MnMyLjAyIDIuMSAzLjI1IDIuMWMuMjggMCAuNTItLjEuNjgtLjM4Ljg2LTEuNDYtOC4wNi04LjIyLTguNTYtMTEuMDEtLjM0LTEuOS4yNC0yLjg1IDEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNMzguNiA3Ni42OWMyLjc1LTQuMDQgMi41NS03LjA3LTEuMjItMTAuODQtMy43OC0zLjc3LTUuOTgtOS4zLTUuOTgtOS4zcy0uODItMy4yLTIuNjktMi45Yy0xLjg3LjMtMy4yNCA1LjA4LjY4IDguMDEgMy45MSAyLjkzLS43OCA0LjkyLTIuMjkgMi4xNy0xLjUtMi43NS01LjYyLTkuODItNy43Ni0xMS4xOC0yLjEzLTEuMzUtMy42My0uNi0zLjEzIDIuMi41IDIuNzkgOS40MyA5LjU1IDguNTYgMTEtLjg3IDEuNDctMy45My0xLjcxLTMuOTMtMS43MXMtOS41Ny04LjcxLTExLjY2LTYuNDRjLTIuMDggMi4yNyAxLjU5IDQuMTcgNi44IDcuMzMgNS4yMyAzLjE2IDUuNjQgNCA0LjkgNS4yLS43NSAxLjItMTIuMjgtOC41My0xMy4zNi00LjQtMS4wOCA0LjExIDExLjc3IDUuMyAxMC45OCA4LjE1LS44IDIuODUtOS4wNi01LjM4LTEwLjc0LTIuMTgtMS43IDMuMjEgMTEuNjUgNi45OCAxMS43NiA3LjAxIDQuMyAxLjEyIDE1LjI1IDMuNDkgMTkuMDgtMi4xMloiLz48cGF0aCBmaWxsPSIjRkY5RDBCIiBkPSJNNzcuNCA0OGMxLjYyIDAgMy4wNy42NiA0LjA3IDEuODdhNS45NyA1Ljk3IDAgMCAxIDEuMzMgMy43NiA3LjEgNy4xIDAgMCAxIDEuOTUtLjNjMS41NSAwIDIuOTUuNTkgMy45NCAxLjY2YTUuOCA1LjggMCAwIDEgLjggNyA1LjMgNS4zIDAgMCAxIDEuNzggMi44MmMuMjQuOS40OCAyLjgtLjggNC43NGE1LjIyIDUuMjIgMCAwIDEgLjM3IDUuMDJjLTEuMDIgMi4zMi0zLjU3IDQuMTQtOC41MSA2LjEtMy4wOCAxLjIyLTUuOSAyLTUuOTIgMi4wMWE0NC4zMyA0NC4zMyAwIDAgMS0xMC45MyAxLjZjLTUuODYgMC0xMC4wNS0xLjgtMTIuNDYtNS4zNC0zLjg4LTUuNjktMy4zMy0xMC45IDEuNy0xNS45MiAyLjc4LTIuNzggNC42My02Ljg3IDUuMDEtNy43Ny43OC0yLjY2IDIuODMtNS42MiA2LjI0LTUuNjJhNS43IDUuNyAwIDAgMSA0LjYgMi40NmMxLTEuMjYgMS45OC0yLjI1IDIuODctMi44MkE3LjQgNy40IDAgMCAxIDc3LjQgNDhabTAgNGMtLjUxIDAtMS4xMy4yMi0xLjgyLjY1LTIuMTMgMS4zNi02LjI1IDguNDMtNy43NiAxMS4xOGEyLjQzIDIuNDMgMCAwIDEtMi4xNCAxLjMxYy0xLjU0IDAtMi43NS0xLjUzLS4xNC0zLjQ4IDMuOTEtMi45MyAyLjU0LTcuNzIuNjctOC4wMWExLjU0IDEuNTQgMCAwIDAtLjI0LS4wMmMtMS43IDAtMi40NSAyLjkzLTIuNDUgMi45M3MtMi4yIDUuNTItNS45NyA5LjNjLTMuNzggMy43Ny0zLjk4IDYuOC0xLjIyIDEwLjgzIDEuODcgMi43NSA1LjQ3IDMuNTggOS4xNSAzLjU4IDMuODIgMCA3LjczLS45IDkuOTMtMS40Ni4xLS4wMyAxMy40NS0zLjggMTEuNzYtNy0uMjktLjU0LS43NS0uNzYtMS4zNC0uNzYtMi4zOCAwLTYuNzEgMy41NC04LjU3IDMuNTQtLjQyIDAtLjcxLS4xNy0uODMtLjYtLjgtMi44NSAxMi4wNS00LjA1IDEwLjk3LTguMTctLjE5LS43My0uNy0xLjAyLTEuNDQtMS4wMi0zLjE0IDAtMTAuMiA1LjUzLTExLjY4IDUuNTMtLjEgMC0uMTktLjAzLS4yMy0uMS0uNzQtMS4yLS4zNC0yLjA0IDQuODgtNS4yIDUuMjMtMy4xNiA4LjktNS4wNiA2LjgtNy4zMy0uMjMtLjI2LS41Ny0uMzgtLjk4LS4zOC0zLjE4IDAtMTAuNjcgNi44Mi0xMC42NyA2Ljgycy0yLjAyIDIuMS0zLjI0IDIuMWEuNzQuNzQgMCAwIDEtLjY4LS4zOGMtLjg3LTEuNDYgOC4wNS04LjIyIDguNTUtMTEuMDEuMzQtMS45LS4yNC0yLjg1LTEuMzEtMi44NVoiLz48cGF0aCBmaWxsPSIjRkZEMjFFIiBkPSJNNTYuMzMgNzYuNjljLTIuNzUtNC4wNC0yLjU2LTcuMDcgMS4yMi0xMC44NCAzLjc3LTMuNzcgNS45Ny05LjMgNS45Ny05LjNzLjgyLTMuMiAyLjctMi45YzEuODYuMyAzLjIzIDUuMDgtLjY4IDguMDEtMy45MiAyLjkzLjc4IDQuOTIgMi4yOCAyLjE3IDEuNTEtMi43NSA1LjYzLTkuODIgNy43Ni0xMS4xOCAyLjEzLTEuMzUgMy42NC0uNiAzLjEzIDIuMi0uNSAyLjc5LTkuNDIgOS41NS04LjU1IDExIC44NiAxLjQ3IDMuOTItMS43MSAzLjkyLTEuNzFzOS41OC04LjcxIDExLjY2LTYuNDRjMi4wOCAyLjI3LTEuNTggNC4xNy02LjggNy4zMy01LjIzIDMuMTYtNS42MyA0LTQuOSA1LjIuNzUgMS4yIDEyLjI4LTguNTMgMTMuMzYtNC40IDEuMDggNC4xMS0xMS43NiA1LjMtMTAuOTcgOC4xNS44IDIuODUgOS4wNS01LjM4IDEwLjc0LTIuMTggMS42OSAzLjIxLTExLjY1IDYuOTgtMTEuNzYgNy4wMS00LjMxIDEuMTItMTUuMjYgMy40OS0xOS4wOC0yLjEyWiIvPjwvc3ZnPiA=" alt="Open in Spaces"></a>
</p>

<p align="center">
  <img src="assets/teaser.jpg" alt="AReA samples across FLUX.1, Qwen-Image, FLUX.2 and Ideogram 4" width="900"/>
</p>

**AReA** is a training-free attention mechanism for generating images beyond the training resolution of diffusion transformers. Attention's effective receptive field narrows during denoising, from global in early steps that form structure to local in later steps that refine details. AReA follows this progression in two phases:

- **Phase I, cross-scale coarse attention.** Queries stay at full resolution while keys and values keep one token per *f × f* block, placed on a grid close to the model's native resolution. Every token still sees the whole image, at 1/*f*² of the cost.
- **Phase II, adaptive windowed fine attention.** Attention runs inside local windows that shrink at every step down to the native size. Windows over smooth regions of a wavelet (DWT) detail map are split into quadrants, a narrow **Halo** of neighbouring tokens extends each window's keys and values, and a **global-KV bridge** adds distinctive tokens from the rest of the image.

AReA never computes full attention at the target resolution and needs no RoPE extrapolation, extra parameters or fine-tuning. It is model agnostic, and its speedup over full attention grows with resolution, up to 3.2× in our experiments.

## News

- **[2026-10]** Code released with FLUX.1 support, together with the HRGen-1K benchmark prompts.
- **Coming soon:** Qwen-Image, FLUX.2 and Ideogram 4.

## Supported models

| Model | `--model` | Default checkpoint | Status |
|---|---|---|---|
| FLUX.1 | `flux1` | [`black-forest-labs/FLUX.1-Krea-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) | Available |
| Qwen-Image | | | Coming soon |
| FLUX.2 | | | Coming soon |
| Ideogram 4 | | | Coming soon |

## Installation

```bash
git clone https://github.com/KooroshRH/AReA.git
cd AReA

# Install PyTorch for your CUDA version first, then:
pip install -r requirements.txt
```

Model weights are downloaded from Hugging Face on first use. Some checkpoints are gated: accept their license on Hugging Face and log in with `huggingface-cli login`.

## Quick start

To try AReA without installing anything, use the [Hugging Face demo](https://huggingface.co/spaces/kooroshrh/AReA) (FLUX.1, 2K to 4K).

```bash
python generate.py --model flux1 \
  --prompt "A vibrant village with colorful houses and a picturesque castle is set against a backdrop of mountains and a clear blue sky" \
  --height 4096 --width 4096 --seed 0
```

The image is written to `outputs/`. `--model` selects the model (see [Supported models](#supported-models)), and each model runs with its own default checkpoint, number of steps and guidance scale. `--checkpoint` loads another checkpoint of the same model, given as a Hugging Face repo id or a local path:

```bash
python generate.py --model flux1 --checkpoint black-forest-labs/FLUX.1-dev \
  --prompt "..." --height 4096 --width 4096
```

On GPUs with less than 60 GB of memory, idle models are offloaded to the CPU automatically.

## Using AReA in your own code

AReA attaches in place to a [diffusers](https://github.com/huggingface/diffusers) pipeline of a supported model: it replaces the transformer's attention processors and adds two hooks, and the pipeline is then called as usual. For example, with FLUX.1:

```python
import torch
from diffusers import FluxPipeline
from area import AReA

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
pipe.vae.enable_tiling()
pipe.to("cuda")              # or pipe.enable_model_cpu_offload()

area = AReA(pipe)

image = pipe(
    "A vibrant village with colorful houses and a picturesque castle",
    height=4096,
    width=4096,
    num_inference_steps=28,
    guidance_scale=4.5,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("village.png")

area.remove()                # back to the original attention
```

Canvases at or below the model's native resolution keep the original attention. Height and width must divide into whole patches, and the two patch counts must share a factor of at least 2, since Phase I pools *f × f* blocks.

## Options

| `AReA(...)` argument | `generate.py` flag | Default | Description |
|---|---|---|---|
| `pooling` | `--pooling` | `subsample` | Phase I: the token kept for each *f × f* block. `subsample` (block centre), `mean`, `medoid` or `random`. |
| `phase1_steps` | `--phase1-steps` | per model | Number of Phase I steps; each model has its own default. |
| `halo` | `--halo` | 8 | Largest Halo width δ<sub>max</sub>, in patches. |
| `bridge_ratio` | `--bridge-ratio` | 0.5 | Global-KV bridge ratio ρ: each window of side *w* gets (ρ*w*)² bridge tokens. |
| `seed` | `--seed` | 0 | Seed for window boundary jitter, Halo widths and `random` pooling. `generate.py` uses the generation seed. |

Other `generate.py` flags: `--model`, `--checkpoint`, `--height`, `--width`, `--steps`, `--guidance`, `--offload`, `--output-dir`, and `--prompt-file` to generate a whole prompt list. Run `python generate.py --help` for details and defaults.

## HRGen-1K

[`HRGen-1K/prompts.jsonl`](HRGen-1K/prompts.jsonl) holds the 1,000 prompts of HRGen-1K, our benchmark for high-resolution generation. The prompts span natural scenes, built environments, people, animals, objects and imaginative worlds, and target failure modes that single-subject prompts rarely reveal: complex scenes with many interacting subjects, spatial relations, explicit object counts, reflections and transparency, and 125 text-rendering prompts. Prompt length ranges from 9 to 91 words (32.5 on average). Each line is a JSON object with an `id` and a `prompt`:

```bash
python generate.py --model flux1 --prompt-file HRGen-1K/prompts.jsonl --height 4096 --width 4096 --seed 0
```

## Repository layout

```text
AReA/
├── area/
│   ├── schedule.py      # Phase I pooling ramp and Phase II window sizes
│   ├── attention.py     # cross-scale coarse attention and windowed fine attention
│   ├── tiling.py        # base tiling, DWT detail map, splits, Halo, merge weights
│   ├── bridge.py        # global-KV bridge token selection
│   └── backbones/       # one adapter per model family
├── generate.py          # command-line generation
├── HRGen-1K/            # benchmark prompts
└── assets/
```

The method lives in the model-independent modules; a backbone adapter only describes how a model lays out its tokens and positions, so supporting a new model means adding one adapter.

## AReA-Bootstrap

AReA-Bootstrap is available for FLUX.1 only. It replaces Phase I with a generation at the native resolution, as multi-stage methods do: the native-resolution latent is upsampled to the target size, re-noised to the level at which Phase II begins, and denoised with Phase II.

```bash
python generate.py --model flux1 --prompt "..." --height 4096 --width 4096 --bootstrap
```

```python
image = area.bootstrap("...", height=4096, width=4096, seed=0).images[0]
```

## Citation

```bibtex
@misc{rajabi2026area,
      title={AReA: Adaptive Receptive-Field Attention for Efficient High-Resolution Diffusion Transformers},
      author={Javad Rajabi and Koorosh Roohi and David B. Lindell and Babak Taati},
      year={2026},
      eprint={2610.17227},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.17227},
}
```
