"""Ideogram 4 with text features that cover only the text tokens.

diffusers' ``Ideogram4Pipeline`` pads the 53248-wide text features out to the whole packed
sequence, zeroes for the unconditional branch included, and its transformer normalises and
projects all of it in float32 at every step: about 30 GB per copy at 6144^2. These subclasses keep
the stock weights and loading and change only that path. The features stop at the text, the
unconditional transformer gets none, and the shared left padding is trimmed, so a single prompt
needs no attention mask. The result is the same as the stock pipeline's.
"""

from typing import Any, Callable

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_ideogram4 import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)
from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
    PROMPT_UPSAMPLE_TEMPERATURE,
    Ideogram4Pipeline,
    _expand_tensor_to_effective_batch,
    _logit_normal_sigmas,
    _resolution_aware_mu,
)
from diffusers.pipelines.ideogram4.pipeline_output import Ideogram4PipelineOutput
from diffusers.utils.peft_utils import apply_lora_scale


class SlimIdeogram4Transformer2DModel(Ideogram4Transformer2DModel):
    """``encoder_hidden_states`` covers the leading text region of the sequence, or is None."""

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        batch_size, seq_len, in_channels = hidden_states.shape
        if in_channels != self.in_channels:
            raise ValueError(f"Expected last dim {self.in_channels}, got {in_channels}.")

        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(hidden_states.dtype).unsqueeze(-1)
        hidden_states = self.input_proj(hidden_states * output_image_mask) * output_image_mask

        t_cond = self.t_embedding(timestep)
        if timestep.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        if encoder_hidden_states is not None:
            text_len = encoder_hidden_states.shape[1]
            text = indicator[:, :text_len] == LLM_TOKEN_INDICATOR
            rows = self.llm_cond_proj(self.llm_cond_norm(encoder_hidden_states[text]))
            cond = hidden_states.new_zeros(batch_size, text_len, hidden_states.shape[-1])
            cond[text] = rows.to(hidden_states.dtype)
            hidden_states[:, :text_len] = hidden_states[:, :text_len] + cond

        hidden_states = hidden_states + self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))

        cos, sin = self.rotary_emb(position_ids)
        image_rotary_emb = (cos.to(hidden_states.dtype), sin.to(hidden_states.dtype))

        if bool((segment_ids == segment_ids[:, :1]).all()):
            attention_mask = None
        else:
            attention_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        for block in self.layers:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, attention_mask, image_rotary_emb, adaln_input
                )
            else:
                hidden_states = block(hidden_states, attention_mask, image_rotary_emb, adaln_input)

        output = self.final_layer(hidden_states, conditioning=adaln_input)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class SlimIdeogram4Pipeline(Ideogram4Pipeline):
    """``Ideogram4Pipeline`` with the slim text path; load it the same way, e.g.
    ``SlimIdeogram4Pipeline.from_pretrained("ideogram-ai/ideogram-4-nf4-diffusers", torch_dtype=torch.bfloat16)``."""

    def __init__(self, scheduler, vae, text_encoder, tokenizer, transformer, unconditional_transformer,
                 prompt_enhancer_head=None) -> None:
        # the slim transformer adds no modules or weights, only its forward, so the loaded stock ones are re-classed
        for model in (transformer, unconditional_transformer):
            if type(model) is Ideogram4Transformer2DModel:
                model.__class__ = SlimIdeogram4Transformer2DModel
        super().__init__(scheduler, vae, text_encoder, tokenizer, transformer, unconditional_transformer,
                         prompt_enhancer_head)

    def encode_prompt(self, prompt, grid_h, grid_w, max_sequence_length, device):
        """As the stock method, but the features cover only the text region, not the image tokens."""
        features, _, _, indicator = super().encode_prompt(prompt, 0, 0, max_sequence_length, device)
        text_lengths = (indicator == LLM_TOKEN_INDICATOR).sum(dim=1).tolist()
        position_ids, segment_ids, indicator = self._prepare_ids(
            text_lengths, grid_h, grid_w, max_sequence_length, device)
        return features, position_ids, segment_ids, indicator

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        height: int = 2048,
        width: int = 2048,
        num_inference_steps: int = 48,
        guidance_scale: float | None = None,
        guidance_schedule: list[float] | torch.Tensor | None = (7.0,) * 45 + (3.0,) * 3,
        mu: float = 0.0,
        std: float = 1.5,
        prompt_upsampling: bool = False,
        prompt_upsampling_temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
        max_sequence_length: int = 2048,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> Ideogram4PipelineOutput | tuple[Any]:
        """Same arguments and result as ``Ideogram4Pipeline.__call__``."""
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt_upsampling:
            prompt = self.upsample_prompt(
                prompt,
                height=height,
                width=width,
                temperature=prompt_upsampling_temperature,
                max_new_tokens=max_sequence_length,
                generator=generator,
                device=device,
            )

        grid_h = height // (self.vae_scale_factor * self.patch_size)
        grid_w = width // (self.vae_scale_factor * self.patch_size)
        num_image_tokens = grid_h * grid_w

        llm_features, position_ids, segment_ids, indicator = self.encode_prompt(
            prompt=prompt, grid_h=grid_h, grid_w=grid_w, max_sequence_length=max_sequence_length, device=device,
        )
        llm_features = _expand_tensor_to_effective_batch(llm_features, batch_size, num_images_per_prompt)
        position_ids = _expand_tensor_to_effective_batch(position_ids, batch_size, num_images_per_prompt)
        segment_ids = _expand_tensor_to_effective_batch(segment_ids, batch_size, num_images_per_prompt)
        indicator = _expand_tensor_to_effective_batch(indicator, batch_size, num_images_per_prompt)

        # padding columns shared by every prompt sit in their own segment, so dropping them is exact
        first = int((segment_ids != SEQUENCE_PADDING_INDICATOR).any(dim=0).to(torch.long).argmax())
        llm_features = llm_features[:, first:]
        position_ids, segment_ids, indicator = position_ids[:, first:], segment_ids[:, first:], indicator[:, first:]
        text_len = position_ids.shape[1] - num_image_tokens

        # the unconditional branch is the image region alone, with no text features
        neg_position_ids = position_ids[:, text_len:]
        neg_segment_ids = segment_ids[:, text_len:]
        neg_indicator = indicator[:, text_len:]

        schedule_mu = _resolution_aware_mu(height=height, width=width, base_mu=mu)
        sigmas = _logit_normal_sigmas(num_inference_steps, schedule_mu, std=std, device=device)
        self.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        if guidance_scale is not None:
            guidance_schedule = [float(guidance_scale)] * num_inference_steps
        gw = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)

        latent_dim = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_image_tokens=num_image_tokens,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        text_z_padding = torch.zeros(batch_size * num_images_per_prompt, text_len, latent_dim,
                                     dtype=torch.float32, device=device)
        llm_features = llm_features.to(self.transformer.dtype)

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                t_model = 1.0 - (t.float() / num_train_timesteps)
                t_model = t_model.expand(batch_size * num_images_per_prompt).to(self.transformer.dtype)

                pos_out = self.transformer(
                    hidden_states=torch.cat([text_z_padding, latents], dim=1).to(self.transformer.dtype),
                    timestep=t_model,
                    encoder_hidden_states=llm_features,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    indicator=indicator,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                pos_v = pos_out[:, text_len:].to(torch.float32)

                neg_v = self.unconditional_transformer(
                    hidden_states=latents.to(self.unconditional_transformer.dtype),
                    timestep=t_model,
                    encoder_hidden_states=None,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0].to(torch.float32)

                self._guidance_scale = guidance_schedule[i]
                v = gw[i] * pos_v + (1.0 - gw[i]) * neg_v
                latents = self.scheduler.step(-v, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            z = latents
            bn_mean = self.vae.bn.running_mean.view(1, 1, -1).to(device=z.device, dtype=z.dtype)
            bn_std = torch.sqrt(self.vae.bn.running_var + self.vae.config.batch_norm_eps).view(1, 1, -1)
            z = z * bn_std.to(device=z.device, dtype=z.dtype) + bn_mean

            patch = self.patch_size
            ae_channels = z.shape[-1] // (patch * patch)
            z = z.view(batch_size * num_images_per_prompt, grid_h, grid_w, patch, patch, ae_channels)
            z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
            z = z.view(batch_size * num_images_per_prompt, ae_channels, grid_h * patch, grid_w * patch)

            decoded = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(decoded.float(), output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return Ideogram4PipelineOutput(images=image)
