import math
import re
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaForCausalLM,
    LLAMA_ATTENTION_CLASSES,
)

logger = logging.get_logger(__name__)


@dataclass
class Interval:
    start: int
    end: int


def _merge_intervals(intervals: List[Interval]) -> List[Interval]:
    merged = []
    if len(intervals) == 0:
        return merged

    intervals.sort(key=lambda r: (r.start, r.end))
    latest = intervals[0]
    merged.append(latest)
    for i in intervals[1:]:
        if i.start <= latest.end:
            latest.end = i.end
        else:
            latest = i
            merged.append(latest)

    return merged


def _canonicalize_list(v, length=1):
    if isinstance(v, List):
        assert len(v) == length
        return v
    else:
        return [v] * length


svd_init_method_regex = re.compile("^svd_(\d+)$")


class MoEConfig(LlamaConfig):
    model_type = "MoE"
    def __init__(
        self,
        num_experts: int | List[int] = 0,
        expert_rank: int | List[int] = 32,
        expert_alpha: Optional[int] | List[Optional[int]] = None,
        expert_init_method: str | List[str] = "svd_0",
        default_experts: List[int] = [],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_experts: int = num_experts
        self.expert_rank = expert_rank
        self.expert_alpha = expert_alpha
        self.expert_init_method = expert_init_method
        self.default_experts = default_experts


class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, alpha: Optional[int], rank: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.alpha = alpha
        self.rank = rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=bias)
        self.scaling = 1. if alpha is None else (alpha / math.sqrt(rank))

    def forward(self, x) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


class LinearMoE(nn.Module):
    def __init__(
        self,
        shared_linear: nn.Linear,
        config: MoEConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.shared_linear = shared_linear
        self._init_experts()

    def _init_experts(self):
        if self.config.num_experts < 0:
            raise ValueError("Number of experts must be non-negative")

        if self.config.num_experts == 0:
            self.experts = nn.ModuleList()
            return

        expert_alphas = _canonicalize_list(self.config.expert_alpha, self.config.num_experts)
        expert_ranks = _canonicalize_list(self.config.expert_rank, self.config.num_experts)
        expert_init_methods = _canonicalize_list(self.config.expert_init_method, self.config.num_experts)

        dtype = self.shared_linear.weight.dtype
        self.experts = nn.ModuleList([
            LoRA(
                in_features=self.shared_linear.in_features,
                out_features=self.shared_linear.out_features,
                bias=self.config.mlp_bias,
                alpha=alpha,
                rank=rank,
            ).to(dtype).to(self.shared_linear.weight.device)
            for alpha, rank in zip(expert_alphas, expert_ranks)
        ])

        V, S, Uh = None, None, None
        svd_intervals = []
        for idx, init_method in enumerate(expert_init_methods):
            if init_method == "none":
                continue

            expert = self.experts[idx]
            if match := svd_init_method_regex.search(init_method):
                if V is None or S is None or Uh is None:
                    # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel}
                    self.shared_linear.weight.data = self.shared_linear.weight.data.to(torch.float32)
                    V, S, Uh = torch.linalg.svd(self.shared_linear.weight.data, full_matrices=False)

                start = int(match.group(1)) # Get starting rank
                end = start + expert.rank

                Vr = V[:, start : end] # (out_channel, rank)
                Sr = torch.sqrt(torch.div(S[start : end], expert.scaling)) # (rank,)
                Uhr = Uh[start : end] # (rank, in_channel)

                svd_intervals.append(Interval(start, end))

                lora_A = (Sr.unsqueeze(-1) * Uhr).contiguous()
                lora_B = (Sr.unsqueeze(0) * Vr).contiguous()

                expert.lora_A.weight.data = lora_A.to(dtype)
                expert.lora_B.weight.data = lora_B.to(dtype)
                if self.config.mlp_bias:
                    nn.init.zeros_(expert.lora_B.bias)

            elif init_method == "std":
                scaled_std = self.config.initializer_range / math.sqrt(expert.scaling)
                nn.init.normal_(expert.lora_A.weight, mean=0., std=scaled_std)
                nn.init.normal_(expert.lora_B.weight, mean=0., std=scaled_std)
                if self.config.mlp_bias:
                    nn.init.zeros_(expert.lora_B.bias)

            elif init_method == "std_zero":
                scaled_std = self.config.initializer_range / expert.scaling
                nn.init.normal_(expert.lora_A.weight, mean=0., std=scaled_std)
                nn.init.zeros_(expert.lora_B.weight)
                if self.config.mlp_bias:
                    nn.init.zeros_(expert.lora_B.bias)

            else:
                assert False

        svd_intervals = _merge_intervals(svd_intervals)
        diff = 0.
        for interval in svd_intervals:
            start, end = interval.start, interval.end
            Vr = V[:, start : end] # (out_channel, rank)
            Sr = S[start : end] # (rank,)
            Uhr = Uh[start : end] # (rank, in_channel)
            diff += Sr * Vr @ Uhr

        self.shared_linear.weight.data = (
            self.shared_linear.weight.data.to(torch.float32) - diff
        ).to(dtype)

    def forward(self, x: torch.Tensor, expert_indices: Optional[torch.Tensor]) -> torch.Tensor:
        x_shape = x.shape
        y_shape = x_shape[:-1] + (-1,)
        flattened_x = x.reshape(-1, x_shape[-1])
        flattened_output = self.shared_linear(flattened_x)

        if expert_indices is None:
            for idx in self.config.default_experts:
                flattened_output += self.experts[idx](flattened_x)
        else:
            n_experts_per_sample = expert_indices.size(-1)
            while expert_indices.ndim < len(x_shape):
                expert_indices = expert_indices.unsqueeze(-2)
            flattened_indices = expert_indices.expand(*x_shape[:-1], -1).reshape(-1, n_experts_per_sample)

            for i, expert in enumerate(self.experts):
                expert_mask = (flattened_indices == i).any(dim=-1)
                if expert_mask.any():
                    expert_input = flattened_x[expert_mask]
                    flattened_output[expert_mask] += expert(expert_input)

        return flattened_output.reshape(y_shape)


class MLPMoE(nn.Module):
    def __init__(
        self,
        config: MoEConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self._init_experts(LlamaMLP(config))

    def _init_experts(self, shared_mlp: LlamaMLP) -> None:
        self.act_fn = shared_mlp.act_fn
        self.gate_proj = LinearMoE(shared_mlp.gate_proj, self.config)
        self.up_proj = LinearMoE(shared_mlp.up_proj, self.config)
        self.down_proj = LinearMoE(shared_mlp.down_proj, self.config)

    def forward(self, x: torch.Tensor, expert_indices: Optional[torch.Tensor]) -> torch.Tensor:
        gate_y = self.act_fn(self.gate_proj(x, expert_indices))
        up_y = self.up_proj(x, expert_indices)
        y = self.down_proj(gate_y * up_y, expert_indices)
        return y


class MoEDecoderLayer(nn.Module):
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp_moe = MLPMoE(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp_moe(hidden_states, expert_indices)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MoEPreTrainedModel(PreTrainedModel):
    config_class = MoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MoEModel(MoEPreTrainedModel):
    def __init__(self, config: MoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    expert_indices,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    expert_indices=expert_indices,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class MoEForCausalLM(MoEPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            expert_indices=expert_indices,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        expert_indices=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        if expert_indices is not None:
            # Canonicalize expert_indices
            if expert_indices.ndim < input_ids.ndim + 1:
                while True:
                    expert_indices = expert_indices.unsqueeze(-2)
                    if expert_indices.ndim == input_ids.ndim + 1:
                        expert_indices = expert_indices.expand(*input_ids.shape, -1)
                        break

        past_length = 0
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            input_ids_len = input_ids.shape[1]
            if attention_mask is not None and attention_mask.shape[1] > input_ids_len:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                if expert_indices is not None:
                    expert_indices = expert_indices[:, -(attention_mask.shape[1] - past_length) : input_ids_len]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids_len:
                input_ids = input_ids[:, past_length:]
                if expert_indices is not None:
                    if (expert_len := expert_indices.shape[1]) < input_ids_len:
                        # Repeat last expert indices
                        last_expert = expert_indices[:, -1:, :]
                        if expert_len <= past_length:
                            expert_indices = last_expert.expand(*input_ids.shape, -1)
                        else:
                            expert_indices = torch.cat([expert_indices[past_length:]] + [last_expert] * (input_ids_len - past_length), dim=1)
                    else:
                        expert_indices = expert_indices[:, past_length:input_ids_len]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            else:
                expert_indices = expert_indices[:, : input_ids_len]

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "expert_indices": expert_indices}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            if expert_indices is not None:
                expert_indices = expert_indices.contiguous()
            model_inputs = {"input_ids": input_ids.contiguous(), "expert_indices": expert_indices}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


    @classmethod
    def from_pretrained_llama(
        cls,
        directory,
        num_experts=0,
        expert_rank=32,
        expert_alpha=None,
        expert_init_method="svd_0",
        default_experts=[],
        *args,
        **kwargs,
    ):
        donor = LlamaForCausalLM.from_pretrained(directory, *args, **kwargs)
        recipient = MoEForCausalLM.from_pretrained(
            directory,
            num_experts=num_experts,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
            expert_init_method=expert_init_method,
            default_experts=default_experts,
            *args,
            **kwargs
        )
        config = recipient.config

        def transfer_layer_weights(donor, recipient):
            recipient.self_attn = donor.self_attn
            recipient.self_attn.config = config

            recipient.input_layernorm = donor.input_layernorm
            recipient.post_attention_layernorm = donor.post_attention_layernorm

            recipient.mlp_moe._init_experts(donor.mlp)

        recipient.model.embed_tokens = donor.model.embed_tokens
        recipient.model.norm = donor.model.norm
        for i in range(len(donor.model.layers)):
            transfer_layer_weights(donor.model.layers[i], recipient.model.layers[i])
        recipient.lm_head = donor.lm_head

        del donor
        return recipient

from transformers import (
    AutoConfig,
    LlamaTokenizerFast,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

AutoConfig.register("MoE", MoEConfig)
AutoTokenizer.register(MoEConfig, fast_tokenizer_class=LlamaTokenizerFast)
AutoModel.register(MoEConfig, MoEModel)
AutoModelForCausalLM.register(MoEConfig, MoEForCausalLM)
