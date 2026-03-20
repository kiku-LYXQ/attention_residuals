from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    create_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer

from .attnres_adapter import BlockAttnResAdapter, FullAttnResAdapter
from .attnres_state import BlockAttnResState, FullAttnResState
from .config import HfAttnResMode


class FullAttnResLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        past_key_values: Optional[Tensor],
        use_cache: Optional[bool],
        cache_position: Optional[Tensor],
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        adapter: FullAttnResAdapter,
        state: FullAttnResState,
        **kwargs,
    ) -> Tensor:
        attn_input, _ = adapter.before_attention(state, layer_idx=self.layer_idx)
        hidden_states = self.input_layernorm(attn_input)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = attn_input + hidden_states
        adapter.append_history(state, hidden_states)

        mlp_input, _ = adapter.before_mlp(state, layer_idx=self.layer_idx)
        hidden_states = self.post_attention_layernorm(mlp_input)
        hidden_states = self.mlp(hidden_states)
        hidden_states = mlp_input + hidden_states
        adapter.append_history(state, hidden_states)
        return hidden_states


class BlockAttnResLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
        position_ids: Optional[Tensor],
        past_key_values: Optional[Tensor],
        use_cache: Optional[bool],
        cache_position: Optional[Tensor],
        position_embeddings: Optional[tuple[Tensor, Tensor]],
        adapter: BlockAttnResAdapter,
        state: BlockAttnResState,
        **kwargs,
    ) -> Tensor:
        attn_input, _ = adapter.before_attention(state, layer_idx=self.layer_idx)
        hidden_states = self.input_layernorm(attn_input)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = attn_input + hidden_states
        adapter.update_partial(state, hidden_states)

        mlp_input, _ = adapter.before_mlp(state, layer_idx=self.layer_idx)
        hidden_states = self.post_attention_layernorm(mlp_input)
        hidden_states = self.mlp(hidden_states)
        hidden_states = mlp_input + hidden_states
        adapter.update_partial(state, hidden_states)
        return hidden_states


class AttnResLlamaModelBase(LlamaPreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig, mode: HfAttnResMode, block_size: int = 2) -> None:
        super().__init__(config)
        self.mode = mode
        self.block_size = block_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            FullAttnResLlamaDecoderLayer(config, layer_idx) if mode == HfAttnResMode.FULL else BlockAttnResLlamaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.adapter: FullAttnResAdapter | BlockAttnResAdapter = (
            FullAttnResAdapter(config.hidden_size)
            if mode == HfAttnResMode.FULL
            else BlockAttnResAdapter(config.hidden_size)
        )
        self.post_init()

    def get_initial_state(self, hidden_states: Tensor) -> FullAttnResState | BlockAttnResState:
        if self.mode == HfAttnResMode.FULL:
            return FullAttnResState(history=[hidden_states])
        block_state = BlockAttnResState(blocks=[hidden_states], partial_block=hidden_states.clone(), block_size=self.block_size)
        return block_state

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        cache_position: Optional[Tensor] = None,
        past_key_values: Optional[Any] = None,
        return_depth_weights: bool = False,
        **kwargs,
    ) -> Tuple[BaseModelOutput, dict] | BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        position_ids = position_ids if position_ids is not None else torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        cache_position = cache_position if cache_position is not None else torch.arange(hidden_states.shape[1], device=hidden_states.device)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        state = self.get_initial_state(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adapter=self.adapter,
                state=state,
            )
            if isinstance(state, BlockAttnResState):
                state.increment_layer()
                if state.block_size > 0 and state.layer_counter % state.block_size == 0:
                    state.blocks.append(state.partial_block.clone())
                    state.partial_block = state.partial_block.clone()
        hidden_states = self.norm(hidden_states)
        output = BaseModelOutput(last_hidden_state=hidden_states)
        if return_depth_weights:
            return output, state.stats.summary()
        return output


class HfAttnResCausalLM(nn.Module):
    def __init__(self, base_model: AttnResLlamaModelBase, vocab_size: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.output_proj = nn.Linear(base_model.config.hidden_size, vocab_size, bias=False)
        self.output_proj.weight = base_model.embed_tokens.weight

    def forward(self, *args, return_depth_weights: bool = False, **kwargs):
        kwargs = dict(kwargs)
        kwargs["return_depth_weights"] = return_depth_weights
        outputs = self.base_model(*args, **kwargs)
        if isinstance(outputs, tuple):
            outputs, stats = outputs
        else:
            stats = {}
        hidden = outputs.last_hidden_state
        logits = self.output_proj(hidden)
        return logits, stats
