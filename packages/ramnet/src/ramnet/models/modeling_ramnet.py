from typing import Any

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, GatedMLP, RMSNorm
from fla.modules.l2warp import l2_warp

from .config_ramnet import RAMNetConfig
from ..layers import RAMCache, RAMNetAttention, RAMState

logger = logging.get_logger(__name__)


class RAMNetBlock(GradientCheckpointingLayer):
    """Pre-norm RAM attention followed by an MLP."""

    def __init__(self, config: RAMNetConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.fuse_norm = config.fuse_norm

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = RAMNetAttention(
            hidden_size=config.hidden_size, n_heads=config.num_attention_heads,
            addr_order=config.addr_order, addr_radix=config.addr_radix, value_chs=config.value_chs,
            gamma=config.gamma, gamma_mode=config.gamma_mode,
            init_mass=config.init_mass,
            embed_num=config.get_embed_num(layer_idx),
            addr_temperature=config.addr_temperature, top_k=config.addr_top_k,
            top_k_re_norm=config.addr_top_k_re_norm,
            mass_aware=config.mass_aware, output_gate=config.output_gate, output_norm=config.output_norm,
            addr_proj_bias=config.addr_proj_bias, value_proj_bias=config.value_proj_bias,
            address_scaling=config.address_scaling,
            ram_attn_eps=config.ram_attn_eps,
            layer_idx=layer_idx,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if config.hidden_act in ('fuse_SwiGLU', 'SwiGLU'):
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                hidden_ratio=config.hidden_ratio,
                intermediate_size=config.intermediate_size,
                hidden_act='swish',
                fuse_swiglu=config.hidden_act == 'fuse_SwiGLU',
            )
        else:
            match config.hidden_act:
                case 'GELU':
                    activation = nn.GELU
                case 'SiLU' | 'Swish':
                    activation = nn.SiLU
                case _:
                    raise ValueError(f'Unsupported hidden_act: {config.hidden_act}')

            intermediate_size = config.intermediate_size
            if intermediate_size is None:
                hidden_ratio = config.hidden_ratio or 4
                intermediate_size = int(config.hidden_size * hidden_ratio)

            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_size),
                activation(),
                nn.Linear(intermediate_size, config.hidden_size),
            )

    def _forward_mlp(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
    ) -> torch.Tensor:
        if self.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, prenorm=True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: RAMCache | None = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, RAMCache | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_ids=position_ids,
        )

        hidden_states = self._forward_mlp(hidden_states, residual)
        return hidden_states, attentions, past_key_values

    def forward_with_state(
            self,
            hidden_states: torch.Tensor,
            init_state: RAMState | tuple[torch.Tensor, torch.Tensor] | None = None,

            mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,

            t_dim: int = 1,
            inplace: bool = False,
    ) -> tuple[torch.Tensor, RAMState]:
        """Run the complete block with explicit RAM state and a custom time axis."""
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, state = self.attn.forward_with_state(
            hidden_states,
            init_ram=init_state,
            mask=mask,
            position_ids=position_ids,
            t_dim=t_dim,
            inplace=inplace,
        )

        hidden_states = self._forward_mlp(hidden_states, residual)
        return hidden_states, state


class RAMNetPreTrainedModel(PreTrainedModel):
    config_class = RAMNetConfig
    base_model_prefix = 'model'
    _no_split_modules = ['RAMNetBlock']
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_packed_sequence = False
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        # HF traverses modules once; initialize only the current module's parameters.
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, RMSNorm):
            module.reset_parameters()
        else:
            RAMNetAttention._init_module_weights(
                module,
                initializer_range=self.config.initializer_range,
                addr_prior_range=self.config.addr_prior_range,
                gate_initializer_bias=self.config.gate_initializer_bias,
            )


class RAMNetModel(RAMNetPreTrainedModel):
    """RAM-Net backbone with the Hugging Face decoder interface."""

    def __init__(self, config: RAMNetConfig) -> None:
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.layers = nn.ModuleList([RAMNetBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embeddings = value

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: RAMCache | None = None,
            inputs_embeds: torch.Tensor | None = None,
            cache_position: torch.Tensor | None = None,

            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if output_attentions:
            raise ValueError('RAMNet does not materialize attention matrices.')
        if kwargs.get('cu_seqlens') is not None:
            raise ValueError('RAMNet does not support packed cu_seqlens; use a padded batch and attention_mask.')
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Specify exactly one of input_ids and inputs_embeds.')
        if past_key_values is not None and not isinstance(past_key_values, RAMCache):
            raise TypeError('past_key_values must be a RAMCache, not a KV cache.')

        if self.gradient_checkpointing and self.training:
            if past_key_values is not None and past_key_values.get_seq_length():
                raise ValueError('Gradient checkpointing requires an uncached input sequence.')
            if use_cache:
                logger.warning_once('use_cache=True is incompatible with gradient checkpointing; disabling cache.')
            use_cache = False
            past_key_values = None

        hidden_states = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if hidden_states.ndim != 3 or hidden_states.size(1) == 0:
            raise ValueError('Model inputs must have shape (B, T, D) with T > 0.')
        if use_cache and past_key_values is None:
            past_key_values = RAMCache()

        # cache_position counts physical tokens; each RAM layer tracks valid CAPE positions.
        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, _, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,
            )
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )
        return outputs if return_dict else outputs.to_tuple()


class RAMNetForCausalLM(RAMNetPreTrainedModel, GenerationMixin):
    """RAM-Net with a causal language-modeling head and recurrent generation."""

    _tied_weights_keys = {'lm_head.weight': 'model.embeddings.weight'}

    def __init__(self, config: RAMNetConfig) -> None:
        super().__init__(config)
        self.model = RAMNetModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.fuse_linear_cross_entropy:
            self.criterion = FusedLinearCrossEntropyLoss(use_l2warp=config.use_l2warp)
        elif config.fuse_cross_entropy:
            self.criterion = FusedCrossEntropyLoss(inplace_backward=True)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embeddings = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: RAMNetModel) -> None:
        self.model = decoder

    def get_decoder(self) -> RAMNetModel:
        return self.model

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        # RAMNetModel creates its recurrent cache during the first forward.
        return False

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: RAMCache | None = None,
            inputs_embeds: torch.Tensor | None = None,
            cache_position: torch.Tensor | None = None,
            labels: torch.LongTensor | None = None,

            use_cache: bool | None = None,
            output_attentions: bool | None = None,
            output_hidden_states: bool | None = None,
            return_dict: bool | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError('logits_to_keep must be non-negative.')
            logit_indices = slice(-logits_to_keep, None)
        elif isinstance(logits_to_keep, torch.Tensor):
            logit_indices = logits_to_keep.to(hidden_states.device)
        else:
            raise TypeError('logits_to_keep must be an int or a tensor of sequence indices.')
        logit_hidden_states = hidden_states[:, logit_indices]

        logits = None
        if not self.config.fuse_linear_cross_entropy or labels is None:
            logits = self.lm_head(logit_hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != hidden_states.shape[:2]:
                raise ValueError('labels must match the (B, T) input shape.')
            criterion = self.criterion

            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), dim=-1)
            labels = labels[:, logit_indices]
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(logit_hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                if self.config.use_l2warp:
                    loss = l2_warp(loss, logits)

        result = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return result if return_dict else result.to_tuple()
