import torch
import einops
from typing import Literal
from torch.distributed.tensor import DTensor, distribute_tensor

from .cache import RAMCache, RAMState
from ramnet_ops import random_access_memory_attention

Activation_mapping = {
    'relu': torch.nn.ReLU,
    'sigmoid': torch.nn.Sigmoid,
    'silu': torch.nn.SiLU,
    'swish': torch.nn.SiLU,
    'gelu': torch.nn.GELU,
}
OutputGateType = Literal['relu', 'sigmoid', 'silu', 'swish', 'gelu']


class RAMNetAttention(torch.nn.Module):
    """
    the layer of RAM Net Attention for [RAM-Net: Expressive Linear Attention with Selectively Addressable Memory] (https://arxiv.org/abs/2602.11958)

    :param hidden_size: Dimensionality of the input and output layers.
    :param n_heads: Number of attention heads.

    :param addr_order: Order U of the address space (U-order product softmax).
    :param addr_radix: Address radix P (paper d_p): choices per digit, giving P^U memory slots.
    :param value_chs: Dimensionality of value channels (d_v), total state size = d_p^U * d_v. Default: hidden_size // num_attention_heads.

    :param gamma: prior forget-rate control. Negative values favor retention, zero balances writing and forgetting, and positive values favor forgetting.
    :param gamma_mode: Mode for gamma: 'fixed' (non-trainable constant), 'learnable'  (per-head learnable parameter), 'dynamic' (input-conditioned via linear map).

    :param init_mass: Persistence strength of the initial zero state. None defaults to 1 / (addr_radix ** addr_order).

    :param embed_num: Number of heads applying Cyclic Address Positional Embedding (CAPE). 0 disables CAPE; explicit values must be in ``[0, n_heads]``.

    :param addr_temperature: Temperature for address localization; smaller values precise memory access.
    :param top_k: Limits sparsity of the address vector. A two-item tuple/list applies separately to (write_k, read_k); int applies to both. Active state per token is proportional to k * d_v.
    :param top_k_re_norm: Whether to re-normalize the address vector after top-k filtering, forcing the top-k entries to sum to 1.0.
        A bool applies to both write and read; a two-item tuple/list specifies separate (write, read) bools.

    :param mass_aware: Weight reads by updated memory mass with epsilon-stabilized normalization.
    :param output_gate: Element-wise output gate function: 'relu', 'sigmoid', 'silu', 'swish', 'gelu', or None.
    :param output_norm: Whether to enable head-wise output RMS-norm.

    :param addr_proj_bias: Whether to enable bias in address projection layers (bias = address prior).
    :param value_proj_bias: Whether to enable bias in the value projection layer.
    :param address_scaling: Whether to re-parameterize address projection with learnable scaling (w = exp(alpha) * w' for training).

    :param ram_attn_eps: Epsilon value for RAM read out and proxy backward limit in the forget gate.
    """

    # noinspection PyTypeChecker
    def __init__(
            self,
            # === Model Architecture ===
            hidden_size: int,
            n_heads: int,
            # === RAM Structure ===
            addr_order: int = 5,
            addr_radix: int = 4,
            value_chs: int | None = None,
            # === Forgetting Modulation ===
            gamma: float = 2.197,
            gamma_mode: Literal['fixed', 'learnable', 'dynamic'] = 'dynamic',
            # === RAM initial mass ===
            init_mass: float | None = None,
            # === Positional Encoding (CAPE) ===
            embed_num: int = 0,
            # === Address Decoder ===
            addr_temperature: float = 2.2,
            top_k: tuple[int, int] | list[int] | int | None = 8,
            top_k_re_norm: bool | tuple[bool, bool] | list[bool] = False,
            # === Readout ===
            mass_aware: bool = False,
            output_gate: OutputGateType | None = 'sigmoid',
            output_norm: bool = False,
            # === Projection ===
            addr_proj_bias: bool = True,
            value_proj_bias: bool = False,
            address_scaling: bool = True,
            # === Numerical Stability ===
            ram_attn_eps: float = 1e-12,
            layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        if gamma_mode not in ('fixed', 'learnable', 'dynamic'):
            raise ValueError(f"Unsupported gamma_mode: {gamma_mode!r}.")
        if output_gate is not None and output_gate not in Activation_mapping:
            raise ValueError(f"Unsupported output_gate: {output_gate!r}.")
        assert hidden_size > 0 and n_heads > 0 and addr_order > 0 and addr_radix > 0, f"Expected positive dimensions, got {hidden_size=}, {n_heads=}, {addr_order=}, {addr_radix=}."
        assert addr_temperature > 0, f"Expected addr_temperature > 0, got {addr_temperature}."
        assert isinstance(embed_num, int) and 0 <= embed_num <= n_heads, f"Expected embed_num in [0, {n_heads}], got {embed_num!r}."
        assert ram_attn_eps > 0, f"Expected ram_attn_eps > 0, got {ram_attn_eps}."
        if layer_idx is not None and layer_idx < 0:
            raise ValueError('layer_idx must be non-negative.')

        if value_chs is None:
            value_chs = int(hidden_size // n_heads)
        assert value_chs > 0, f"Expected value_chs > 0, got {value_chs}."

        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.addr_order = addr_order
        self.addr_radix = addr_radix
        self.value_chs = value_chs
        self.gamma = gamma
        self.gamma_mode = gamma_mode
        self.addr_temperature = addr_temperature
        self.embed_num = embed_num
        self.init_mass = addr_radix ** (- addr_order) if init_mass is None else init_mass
        self.top_k = top_k
        self.top_k_re_norm = top_k_re_norm
        self.mass_aware = mass_aware
        self.addr_proj_bias = addr_proj_bias
        self.value_proj_bias = value_proj_bias
        self.ram_attn_eps = ram_attn_eps

        proj_v_chs: int = self.n_heads * self.value_chs
        proj_addr_chs: int = self.n_heads * self.addr_radix * self.addr_order

        self.r_proj = torch.nn.Linear(hidden_size, proj_addr_chs, bias=self.addr_proj_bias)
        self.w_proj = torch.nn.Linear(hidden_size, proj_addr_chs, bias=self.addr_proj_bias)
        self.r_proj._ramnet_address = (self.n_heads, self.addr_order, self.addr_radix)
        self.w_proj._ramnet_address = (self.n_heads, self.addr_order, self.addr_radix)
        self.v_proj = torch.nn.Linear(hidden_size, proj_v_chs, bias=self.value_proj_bias)
        self.o_proj = torch.nn.Linear(proj_v_chs, hidden_size, bias=self.value_proj_bias)

        if gamma_mode == 'learnable':
            self.gamma_modifier = torch.nn.Parameter(torch.zeros(self.n_heads))
        if gamma_mode == 'dynamic':
            self.gamma_modifier = torch.nn.Linear(hidden_size, self.n_heads)

        self.o_norm = torch.nn.RMSNorm(self.value_chs) if output_norm else None
        self.o_gate = torch.nn.Sequential(torch.nn.Linear(hidden_size, proj_v_chs), Activation_mapping[output_gate]()) if output_gate else None
        if self.o_gate is not None:
            self.o_gate[0]._ramnet_output_gate = True

        if address_scaling:
            self.r_scale = torch.nn.Parameter(torch.zeros(self.n_heads, self.addr_order))
            self.w_scale = torch.nn.Parameter(torch.zeros(self.n_heads, self.addr_order))
        else:
            self.r_scale = None
            self.w_scale = None

    @torch.no_grad()
    def _init_weights(self, *, initializer_range: float = 0.02, addr_prior_range: float = 1.5, gate_initializer_bias: float = 3.0) -> None:
        """Initialize all parameters and submodules owned by this attention layer."""
        for module in self.modules():
            self._init_module_weights(
                module,
                initializer_range=initializer_range,
                addr_prior_range=addr_prior_range,
                gate_initializer_bias=gate_initializer_bias,
            )

    @staticmethod
    @torch.no_grad()
    def _init_module_weights(
            module: torch.nn.Module,
            *,
            initializer_range: float,
            addr_prior_range: float,
            gate_initializer_bias: float,
    ) -> None:
        """Initialize one module; leave traversal and initialization tracking to the caller."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                address_shape = getattr(module, '_ramnet_address', None)
                if address_shape is not None:
                    H, U, P = address_shape
                    assert module.out_features == H * U * P, "Address projection shape mismatch."
                    bias = torch.linspace(0, addr_prior_range, U)[:, None] * torch.linspace(-1, 1, P)[None, :]
                    bias = bias.expand(H, U, P).flatten()
                    if isinstance(module.bias, DTensor):
                        bias = distribute_tensor(bias, module.bias.device_mesh, module.bias.placements)
                        module.bias.to_local().copy_(bias.to_local())
                    else:
                        module.bias.copy_(bias.to(module.bias))
                elif getattr(module, '_ramnet_output_gate', False):
                    torch.nn.init.constant_(module.bias, gate_initializer_bias)
                else:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.RMSNorm):
            module.reset_parameters()
        elif isinstance(module, RAMNetAttention):
            for parameter in (module.r_scale, module.w_scale, getattr(module, 'gamma_modifier', None)):
                if isinstance(parameter, torch.nn.Parameter):
                    torch.nn.init.zeros_(parameter)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: RAMCache | None = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None, RAMCache | None]:
        """Run attention on (B, T, D) inputs using the model cache protocol."""
        if hidden_states.ndim != 3:
            raise ValueError('forward expects (B, T, D); use forward_with_state for other layouts.')
        if output_attentions:
            raise ValueError('RAMNetAttention does not materialize attention matrices.')
        if use_cache and self.layer_idx is None:
            raise ValueError('layer_idx is required when use_cache=True.')

        if use_cache and past_key_values is None:
            past_key_values = RAMCache()
        cache_layer = None
        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError('layer_idx is required when reading a cache.')
            if len(past_key_values) > self.layer_idx:
                cache_layer = past_key_values.get_layer(self.layer_idx)

        B, T, _ = hidden_states.shape
        if T == 0:
            raise ValueError('The sequence must contain at least one token.')
        device = hidden_states.device
        if cache_layer is not None and cache_layer.next_position is not None:
            offset = cache_layer.next_position
        else:
            cache_length = cache_layer.get_seq_length() if cache_layer is not None else 0
            offset = torch.full((B, 1), cache_length, device=device, dtype=torch.long)

        mask = None
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.size(-1) < T:
                raise ValueError('attention_mask must be a (B, T) token mask or include the cached prefix.')
            attention_mask = attention_mask.to(device=device, dtype=torch.bool).expand(B, -1)
            mask = attention_mask[:, -T:]

        if position_ids is not None:
            if position_ids.ndim not in (1, 2) or position_ids.size(-1) < T:
                raise ValueError('position_ids must cover the current query.')
            position_ids = position_ids.to(device=device, dtype=torch.long)[..., -T:].expand(B, T)
        elif attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1)[:, -T:] - 1
            if attention_mask.size(-1) == T:  # in case mask don't include prefix
                position_ids = position_ids + offset
        else:
            position_ids = offset + torch.arange(T, device=device)

        if mask is not None:
            position_ids = position_ids.masked_fill(~mask, 0)

        output, state = self.forward_with_state(
            hidden_states,
            init_ram=None if cache_layer is None else cache_layer.state,
            mask=mask,
            position_ids=position_ids,
            inplace=None if use_cache else False,
        )

        if use_cache:
            next_position = position_ids[:, -1:] + 1
            if mask is not None:
                # Keep the last valid position, including for fully masked chunks.
                indices = torch.arange(T, device=device).expand(B, T).masked_fill(~mask, -1)
                last = indices.amax(-1, keepdim=True)
                next_position = torch.where(last >= 0, position_ids.gather(-1, last.clamp_min(0)) + 1, offset)
            past_key_values.update(
                recurrent_state=state,
                layer_idx=self.layer_idx,
                offset=T,
                cache_kwargs={'next_position': next_position},
            )

        return output, None, past_key_values if use_cache else None

    def forward_with_state(
            self,
            hidden_states: torch.Tensor,
            init_ram: RAMState | tuple[torch.Tensor, torch.Tensor] | None = None,

            mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,

            t_dim: int = 1,
            inplace: bool | None = False,
    ) -> tuple[torch.Tensor, RAMState]:
        """Return output and explicit state, without creating or updating a cache.

        A (slot, mass) pair initializes RAM directly.
        t_dim indexes a non-channel axis of hidden_states and its metadata.
        """
        if not -(hidden_states.ndim - 1) <= t_dim < hidden_states.ndim - 1:
            raise ValueError('t_dim must index a non-channel axis of hidden_states.')
        if hidden_states.size(t_dim) == 0:
            raise ValueError('The sequence must contain at least one token.')
        match init_ram:
            case None:
                init_slot, init_mass = None, self.init_mass
            case (init_slot, init_mass):
                pass
            case _:
                raise TypeError('init_ram must be None or a (slot, mass) pair.')

        # project address and value
        r_addr = einops.rearrange(self.r_proj(hidden_states), '... (h u p) -> ... h u p', h=self.n_heads, u=self.addr_order, p=self.addr_radix)
        w_addr = einops.rearrange(self.w_proj(hidden_states), '... (h u p) -> ... h u p', h=self.n_heads, u=self.addr_order, p=self.addr_radix)
        values = einops.rearrange(self.v_proj(hidden_states), '... (h c) -> ... h c', h=self.n_heads)

        # apply address scale and temperature
        r_addr = (torch.exp(self.r_scale) / self.addr_temperature).to(dtype=r_addr.dtype).unsqueeze(-1) * r_addr if self.r_scale is not None else r_addr / self.addr_temperature
        w_addr = (torch.exp(self.w_scale) / self.addr_temperature).to(dtype=w_addr.dtype).unsqueeze(-1) * w_addr if self.r_scale is not None else w_addr / self.addr_temperature

        # project forgetting modulator
        gamma = self.gamma
        if self.gamma_mode == 'learnable':
            gamma = (self.gamma_modifier + gamma).to(dtype=values.dtype).expand(*values.shape[:-1])
        if self.gamma_mode == 'dynamic':
            gamma = self.gamma_modifier(hidden_states) + gamma

        if inplace is None:
            inplace = (
                    not self.training
                    and init_slot is not None
                    and all(not value.requires_grad for value in (r_addr, w_addr, values, gamma, init_slot, init_mass) if isinstance(value, torch.Tensor))
            )

        # run RAM-Attn
        outputs, (slot, mass) = random_access_memory_attention(
            w_addr=w_addr, r_addr=r_addr,
            values=values, gamma=gamma,
            init_ram=(init_slot, init_mass),
            mask=mask, embed_num=self.embed_num, position_ids=position_ids,
            top_k=self.top_k, top_k_re_norm=self.top_k_re_norm,
            mass_aware=self.mass_aware, eps=self.ram_attn_eps,
            t_dim=t_dim, inplace=inplace, impl_backend='cuda',
        )

        if self.o_norm is not None:
            outputs = self.o_norm(outputs)

        outputs = einops.rearrange(outputs, '... h c -> ... (h c)')

        if self.o_gate is not None:
            outputs = self.o_gate(hidden_states) * outputs

        return self.o_proj(outputs), RAMState(slot, mass)

    def extra_repr(self) -> str:
        return (
            f"{self.n_heads=}, U={self.addr_order}, P={self.addr_radix}, slots={self.addr_radix ** self.addr_order}, C={self.value_chs}, "
            f"{self.addr_temperature=:.1f}, {self.gamma=:.1f}, {self.gamma_mode=}, "
            f"{self.embed_num=}, {self.top_k=}, {self.top_k_re_norm=}"
        ).replace("self.", '')
