from typing import Any, Literal

import torch
from transformers.configuration_utils import PretrainedConfig


class RAMNetConfig(PretrainedConfig):
    """
    Configuration for RAM-Net Attention from
    [RAM-Net: Expressive Linear Attention with Selectively Addressable Memory]
    (https://arxiv.org/abs/2602.11958).

    Args:
        hidden_size (int): Dimensionality of the input and output layers.
        num_attention_heads (int): Number of attention heads.
        num_hidden_layers (int): Number of hidden layers in the model.

        addr_order (int): Order U of the address space (U-order product softmax).
        addr_radix (int): Address radix P (paper d_p): choices per digit, giving P^U memory slots.
        value_chs (int | None): Dimensionality of value channels (d_v), total state size = d_p^U * d_v. None defaults to hidden_size // num_attention_heads.

        gamma (float): Prior forget-rate control. Negative values favor retention and positive values favor forgetting.
        gamma_mode (Literal['fixed', 'learnable', 'dynamic']): Mode for gamma: 'fixed' (non-trainable constant), 'learnable'  (per-head learnable parameter), 'dynamic' (input-conditioned via linear map).

        init_mass (float, optional): Persistence strength of the initial zero state. None defaults to 1 / (addr_radix ** addr_order).

        embed_num (list[int] | tuple[int, ...] | int): Number of heads applying Cyclic Address Positional Embedding (CAPE), optionally per layer. 0 disables CAPE; explicit values must not exceed num_attention_heads.
        addr_temperature (float | None): Temperature for address localization; smaller values sharpen memory access. None defaults to sqrt(addr_order).
        addr_top_k (tuple[int, int] | list[int] | int | None): Limits sparsity of the address vector. If set, retains only the top-k weights. A two-item tuple/list applies separately to (write_k, read_k); int applies to both.
        addr_top_k_re_norm (bool | tuple[bool, bool] | list[bool]): Whether to re-normalize the address vector after top-k filtering, forcing the top-k entries to sum to 1.0. A bool applies to both write and read; a two-item tuple/list specifies separate (write, read) bools.

        mass_aware (bool): Weight reads by updated memory mass with epsilon-stabilized normalization.
        output_gate (Literal['relu', 'sigmoid', 'silu', 'swish', 'gelu'] | None): Element-wise output gate function name. Use None to disable output gating.
        output_norm (bool): Whether to enable head-wise output RMS-norm.

        addr_proj_bias (bool): Whether to enable bias in address projection layers (bias = address prior).
        addr_prior_range (float): Initializes address projection bias with a prior to improve convergence for large memory sizes.
        gate_initializer_bias (float): Initial bias for the optional output gate projection.
        value_proj_bias (bool): Whether to enable bias in the value projection layer.
        address_scaling (bool): Whether to re-parameterize address projection with learnable scaling (w = exp(alpha) * w' for training).

        ram_attn_eps (float): Small value to avoid division by zero.

        hidden_ratio (int): Expansion ratio for the MLP intermediate dimension.
        intermediate_size (int, optional): Dimensionality of the MLP intermediate layer. Overrides hidden_ratio.
        hidden_act (str): 'fuse_SwiGLU' or 'SwiGLU' selects a gated MLP; 'GELU', 'SiLU', or 'Swish' selects a standard MLP.

        fuse_norm (bool): Whether to use fused normalization kernels.
        norm_eps (float): Epsilon used for normalization layers.

        vocab_size (int): Vocabulary size of the model.
        pad_token_id (int): ID of the padding token.
        bos_token_id (int): ID of the BOS token.
        eos_token_id (int): ID of the EOS token.
        tie_word_embeddings (bool): Whether to tie word embeddings.

        use_cache (bool): Whether to return recurrent state by default.
        fuse_cross_entropy (bool): Whether to use fused cross-entropy loss.
        fuse_linear_cross_entropy (bool): Whether to fuse the linear head and cross-entropy loss.
        initializer_range (float): Standard deviation for weight initialization.
        use_l2warp (bool): Whether to use L2-Warp normalization logic.
        dtype (str | torch.dtype): Model dtype passed to the Hugging Face config (for example, "bfloat16").
    """

    model_type = 'ramnet'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
            self,
            # === Model Architecture ===
            hidden_size: int = 2048,
            num_attention_heads: int = 32,
            num_hidden_layers: int = 25,
            # === RAM Structure ===
            addr_order: int = 5,
            addr_radix: int = 4,
            value_chs: int | None = None,
            # === Forgetting Modulation ===
            gamma: float = 0.0,
            gamma_mode: Literal['fixed', 'learnable', 'dynamic'] = 'dynamic',
            # === RAM initial mass ===
            init_mass: float | None = None,
            # === Positional Encoding (CAPE) ===
            embed_num: list[int] | tuple[int, ...] | int = (
                32, 32, 32, 32,
                16, 16, 16, 16,
                8, 8, 8, 8,
                4, 4, 4, 4,
                2, 2, 2, 2, 2, 2, 2, 2, 2,
            ),
            # === Address Decoder ===
            addr_temperature: float | None = None,
            addr_top_k: tuple[int, int] | list[int] | int | None = 8,
            addr_top_k_re_norm: bool | tuple[bool, bool] | list[bool] = False,
            # === RAM Readout ===
            mass_aware: bool = False,
            output_gate: Literal['relu', 'sigmoid', 'silu', 'swish', 'gelu'] | None = "sigmoid",
            output_norm: bool = True,
            # === RAM Projection ===
            addr_proj_bias: bool = True,
            addr_prior_range: float = 1.5,
            gate_initializer_bias: float = 3.,
            value_proj_bias: bool = False,
            address_scaling: bool = True,
            # === RAM Numerical Stability ===
            ram_attn_eps: float = 1e-12,
            # === Feed Forward Net ===
            hidden_ratio: int = 4,
            intermediate_size: int | None = None,
            hidden_act: Literal['fuse_SwiGLU', 'SwiGLU', 'GELU', 'SiLU', 'Swish'] = 'fuse_SwiGLU',
            # === Normalization ===
            fuse_norm: bool = True,
            norm_eps: float = 1e-6,
            # === Tokenization ===
            vocab_size: int = 32_000,
            pad_token_id: int | None = None,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            tie_word_embeddings: bool = False,
            # === Optimization & Init ===
            use_cache: bool = True,
            fuse_cross_entropy: bool = True,
            fuse_linear_cross_entropy: bool = False,
            initializer_range: float = 0.02,
            use_l2warp: bool = False,
            dtype: str | torch.dtype = "bfloat16",

            **kwargs: Any,
    ) -> None:
        if isinstance(embed_num, (list, tuple)) and len(embed_num) > num_hidden_layers:
            raise ValueError('embed_num has more entries than num_hidden_layers.')

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        self.addr_order = addr_order
        self.addr_radix = addr_radix
        self.value_chs = value_chs

        self.gamma = gamma
        self.gamma_mode: Literal['fixed', 'learnable', 'dynamic'] = gamma_mode

        self.embed_num = embed_num
        self.init_mass = init_mass

        self.addr_temperature = addr_temperature
        self.addr_top_k = addr_top_k
        self.addr_top_k_re_norm = addr_top_k_re_norm
        self.mass_aware = mass_aware

        self.output_gate = output_gate
        self.output_norm = output_norm
        self.addr_proj_bias = addr_proj_bias
        self.addr_prior_range = addr_prior_range
        self.gate_initializer_bias = gate_initializer_bias
        self.value_proj_bias = value_proj_bias
        self.address_scaling = address_scaling
        self.ram_attn_eps = ram_attn_eps

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.fuse_norm = fuse_norm
        self.norm_eps = norm_eps

        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            **kwargs,
        )

    def get_embed_num(self, layer_idx: int) -> int:
        """Return the CAPE head count; unspecified layers use content-only heads."""
        if isinstance(self.embed_num, (list, tuple)):
            return self.embed_num[layer_idx] if layer_idx < len(self.embed_num) else 0
        return self.embed_num
