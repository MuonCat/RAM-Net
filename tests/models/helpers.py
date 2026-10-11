"""Shared RAMNet configuration, inputs, cache helpers, and assertions."""

import importlib.util
import unittest

import torch

from tests.helpers import assert_close_with_stats, require_cuda_ops

MODEL_CFG = {
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_hidden_layers": 2,

    "addr_order": 5,
    "addr_radix": 4,
    "value_chs": 64,

    "gamma": 2.197,
    "gamma_mode": 'dynamic',

    "embed_num": [4, 2],

    "addr_temperature": 2.2,
    "addr_top_k": 8,

    "output_gate": 'sigmoid',

    "fuse_norm": False,
    "hidden_act": "SwiGLU",
    "vocab_size": 32_000,

    "pad_token_id": 0,

    "dtype": "float32",
}
TOLERANCES_CFG = {"atol": 5e-4, "rtol": 2e-3, "max_err_rate": 0.002}
PREFIX_PAD_WIDTH = 2
POSITION_PAD_WIDTH = 3
MIXED_PREFIX_PAD_WIDTH = 2
MIXED_INSERT_PAD_WIDTH = 2
MIXED_SUFFIX_PAD_WIDTH = 2
PADDING_CFG = {
    "prefix": {"prefix": PREFIX_PAD_WIDTH, "insert": 0, "suffix": 0},
    "position": {"prefix": POSITION_PAD_WIDTH, "insert": 0, "suffix": 0},
    "mixed": {
        "prefix": MIXED_PREFIX_PAD_WIDTH,
        "insert": MIXED_INSERT_PAD_WIDTH,
        "suffix": MIXED_SUFFIX_PAD_WIDTH,
    },
}


def load_model_classes():
    if importlib.util.find_spec("transformers") is None:
        raise unittest.SkipTest("transformers is not available.")
    try:
        fla_modules = importlib.util.find_spec("fla.modules")
    except ModuleNotFoundError:
        fla_modules = None
    if fla_modules is None:
        raise unittest.SkipTest("fla.modules is not available.")
    require_cuda_ops()
    from ramnet import RAMCache, RAMNetConfig, RAMNetForCausalLM
    return RAMCache, RAMNetConfig, RAMNetForCausalLM


def make_model(**config_overrides):
    _, RAMNetConfig, RAMNetForCausalLM = load_model_classes()
    config = RAMNetConfig(**(MODEL_CFG | config_overrides))
    return config, RAMNetForCausalLM(config).cuda()


def make_inputs(config, *, batch: int = 2, seq_len: int = 32, padding: str = "prefix") -> tuple[torch.Tensor, torch.Tensor]:
    padding_cfg = PADDING_CFG[padding]
    prefix_width = padding_cfg["prefix"]
    insert_width = padding_cfg["insert"]
    suffix_width = padding_cfg["suffix"]
    if prefix_width + insert_width + suffix_width >= seq_len:
        raise ValueError(f"Padding exceeds sequence length: {padding_cfg=}, {seq_len=}")

    input_ids = torch.randint(low=3, high=config.vocab_size, size=(batch, seq_len), device="cuda")
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool, device="cuda")

    if prefix_width:
        attention_mask[:, :prefix_width] = False
    if suffix_width:
        attention_mask[:, -suffix_width:] = False
    if insert_width:
        insert_lo = max(prefix_width, seq_len // 3)
        insert_hi = min(seq_len - suffix_width - insert_width, (2 * seq_len) // 3 - insert_width)
        insert_starts = torch.randint(low=insert_lo, high=insert_hi + 1, size=(batch,), device="cuda")
        for batch_idx, start in enumerate(insert_starts.tolist()):
            attention_mask[batch_idx, start:start + insert_width] = False

    input_ids = input_ids.masked_fill(~attention_mask, config.pad_token_id)

    return input_ids, attention_mask


def make_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(~attention_mask, 0)


def compact_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_idx: int, *, end: int | None = None) -> torch.Tensor:
    input_slice = input_ids[batch_idx, :end]
    mask_slice = attention_mask[batch_idx, :end]
    return input_slice[mask_slice].unsqueeze(0)


def make_cache():
    RAMCache, _, _ = load_model_classes()
    return RAMCache()


def model_logits(model, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
    return model(input_ids, **kwargs).logits


def assert_logits_close(name: str, val: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    if mask is not None:
        valid = mask.to(device=val.device, dtype=torch.bool)
        val = val[valid]
        ref = ref[valid]
    assert_close_with_stats(name, [("logits", val, ref)], **TOLERANCES_CFG)


def assert_cache_length(name: str, cache, config, expected: int) -> None:
    lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(config.num_hidden_layers)]
    if any(length != expected for length in lengths):
        raise AssertionError(f"{name} cache length mismatch: {lengths=}, {expected=}")
    print(f"{name}: cache_lengths={lengths}, expected={expected}, ok=True")


def assert_tokens_equal(name: str, val: torch.Tensor, ref: torch.Tensor) -> None:
    print(f"{name:=^80}")
    print(f"    val={tuple(val.shape)}, ref={tuple(ref.shape)}, equal={torch.equal(val, ref)}")
    print("=" * 80)
    if not torch.equal(val, ref):
        raise AssertionError(f"{name} failed token consistency.")


def assert_all_parameters_have_grad(name: str, model: torch.nn.Module) -> None:
    missing_grads = [param_name for param_name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    if missing_grads:
        raise AssertionError(f"{name} parameters without gradients: {missing_grads}")
