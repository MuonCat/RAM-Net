<div align="center">

<img src="https://raw.githubusercontent.com/MuonCat/RAM-Net/main/assets/logo.png" alt="RAM-Net" width="420">

# ramnet

[![arXiv](https://img.shields.io/badge/arXiv-2602.11958-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.11958)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://muoncat.github.io/ramnet_web/)
[![Weights](https://img.shields.io/badge/Weights-Hugging%20Face-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/kaichengx/RAM-Net-Models/tree/main)
[![GitHub](https://img.shields.io/badge/GitHub-RAM--Net-181717?logo=github&logoColor=white)](https://github.com/MuonCat/RAM-Net)
[![Models](https://img.shields.io/badge/Models-ramnet-82afe6?logo=pytorch&logoColor=white)](https://github.com/MuonCat/RAM-Net/blob/main/packages/ramnet/README.md)
[![Operators](https://img.shields.io/badge/Operators-ramnet--ops-76B900?logo=nvidia&logoColor=white)](https://github.com/MuonCat/RAM-Net/blob/main/packages/ramnet-ops/README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MuonCat/RAM-Net/blob/main/LICENSE)

</div>

RAM-Net attention layers and Hugging Face compatible language models.
This pure Python package uses [ramnet-ops](https://github.com/MuonCat/RAM-Net/blob/main/packages/ramnet-ops/README.md) for CUDA execution.
See the [repository overview](https://github.com/MuonCat/RAM-Net#overview) for the method and experimental results.

## Structure

```text
.
`-- src/
    `-- ramnet/
        |-- __init__.py             # Public exports
        |-- layers/
        |   |-- ram_attn_block.py   # Attention layer and explicit state API
        |   `-- cache.py            # Recurrent state and Hugging Face cache integration
        `-- models/
            |-- __init__.py         # Hugging Face Auto class registration
            |-- config_ramnet.py    # RAMNetConfig
            `-- modeling_ramnet.py  # Blocks, backbone, causal LM, and generation
```

Public exports: `RAMNetConfig`, `RAMNetModel`, `RAMNetForCausalLM`, `RAMNetAttention`, `RAMState`, and `RAMCache`.
Importing `ramnet` registers its Hugging Face Auto classes.

## Installation

Requires:

```text
Python >= 3.10

ramnet
|-- torch >= 2.5
|-- einops
|-- ramnet-ops >= 0.1.0b0
|   `-- numpy
|-- transformers >= 4.57.3
`-- flash-linear-attention >= 0.3.2
```

The attention layer uses the CUDA backend, so a compatible NVIDIA GPU and CUDA-enabled `ramnet-ops` build are **required**.

### 1. Install dependencies

First install PyTorch following the [official installation guide](https://pytorch.org/get-started/locally/).
Choose PyTorch ≥ 2.10 with CUDA 13.0 for the default ops wheels, then install the remaining dependencies from TestPyPI / PyPI:

```bash
python -m pip install einops numpy "transformers>=4.57.3" "flash-linear-attention>=0.3.2"
```

### 2. Install ramnet-ops

Select a CUDA build matching your PyTorch version, GPU, and platform using the [ops installation guide](../ramnet-ops/README.md#installation).
For environments matching the [default stable ABI wheel requirements](../ramnet-ops/README.md#2-install-ramnet-ops), install directly from TestPyPI / PyPI:

```bash
python -m pip install -i https://test.pypi.org/simple/ ramnet-ops
```

### 3. Install ramnet

The model package is pure Python and needs no compiler.
Install it from TestPyPI / PyPI:

```bash
python -m pip install -i https://test.pypi.org/simple/ ramnet
```

#### From source

With the dependencies above installed, choose one source.

**Download with pip** — install the published source archive from TestPyPI / PyPI:

```bash
python -m pip install -i https://test.pypi.org/simple/ ramnet --no-binary=ramnet
```

**Clone from GitHub and install locally**:

```bash
git clone https://github.com/MuonCat/RAM-Net.git
cd RAM-Net
python -m pip install ./packages/ramnet
```

If you also need to build the CUDA operators, follow [ops source installation](../ramnet-ops/README.md#from-source).

## Quick start

`ramnet` provides the Hugging Face model interface, including Auto classes, language-modeling loss, and cached generation with `generate`.
The table maps paper notation to model configuration parameters.

| Paper quantity / symbol | Model parameter |
| :---: | :---: |
| Address order $U$ (digits per address) | `addr_order` |
| Address radix $d_p$ (values per digit) | `addr_radix` |
| Value dimension $d_v$ | `value_chs` |
| Slots per head $d_p^U$ | `addr_radix ** addr_order` (derived) |
| Selected slots $K$ | `addr_top_k` |

### Training and generation

This example uses a small model with random weights, matching [model_smoke.py](../../examples/model_smoke.py):

```python
import torch
from transformers import AutoModelForCausalLM
from ramnet import RAMNetConfig

config = RAMNetConfig(
    hidden_size=128, num_attention_heads=4, num_hidden_layers=2,
    addr_order=3, addr_radix=4,
    embed_num=2, addr_top_k=8,
    output_norm=True,
    hidden_act="SwiGLU",
    vocab_size=1024, pad_token_id=0,
    dtype="float32",
)
model = AutoModelForCausalLM.from_config(config).cuda()
input_ids = torch.randint(3, config.vocab_size, (2, 32), device="cuda")
attention_mask = torch.ones_like(input_ids)

model.train()
output = model(
    input_ids=input_ids, attention_mask=attention_mask,
    labels=input_ids, use_cache=False,
)
output.loss.backward()
print(output.logits.shape)  # (2, 32, 1024)

model.eval()
with torch.inference_mode():
    tokens = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=8, do_sample=False, use_cache=True,
    )
print(tokens.shape)
```

Run the same example from the repository root with `python examples/model_smoke.py`.

`embed_num=2` enables **Cyclic Address Positional Embedding (CAPE)** on the first two heads of each layer, shifting read and write slot indices by token position.
The other heads address by content alone; `0` disables CAPE and `num_attention_heads` enables it for all heads.
An integer applies to every layer; a list or tuple gives per-layer counts, with unspecified layers using `0`.
Set `embed_num` explicitly when changing the model depth or head count, since the default schedule targets 25 layers and 32 heads.

### Save and load

Use the standard Hugging Face checkpoint methods after training:

```python
model.save_pretrained("ramnet-checkpoint")
model = AutoModelForCausalLM.from_pretrained("ramnet-checkpoint").cuda().eval()
```

Import `ramnet` before using Auto classes to load a checkpoint in a new process.

## Standalone attention layer

```python
import torch
from ramnet import RAMNetAttention

attn = RAMNetAttention(
    hidden_size=128, n_heads=4, addr_order=3, addr_radix=4,
    embed_num=2, top_k=8, output_norm=True,
)
attn._init_weights()
attn = attn.cuda()
x = torch.randn(2, 32, 128, device="cuda")
slot = torch.zeros(2, 4, 4 ** 3, 32, device="cuda")
mass = torch.full((2, 4, 4 ** 3), 1 / (4 ** 3), device="cuda")
y, state = attn.forward_with_state(x, init_ram=(slot, mass))
slot, mass = state.slot, state.mass
print(y.shape)  # (2, 32, 128)
print(slot.shape)  # (2, 4, 64, 32)
print(mass.shape)  # (2, 4, 64)
```

Standalone layers need the explicit initialization above; full models apply it automatically.
The standard attention `forward` accepts `(B, T, D)` inputs and returns `(output, None, past_key_values)`, following the FLA layer convention.
Set `layer_idx` when constructing attention that uses `RAMCache`.

Both attention and complete model blocks provide `forward_with_state`:

- `attn.forward_with_state(..., init_ram=state)` runs the attention layer.
- `model.model.layers[i].forward_with_state(..., init_state=state)` runs the full block, including normalization, MLP, and residual connections.

Both accept a `(slot, mass)` pair or `RAMState(slot, mass)` and return `(output, RAMState)`; these APIs also expose `t_dim` and `inplace`.
The default `inplace=False` preserves input states and supports differentiation through consecutive calls.
Explicit-state callers supply continuation `position_ids` for CAPE.

## Configuration

Blocks use pre-norm RMSNorm; `fuse_norm` selects its fused implementation.
`hidden_act="fuse_SwiGLU"` (default) or `"SwiGLU"` selects FLA's `GatedMLP`, with or without fusion.
`"GELU"`, `"SiLU"`, and `"Swish"` select a standard two-layer MLP; `"SiLU"` and `"Swish"` use the same activation.
Models support `input_ids` or `inputs_embeds`, `return_dict`, hidden-state outputs, and `logits_to_keep`.

- `addr_top_k` sets the number of selected slots for both writing and reading; a `(write_k, read_k)` pair specifies separate counts.
- `addr_top_k_re_norm` explicitly enables renormalization after Top-K: a bool for both sides, or a two-item tuple/list of `(write, read)` bools; defaults to `False`.
- `output_norm` enables head-wise RMSNorm; `mass_aware` enables mass-weighted reads.

See [RAMNetConfig](src/ramnet/models/config_ramnet.py) for the full parameter list and defaults.

## Cache and sequence handling

`config.use_cache` controls the evaluation default; training disables caching unless explicitly requested.

- `use_cache=True` carries recurrent state during generation.
  The attention layer automatically uses inplace updates for initialized, gradient-free inference caches.
- `RAMCache()` allocates states on first use and stores per-layer `recurrent_state` entries.
  It supports beam reordering, batch selection/repetition, and reset.
  Logical CAPE positions are tracked per sample, so a full accumulated mask or a query-only mask can be used after padded prefill.
  Explicit `position_ids` override automatic positions.
- `use_cache=False` reads a supplied cache without modifying or returning it.
  Gradient checkpointing requires an uncached sequence and disables cache updates during training.
- Cache rollback, speculative decoding, static/offloaded KV-cache implementations, and full-graph compiled generation are not supported.
- Packed `cu_seqlens` inputs and attention-matrix outputs (`output_attentions=True`) are not supported.

## Development

See [RELEASING](RELEASING.md) for tests and publishing.

MIT License. See [LICENSE](LICENSE).
