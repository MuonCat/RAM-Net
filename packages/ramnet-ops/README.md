<div align="center">

<img src="https://raw.githubusercontent.com/MuonCat/RAM-Net/main/assets/logo.png" alt="RAM-Net" width="420">

# ramnet-ops

[![arXiv](https://img.shields.io/badge/arXiv-2602.11958-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.11958)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://muoncat.github.io/ramnet_web/)
[![Weights](https://img.shields.io/badge/Weights-Hugging%20Face-FFD21E?logo=huggingface&logoColor=white)](https://huggingface.co/kaichengx/RAM-Net-Models/tree/main)
[![GitHub](https://img.shields.io/badge/GitHub-RAM--Net-181717?logo=github&logoColor=white)](https://github.com/MuonCat/RAM-Net)
[![Models](https://img.shields.io/badge/Models-ramnet-82afe6?logo=pytorch&logoColor=white)](https://github.com/MuonCat/RAM-Net/blob/main/packages/ramnet/README.md)
[![Operators](https://img.shields.io/badge/Operators-ramnet--ops-76B900?logo=nvidia&logoColor=white)](https://github.com/MuonCat/RAM-Net/blob/main/packages/ramnet-ops/README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MuonCat/RAM-Net/blob/main/LICENSE)

</div>

The attention computation core of RAM-Net: sparse address decoding, memory updates, readout, and autograd.
Import as `ramnet_ops` for CUDA kernels and a PyTorch reference implementation.

## Structure

```text
.
|-- src/
|   `-- ramnet_ops/
|       |-- ram_attn.py           # Public API, input normalization, backend selection
|       `-- impl/
|           |-- torch_impl.py     # PyTorch reference
|           |-- cuda_impl.py      # CUDA wrappers and segment layout
|           |-- _cuda_autograd.py # FakeTensor and autograd registrations
|           `-- _cuda_config.py   # Address and index-capacity checks
`-- csrc/
    |-- ram_attn_bind.cpp         # Torch schemas, CUDA registration, extension entry point
    |-- ram_attn_ops.cpp          # Native wrappers, output allocation, kernel dispatch
    |-- ram_attn_*.cu             # Address, segment, and autoregressive kernels
    `-- torch_compat*             # Ordinary/stable Torch ABI interfaces
```

## Installation

Requires:

```text
Python >= 3.10

ramnet-ops
|-- torch >= 2.5
|-- einops
`-- numpy
```

### 1. Install dependencies

If PyTorch is not already installed, we **recommend PyTorch ≥ 2.10 with CUDA 13.0 (`cu130`)** for the default ops wheels below.
See the [official installation guide](https://pytorch.org/get-started/locally/) for installation commands.
Then install the remaining dependencies from TestPyPI / PyPI:

```bash
python -m pip install einops numpy
```

### 2. Install ramnet-ops

The default `ramnet-ops` wheels use both **Python and PyTorch stable ABIs** and target **CUDA 13.0 (`cu130`)**.
If your environment meets **all** of the following requirements, install directly from pip:

- **Python:** CPython ≥ 3.10.
- **PyTorch:** ≥ 2.10 with CUDA 13.0.
- **GPU:** SM 8.9, 9.0, 10.0, or 12.0.
- **Platform:** Windows x64 or Linux x86_64 with glibc ≥ 2.35.

```bash
python -m pip install -i https://test.pypi.org/simple/ ramnet-ops
```

Prebuilt wheels need no local compiler or CUDA Toolkit.
If your environment **does not meet all of the requirements above**, follow [Other Torch/CUDA builds (GitHub Releases)](#other-torchcuda-builds-github-releases) below to select another prebuilt wheel.

#### Other Torch/CUDA builds (GitHub Releases)

GitHub Releases provides additional wheels for older PyTorch versions, other CUDA builds, and GPUs outside the default `cu130` targets.
Open [GitHub Releases](https://github.com/MuonCat/RAM-Net/releases), select the desired package version under `ramnet-ops/v<VERSION>`, and expand **Assets** to find its `.whl` files.
Choose a wheel using [the selection guide below](#choose-a-matching-wheel), then copy its download link into `python -m pip install "<wheel URL>"`.
You must select the Torch/CUDA variant yourself; pip does not infer it from your environment.

With the matching PyTorch build already installed, the commands look like this for package version `0.1.0rc0`:

Windows x64, **Torch 2.8.0 / CUDA 12.8** (ordinary ABI):

```bash
python -m pip install "https://github.com/MuonCat/RAM-Net/releases/download/ramnet-ops%2Fv0.1.0rc0/ramnet_ops-0.1.0rc0-1torch2_8_0cu128-cp310-abi3-win_amd64.whl"
```

Linux x86_64, **Torch ≥ 2.10 / CUDA 12.6** (stable ABI):

```bash
python -m pip install "https://github.com/MuonCat/RAM-Net/releases/download/ramnet-ops%2Fv0.1.0rc0/ramnet_ops-0.1.0rc0-1torchabi2_10cu126-cp310-abi3-manylinux_2_35_x86_64.whl"
```

##### Choose a matching wheel

First, match the **Torch ABI** marker in the filename to your installed PyTorch version:

- **Ordinary ABI (`torch2_6_0` or `torch2_8_0`):** requires exactly **PyTorch 2.6.0** or **2.8.0**, respectively.
- **Stable ABI (`torchabi2_10`):** supports **PyTorch ≥ 2.10** with a matching CUDA/platform combination; the Torch 2.14 SDK used to build these wheels is not required at runtime.

Next, find your **GPU row** and **PyTorch column** in the table below.
Each cell lists the available CUDA builds: choose one matching your installed PyTorch's CUDA version (`torch.version.cuda`), such as `cu126` for CUDA 12.6.

| [GPU CC](https://developer.nvidia.com/cuda/gpus) (representative GPUs) | PyTorch 2.6.0 | PyTorch 2.8.0 | PyTorch ≥ 2.10 |
| --- | --- | --- | --- |
| 7.5 (T4 / RTX 20) | 11.8 | ❌ | ❌ |
| 8.0 (A100) | 11.8 / 12.6 | 12.6 | 12.6 |
| 8.6 (RTX 30) | 11.8 / 12.6 | 12.6 | 12.6 |
| 8.9 (RTX 40 / L40S) | 12.6 | 12.6 / 12.8 | 12.6 / **13.0 ★** |
| 9.0 (H100 / H200) | 12.6 | 12.6 / 12.8 | 12.6 / **13.0 ★** |
| 10.0 (B200 / GB200) | ❌ | 12.8 | **13.0 ★** |
| 12.0 (RTX 50) | ❌ | 12.8 | **13.0 ★** |

★ marks the default TestPyPI / PyPI build; ❌ means no prebuilt wheel is provided for that combination.

This table describes the architectures included in our wheels, not every GPU supported by each CUDA Toolkit.

Finally, match the **platform** suffix: `win_amd64` for Windows x64 or `manylinux_2_35_x86_64` for Linux x86_64 with glibc ≥ 2.35.
All wheels use `cp310-abi3` for GIL-enabled **CPython ≥ 3.10**.
Check the driver requirements below before installing; if no wheel matches your environment, see [From source](#from-source).

##### NVIDIA drivers

The table lists driver baselines from NVIDIA's CUDA GA release notes; use these or newer drivers that support your GPU.

| CUDA | Windows driver | Linux driver |
| --- | --- | --- |
| [11.8](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html) | ≥ 522.06 | ≥ 520.61.05 |
| [12.6](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html) | ≥ 560.76 | ≥ 560.28.03 |
| [12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html) | ≥ 570.65 | ≥ 570.26 |
| **[13.0 ★](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)** | **R580+** | **≥ 580.65.06** |

CUDA 13.0 no longer bundles a Windows driver; NVIDIA requires the R580 branch or newer.
CUDA 11.x/12.x may run on older drivers through [minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), subject to feature and PTX restrictions; newer GPUs may require newer drivers.

#### From source

Source installation supports both CUDA extension builds and PyTorch-only installations.

##### With CUDA extension

Use this option when no prebuilt wheel matches your environment or when modifying the CUDA operators.
Prepare the build environment first:

- **PyTorch:** install the intended CUDA build (Torch ≥ 2.5).
- **CUDA Toolkit:** install a version compatible with that PyTorch build, including `nvcc`.
- **C++ compiler:** use a compiler supported by the Toolkit; on Windows, run the commands in an **x64 Native Tools Command Prompt for VS 2022**.

By default (`RAMNET_TORCH_STABLE_ABI=auto`), Torch 2.5–2.13 builds use the ordinary ABI and require that exact Torch version at runtime; Torch ≥ 2.14 builds use stable ABI targeting Torch ≥ 2.10.

Install the build tools in the same Python environment as PyTorch:

```bash
python -m pip install "setuptools>=77.0.3" wheel ninja
```

Then choose one of the following sources.

**Download and build with pip** — install the published source archive from TestPyPI / PyPI:

```bash
python -m pip install -i https://test.pypi.org/simple/ ramnet-ops --no-binary=ramnet-ops --no-build-isolation
```

**Clone from GitHub and build locally** — use the repository source, for example when modifying the operators:

```bash
git clone https://github.com/MuonCat/RAM-Net.git
cd RAM-Net
python -m pip install ./packages/ramnet-ops --no-build-isolation
```

The CUDA extension is compiled during installation.

##### Environment variables

Set these before source installation; ABI and architecture settings apply only when building the CUDA extension.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SKIP_CUDA` | `0` | Set to `1` to skip the CUDA extension and install only the Python package and PyTorch reference. |
| `RAMNET_TORCH_STABLE_ABI` | `auto` | Select the ABI from the build SDK; `0` forces ordinary ABI; `1` forces stable ABI and requires Torch ≥ 2.14 to build. |
| `TORCH_CUDA_ARCH_LIST` | GPUs visible during the build | Set the target GPU architectures explicitly, e.g. `8.9 9.0`. |

Set `SKIP_CUDA=1` to install from source without building the native CUDA extension.
PyTorch, `einops`, and `numpy` are still required; a CUDA Toolkit and C++ compiler are not needed.

## Quick start

`random_access_memory_attention` performs the sparse reads and writes of a RAM-Net layer, covering address decoding, CAPE, and the Gated Sparse Update.
It takes address logits, values, and forgetting modulators, and returns the read outputs and the updated slot/mass state.

The table maps paper notation to parameters and tensor dimensions in the example below.
`B`, `T`, and `H` denote batch size, sequence length, and number of heads.

| Paper quantity / symbol | Operator parameter / axis |
| :---: | :---: |
| Address order $U$ (digits per address) | `U` |
| Address radix $d_p$ (values per digit) | `P` |
| Value dimension $d_v$ | `C` |
| Slots per head $d_p^U$ | `A = P ** U` |
| Selected slots $K$ | `top_k` |

This example computes attention outputs and the updated state, then backpropagates through both:

```python
import torch
from ramnet_ops import random_access_memory_attention

device = "cuda"  # Set to "cpu" for the PyTorch reference.
backend = "cuda" if device == "cuda" else "torch"
torch.manual_seed(0)

B, T, H, U, P, C = 2, 32, 4, 3, 4, 32
w_addr = torch.randn(B, T, H, U, P, device=device, requires_grad=True)
r_addr = torch.randn(B, T, H, U, P, device=device, requires_grad=True)
values = torch.randn(B, T, H, C, device=device, requires_grad=True)
gamma = torch.randn(B, T, H, device=device, requires_grad=True)

y, (slot, mass) = random_access_memory_attention(
    w_addr, r_addr, values, gamma,
    embed_num=2, top_k=8,
    impl_backend=backend,
)
loss = y.square().mean() + slot.square().mean() + mass.square().mean()
loss.backward()
print(y.shape)     # (B, T, H, C)
print(slot.shape)  # (B, H, P ** U, C)
print(mass.shape)  # (B, H, P ** U)
```

`embed_num` selects how many heads, starting from head 0, use **Cyclic Address Positional Embedding (CAPE)** to cyclically shift read and write slot indices by token position.
With `H=4` and `embed_num=2`, heads 0 and 1 use CAPE, while heads 2 and 3 address by content alone.
Its range is `0` to `H`: `0` disables CAPE and `H` enables it for all heads.

From the repository root, run [ops_smoke.py](../../examples/ops_smoke.py) with `python examples/ops_smoke.py --device cuda`, or use `--device cpu` for the PyTorch reference.
This example demonstrates the API; it is not an experiment configuration.

### Arguments

- `w_addr`, `r_addr`: write and read address logits of shape `(…, T, …, H, U, P)`. The operator applies a softmax over each digit.
- `values`: values to write, of shape `(…, T, …, H, C)`. The output `y` has the same shape and dtype.
- `gamma`: forgetting modulator of shape `(…, T, …, H)`, computed by the model for each token. Larger values overwrite old slot content faster.
- `init_ram`: the `(slot, mass)` state returned by a previous call, used to continue a sequence. Without it, every slot starts empty.
- `inplace`: if `True`, updates the `init_ram` tensors in place instead of returning new ones. This requires contiguous FP32 state and does not support gradients.
- `mask`: marks valid tokens. Masked tokens are not written to the state, and their outputs should be ignored.
- `embed_num`: number of heads that use CAPE, counted from head 0. These heads shift each token's addresses by its position.
- `top_k`: `K`, given as an int or a `(write_k, read_k)` pair. `None` keeps all slots and is supported only by the PyTorch backend.
- `top_k_re_norm`: explicitly enables renormalization after Top-K. Accepts a bool for both sides, or a two-item tuple/list of `(write, read)` bools; defaults to `False`.
- `t_dim`: index of the time axis. Dimensions before `H` other than the time axis are broadcast as batch dimensions.
- `impl_backend`: `'cuda'` or `'torch'`. By default, CUDA inputs use the CUDA backend when supported and otherwise fall back to PyTorch with a warning.

## CUDA execution

The CUDA backend automatically selects between **autoregressive** and **parallel** execution.

**Autoregressive mode** handles single-token inputs (`T = 1`) when no input requires gradients.
A dedicated kernel decodes the current token's addresses, updates the selected write slots, and reads the selected read slots directly.

**Parallel mode** handles all other inputs, including training and prefill.
It decodes all tokens' addresses in parallel, treating each selected read or write access as an **event**.
Events targeting the same `(batch, head, slot)` are grouped into a **segment**.
The kernels process independent segments and their value channels in parallel while preserving causal dependencies within each segment, then accumulate read contributions into the token outputs.

## Autograd and compile

Operators are registered under `torch.ops.ramnet_ops` with FakeTensor and first-order autograd support.
Compile coverage targets fixed shapes/configuration with `dynamic=False`, tested through `aot_eager`;
it does not establish Inductor or CUDA Graph support.

Segment capacity is `B * H * A`; event capacity is `B * H * T_padded * (write_k + read_k)`.
Masks do not reduce these allocations.
Higher-order gradients are not supported.

## Development

See [RELEASING](RELEASING.md) for tests and publishing.

MIT License. See [LICENSE](LICENSE).
