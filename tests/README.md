<!--
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2026-10-11 12:27:54
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2026-10-11 22:21:47
 * @FilePath: \RAM_attention\tests\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Tests

Run commands from the repository root after installing [ramnet-ops](../packages/ramnet-ops/README.md#installation) and, for model tests, [ramnet](../packages/ramnet/README.md#installation).
Tests use the installed packages; reinstall after code changes unless using an editable installation.
CUDA tests require a compatible GPU and the compiled ops extension; all model tests require CUDA.
Inductor tests also require a working Triton/compiler toolchain (`triton-windows` on Windows).

## Run

```bash
# Both packages, FP32, seeds 0 1 2.
python -m tests

# Select a package.
python -m tests --part ops
python -m tests --part models

# Add optional operator precision checks.
python -m tests --part ops --fp16 --bf16

# Select a group or test module, using one seed.
python -m tests --part models --group training --seeds 0
python -m tests --test tests.ops.test_cuda_compile --seeds 0

# CPU-only API and PyTorch reference checks.
python -m tests --part ops --device cpu
```

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--part` | `all` | Select `ops`, `models`, or both |
| `--group` | All groups | Select a group from the table below |
| `--test` | All tests | Select a dotted module/class prefix or a full test method ID |
| `--device` | `cuda` | `cuda` includes CPU and CUDA tests; `cpu` excludes CUDA tests |
| `--seeds` | `0 1 2` | Space-separated seeds; repeat the selection for each seed |
| `--fp16` | Disabled | Add CUDA FP16 operator cases |
| `--bf16` | Disabled | Add CUDA BF16 operator cases; requires BF16 support |
| `-h`, `--help` | — | Show command-line help |

Filters intersect; `--test` does not bypass other filters.
Precision flags apply only to CUDA operator tests and can be enabled independently; model tests remain FP32.
The runner resets the Torch seed before each test and records environment details, test IDs, and results.
Empty selections, missing required CUDA support, failures, errors, and unexpected skips return a nonzero exit code.

## Groups

The hierarchy is `part → group → module → class → test`.
A test is one `test_*` method, including all its `subTest` cases.
Groups are logical categories defined in [TEST_GROUPS](__main__.py), not directories.

| Group | `ops` | `models` |
| --- | --- | --- |
| `contracts` | API, registration, autograd, compile | Configuration, initialization, HF interfaces, cache operations |
| `behavior` | Masks, positions, chunking, autoregressive/parallel, inplace | Masks, positions, chunked execution, cache continuity |
| `numerics` | FP32 CUDA vs PyTorch; optional FP16/BF16 vs CUDA FP32 | — |
| `training` | — | Forward/backward, gradient checkpointing, compiled training |
| `generation` | — | Greedy/beam generation, generation with compiled forward |
