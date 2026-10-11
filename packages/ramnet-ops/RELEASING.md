# Releasing ramnet-ops

`ramnet` and `ramnet-ops` are built and published independently, with versions read from their own `pyproject.toml`.
Use `ramnet/v<VERSION>` for this package and `ramnet-ops/v<VERSION>` for the operators.

## Tests

Install the candidate wheel and its [dependencies](README.md#installation), then run from the repository root on a CUDA GPU:

```bash
python -m tests --part ops --device cuda
```

- Add `--fp16` and/or `--bf16` for optional low-precision checks; BF16 requires a supported GPU.
- Use `--group contracts`, `--group behavior`, or `--group numerics` to select a group.
- Seeds default to `0 1 2`; use `--seeds 0` for a focused rerun.
- Use `--device cpu` for CPU-only checks; this does not validate the CUDA kernels.

After public API changes, also check model integration with `ramnet` and its dependencies installed:

```bash
python -m tests --part models --device cuda
```

Tests run locally; the build and publish workflows do not run them.
See the [test guide](../../tests/README.md) for selection options, parameter coverage, and acceptance criteria.

## Release checklist

- [ ] Set a new `version` in [pyproject.toml](pyproject.toml).
- [ ] Update version references in [README → Other Torch/CUDA builds (GitHub Releases)](README.md#other-torchcuda-builds-github-releases)
- [ ] Push the release commit to `main` and wait for [ramnet-ops build](../../.github/workflows/ramnet-ops-build.yml) to succeed.
- [ ] Install a wheel from that run's artifacts and run the tests above.
- [ ] Create and push `ramnet-ops/v<VERSION>` at the exact build commit; the tag version must match `pyproject.toml` at that commit.
- [ ] Keep the successful build run ID and test results for publishing.

## Publish

Run [ramnet-ops publish](../../.github/workflows/ramnet-ops-publish.yml) manually with:

- `build_run_id`: the numeric ID from the successful build's `/actions/runs/<ID>` URL.
- `tag`: the full `ramnet-ops/v<VERSION>` tag.
- `target`: `testpypi`, `pypi`, or `release` for GitHub Release only.

Publish to TestPyPI first, verify installation and tests, then publish to PyPI using the same build ID and tag.
The workflow checks build/tag/version agreement and reuses the original artifacts without rebuilding.
Every target creates or verifies the GitHub Release first; an existing matching release is reused without uploading its assets again.
