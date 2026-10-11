# Releasing ramnet

`ramnet` and `ramnet-ops` are built and published independently, with versions read from their own `pyproject.toml`.
Use `ramnet/v<VERSION>` for this package and `ramnet-ops/v<VERSION>` for the operators.

## Tests

Install the candidate wheel and its [dependencies](README.md#installation), including CUDA-enabled `ramnet-ops`, then run from the repository root on a CUDA GPU:

```bash
python -m tests --part models --device cuda
```

- Use `--group contracts`, `--group training`, `--group behavior`, or `--group generation` to select a group.
- Seeds default to `0 1 2`; use `--seeds 0` for a focused rerun.

Tests run locally; the build and publish workflows do not run them.
See the [test guide](../../tests/README.md) for selection options, parameter coverage, and acceptance criteria.

## Release checklist

- [ ] Set a new `version` in [pyproject.toml](pyproject.toml).
- [ ] Push the release commit to `main` and wait for [ramnet build](../../.github/workflows/ramnet-build.yml) to succeed.
- [ ] Install a wheel from that run's artifacts and run the tests above.
- [ ] Create and push `ramnet/v<VERSION>` at the exact build commit; the tag version must match `pyproject.toml` at that commit.
- [ ] Keep the successful build run ID and test results for publishing.

## Publish

Run [ramnet publish](../../.github/workflows/ramnet-publish.yml) manually with:

- `build_run_id`: the numeric ID from the successful build's `/actions/runs/<ID>` URL.
- `tag`: the full `ramnet/v<VERSION>` tag.
- `target`: `testpypi`, `pypi`, or `release` for GitHub Release only.

Publish to TestPyPI first, verify installation and tests, then publish to PyPI using the same build ID and tag.
The workflow checks build/tag/version agreement and reuses the original artifacts without rebuilding.
Every target creates or verifies the GitHub Release first; an existing matching release is reused without uploading its assets again.
