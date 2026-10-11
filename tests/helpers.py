import unittest

import torch


def require_cuda_ops():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is not available.")
    try:
        from ramnet_ops.impl.cuda_impl import _cuda_ram_attn
    except Exception as exc:
        raise unittest.SkipTest(f"RAM attention CUDA backend is unavailable: {exc}")
    return _cuda_ram_attn


def select_valid_tokens(val: torch.Tensor, valid_mask: torch.Tensor, t_dim: int) -> torch.Tensor:
    valid_mask = valid_mask.to(device=val.device, dtype=torch.bool)
    t_dim = t_dim % val.ndim
    assert val.size(0) == valid_mask.size(0), f"Batch mismatch: {val.shape=}, {valid_mask.shape=}"
    assert val.size(t_dim) == valid_mask.size(1), f"Time mismatch: {val.shape=}, {valid_mask.shape=}, {t_dim=}"

    view_shape = [1] * val.ndim
    view_shape[0] = valid_mask.size(0)
    view_shape[t_dim] = valid_mask.size(1)
    return val[valid_mask.reshape(*view_shape).expand_as(val)]


def assert_close_with_stats(
        title: str,
        pairs: list[tuple[str, torch.Tensor, torch.Tensor]],
        *,
        atol: float,
        rtol: float,
        max_err_rate: float | None = 0.002,
        max_q99: float | None = 1.0,
) -> None:
    digits = 4
    header = "              median   mean      std    q95%    q99%   err >1        max    bad / total (limit)"
    width = len(header)
    lines = [
        f"{title:=^{width}}",
        header,
    ]
    failed: list[tuple[str, str]] = []

    for name, val, ref in pairs:
        if val.shape != ref.shape:
            failed.append((name, f"Shape mismatch: {val.shape} != {ref.shape}."))
            lines.append(f"{name:>12}: shape mismatch  <-- [{len(failed)}]")
            continue
        if val.numel() == 0:
            lines.append(f"{name:>12}: empty")
            continue
        val = val.detach().double()
        ref = ref.detach().double()
        if not (torch.isfinite(val).all().item() and torch.isfinite(ref).all().item()):
            failed.append((name, "Non-finite values in result or reference."))
            lines.append(f"{name:>12}: non-finite values  <-- [{len(failed)}]")
            continue
        err = ((val - ref).abs() / (atol + rtol * ref.abs())).flatten()
        q95 = err.quantile(q=0.95).item()
        q99 = err.quantile(q=0.99).item()
        # Equality is within tolerance, as in torch.allclose.
        err_count = (err > 1).sum().item()
        err_rate = err_count / err.numel()
        max_err_count = None
        if max_err_rate is not None:
            max_err_count = int(max_err_rate * err.numel())
        row = (
            f"{name:>12}: "
            f"{err.median().item():{digits + 2}.{digits}f}, "
            f"{err.mean().item():{digits + 2}.{digits}f}, "
            f"{err.std(unbiased=False).item():{digits + 3}.{digits}f}, "
            f"{q95:{digits + 2}.{digits}f}, "
            f"{q99:{digits + 2}.{digits}f}, "
            f"{err_rate:{digits + 3}.{digits}%}, "
            f"{err.amax().item():{digits + 5}.{digits}f}    "
            f"{err_count} / {err.numel()} ({max_err_count if max_err_count is not None else '-'})"
        )
        allclose_ok = torch.allclose(val, ref, atol=atol, rtol=rtol)
        ok = allclose_ok
        failed_checks: list[str] = []
        if max_err_rate is not None:
            checks = [
                (err_count <= max_err_count,
                 f"err >1 count {err_count}/{err.numel()} ({err_rate:.4%}) exceeds {max_err_count} "
                 f"(rate budget {max_err_rate:.4%}, rounded down)"),
                (q95 <= 1, f"q95 {q95:.4f} is not <= 1.0000"),
            ]
            if max_q99 is not None:
                checks.append((q99 <= max_q99, f"q99 {q99:.4f} is not <= {max_q99:.4f}"))
            failed_checks = [reason for passed, reason in checks if not passed]
            ok = ok or not failed_checks
        if not ok:
            reason = f"torch.allclose failed (atol={atol}, rtol={rtol})"
            if failed_checks:
                reason += f"; statistical tolerance failed: {', '.join(failed_checks)}"
            failed.append((name, reason))
            row += f"  <-- [{len(failed)}]"
        lines.append(row)

    lines.append("=" * width)
    if failed:
        lines.append("Errors:")
        lines.extend(f"[{index}] {name}: {reason}" for index, (name, reason) in enumerate(failed, start=1))
    report = "\n".join(lines)
    print(report)

    if failed:
        raise AssertionError(f"{title} failed for {', '.join(name for name, _ in failed)}\n{report}")
