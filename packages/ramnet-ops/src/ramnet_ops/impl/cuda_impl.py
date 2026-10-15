import torch
import einops
import warnings
from dataclasses import dataclass

import ramnet_ops._C  # Load native dispatcher registrations.
from . import _cuda_autograd  # Register Fake implementations and autograd formulas.
from ._cuda_config import ADDRESS_LIMIT, check_int_capacity


def _cuda_topk_autoregressive_(
        w_addr: torch.Tensor,  # (B, H, U, P)
        r_addr: torch.Tensor,  # (B, H, U, P)

        x: torch.Tensor,  # (B, H, C)
        gamma: torch.Tensor,  # (B, H)

        slot_: torch.Tensor,  # r/w (B, H, A, C)
        mass_: torch.Tensor,  # r/w (B, H, A)

        mask: torch.Tensor | None,  # (B)
        embed_num: int,
        position_ids: torch.Tensor,  # (B)

        w_top_k: int,
        r_top_k: int,

        w_re_norm: bool,
        r_re_norm: bool,

        mass_aware: bool,
        eps: float,
) -> torch.Tensor:  # (B, H, C)
    """
    [Inference Only] Top-K Autoregressive Step

    - Mode: Token-by-token generation (T=1) with Top-K selection.
    - Operation: Computes Top-K address on-the-fly, updates states via decay factor, and aggregates output.
    - Parallelism: Specialized warp-level reduction for Top-K sorting and channel updates.
    """
    assert slot_.is_contiguous() and mass_.is_contiguous(), f"Expected contiguous state, got slot={slot_.is_contiguous()}, mass={mass_.is_contiguous()}."

    w_addr = w_addr.contiguous()
    r_addr = r_addr.contiguous()
    x = x.contiguous()
    gamma = gamma.contiguous()
    position_ids = position_ids.contiguous()
    mask = torch.empty(0, dtype=torch.bool, device=x.device) if mask is None else mask.to(device=x.device, dtype=torch.bool, non_blocking=True).contiguous()

    return torch.ops.ramnet_ops.topk_autoregressive_(
        w_addr, r_addr,
        x, gamma,
        slot_, mass_,
        mask, embed_num, position_ids,
        w_top_k, r_top_k,
        w_re_norm, r_re_norm,
        mass_aware, eps,
    )


# noinspection PyClassHasNoInit
@dataclass(slots=True)
class SegmentMeta:
    """Fixed-capacity layout; event_v is populated by the native forward op."""
    seg_idx: torch.Tensor  # (S)
    seg_off: torch.Tensor  # (S)
    seg_len: torch.Tensor  # (S)
    event_i: torch.Tensor  # (B * H * T * (Kw + Kr))
    event_v: torch.Tensor | None = None  # (B * H * T * (Kw + Kr))


def gather_segment(
        addr_idx: torch.Tensor,  # (B, H, Kw + Kr, T)

        mask: torch.Tensor | None,  # (B, T)

        address_size: int,
) -> SegmentMeta:
    """Group events without data-dependent shapes or host synchronization."""
    B, H, K, T = addr_idx.shape
    slot_count = B * H * address_size

    # flatten and change order
    addr_idx = einops.rearrange(addr_idx, 'b h k t -> (b h t k)')

    if mask is not None:
        mask = mask[:, None, :, None].expand(B, H, T, K).reshape(-1)
        addr_idx = torch.where(mask, addr_idx, slot_count)

    # Stable ordering preserves WRITE before READ at the same timestep
    sorted_idx, event_i = torch.sort(addr_idx, stable=True)
    event_i = event_i.to(torch.int32)

    # Future work: use event capacity when A >> T * K to reduce memory and scheduling overhead
    boundaries = torch.searchsorted(sorted_idx, torch.arange(slot_count + 1, device=addr_idx.device, dtype=torch.int32), out_int32=True)
    seg_off = boundaries[:-1]
    seg_len = boundaries[1:] - seg_off
    seg_idx = torch.arange(slot_count, device=addr_idx.device, dtype=torch.int32)

    # Schedule longer segments first (CUDA launch kernel in sequence, so loner segments first to reduce bubble)
    seg_len, perm = torch.sort(seg_len, descending=True)
    return SegmentMeta(seg_idx[perm], seg_off[perm], seg_len, event_i)


def _cuda_topk_parallel(
        addr_idx: torch.Tensor,  # r (B, H, Kw + Kr, T)
        addr_val: torch.Tensor,  # r (B, H, Kw + Kr, T)

        x: torch.Tensor,  # r (B, H, T, C)
        gamma: torch.Tensor,  # r (B, H, T)

        init_slot: torch.Tensor,  # r or r/w (B, H, A, C)
        init_mass: torch.Tensor,  # r or r/w (B, H, A)

        mask: torch.Tensor | None,  # r (B, T)

        w_top_k: int,
        r_top_k: int,

        mass_aware: bool,
        eps: float,

        inplace: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, A, _ = init_slot.shape
    seg_meta = gather_segment(addr_idx, mask, A)
    args = (
        addr_val,
        seg_meta.seg_idx, seg_meta.seg_off, seg_meta.seg_len, seg_meta.event_i,
        x, gamma,
        init_slot, init_mass,
        w_top_k, r_top_k,
        mass_aware, eps,
    )
    if inplace:
        y = torch.ops.ramnet_ops.topk_segment_forward_(*args)
        return y.to(x.dtype), init_slot, init_mass

    # The fourth output is a private, non-differentiable replay cache for autograd.
    y, slot, mass, _ = torch.ops.ramnet_ops.topk_segment_forward(*args)
    return y.to(x.dtype), slot, mass


def _cuda_ram_attn(
        w_addr: torch.Tensor,  # (B, H, U, P, T)
        r_addr: torch.Tensor,  # (B, H, U, P, T)

        xs: torch.Tensor,  # (B, H, T, C)
        gamma: torch.Tensor,  # (B, H, T)

        init_slot: torch.Tensor,  # (B, H, A, C)
        init_mass: torch.Tensor,  # (B, H, A)

        mask: torch.Tensor | None,  # (B, T)
        embed_num: int,
        position_ids: torch.Tensor,  # (B, T)

        top_k: tuple[int, int] | None,
        top_k_re_norm: tuple[bool, bool],

        mass_aware: bool,
        eps: float,

        inplace: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # shapes: (B, H, T, C), (B, H, A, C), (B, H, A)
    """
    [Main Entry] RAM Attention Dispatcher
    Automatically dispatches to the efficient kernel based on input parameters:
    1. Autoregressive Inference: If T=1 and no gradients required, calls `_cuda_topk_autoregressive_`.
    2. Parallel Training/Prefill: Calls `_cuda_topk_parallel` (Sparse Segment Scan).

    Handles data layout rearrangement (e.g., contiguous checks, padding, dimension swapping) required by CUDA kernels.
    """
    assert xs.ndim == 4 and w_addr.ndim == 5 and r_addr.ndim == 5, f"Expected xs (B, H, T, C), w/r (B, H, U, P, T); got {tuple(xs.shape)}, {tuple(w_addr.shape)}, {tuple(r_addr.shape)}."
    if top_k is None:
        raise ValueError("CUDA RAM attention only supports Top-K mode. Use top_k or impl_backend='torch' for dense mode.")
    assert isinstance(top_k, tuple) and len(top_k) == 2 and all(isinstance(k, int) and k > 0 for k in top_k), f"Expected top_k=(positive write_k, positive read_k), got {top_k!r}."
    assert isinstance(top_k_re_norm, tuple) and len(top_k_re_norm) == 2 and all(isinstance(v, bool) for v in top_k_re_norm), f"Expected top_k_re_norm=(write_bool, read_bool), got {top_k_re_norm!r}."

    (B, H, T, C), (_, _, U, P, _) = xs.shape, w_addr.shape
    A = P ** U
    w_top_k, r_top_k = top_k
    w_top_k_re_norm, r_top_k_re_norm = top_k_re_norm

    assert all(k <= A for k in top_k), f"Expected top_k <= A={A}, got {top_k}."
    assert isinstance(embed_num, int) and 0 <= embed_num <= H, f"Expected embed_num in [0, {H}], got {embed_num!r}."
    assert A < ADDRESS_LIMIT, f"Address space overflow: A={A}, limit={ADDRESS_LIMIT}."

    requires_grad = any(t.requires_grad for t in [w_addr, r_addr, xs, gamma, init_slot, init_mass])
    padded_T = T if T == 1 and not requires_grad else T + (32 - T % 32) % 32
    check_int_capacity(
        B, H, U, P, A, C, padded_T, embed_num, w_top_k, r_top_k,
        B * H * U * P * padded_T, B * H * A * C,
        B * H * padded_T * C, B * H * padded_T * (w_top_k + r_top_k),
    )
    position_ids = position_ids.to(device=xs.device, dtype=torch.int32, non_blocking=True)
    if mask is not None:
        mask = mask.to(device=xs.device, dtype=torch.bool, non_blocking=True)

    assert w_addr.dtype == r_addr.dtype == xs.dtype == gamma.dtype, f"Expected matching dtypes, got w={w_addr.dtype}, r={r_addr.dtype}, x={xs.dtype}, gamma={gamma.dtype}."
    assert w_addr.device == r_addr.device == xs.device == gamma.device == init_slot.device == init_mass.device == position_ids.device, \
        f"Expected one device, got w={w_addr.device}, r={r_addr.device}, x={xs.device}, gamma={gamma.device}, slot={init_slot.device}, mass={init_mass.device}, position_ids={position_ids.device}."
    assert mask is None or mask.device == position_ids.device, f"Expected mask.device={position_ids.device}, got {mask.device}."

    expected_shapes = (
        ("w_addr", w_addr, (B, H, U, P, T)),
        ("r_addr", r_addr, (B, H, U, P, T)),
        ("xs", xs, (B, H, T, C)),
        ("gamma", gamma, (B, H, T)),
        ("position_ids", position_ids, (B, T)),
        ("init_slot", init_slot, (B, H, A, C)),
        ("init_mass", init_mass, (B, H, A)),
    )
    for name, tensor, expected_shape in expected_shapes:
        assert tensor.shape == expected_shape, f"Expected {name}.shape == {expected_shape}, but got {tuple(tensor.shape)}."
    assert mask is None or mask.shape == (B, T), f"Expected mask.shape={(B, T)}, got {tuple(mask.shape)}."

    acc_type = torch.float32 if xs.dtype in (torch.float16, torch.bfloat16) else xs.dtype
    if inplace:
        assert not requires_grad, "inplace=True does not support backward."
        assert init_slot.dtype == init_mass.dtype == acc_type, f"Expected inplace state dtype={acc_type}, got slot={init_slot.dtype}, mass={init_mass.dtype}."
        assert init_slot.is_contiguous() and init_mass.is_contiguous(), f"Expected contiguous inplace state, got slot={init_slot.is_contiguous()}, mass={init_mass.is_contiguous()}."

    if T == 1 and not requires_grad:
        with torch.no_grad():
            post_slot = init_slot if inplace else init_slot.to(dtype=acc_type, memory_format=torch.contiguous_format, copy=True, non_blocking=True)
            post_mass = init_mass if inplace else init_mass.to(dtype=acc_type, memory_format=torch.contiguous_format, copy=True, non_blocking=True)
            mask = mask.squeeze(-1) if mask is not None else None
            ys = _cuda_topk_autoregressive_(
                w_addr.squeeze(-1), r_addr.squeeze(-1),
                xs.squeeze(-2), gamma.squeeze(-1),
                post_slot, post_mass,
                mask, embed_num, position_ids.squeeze(-1),
                w_top_k, r_top_k,
                w_top_k_re_norm, r_top_k_re_norm,
                mass_aware, eps,
            )
            ys = ys.unsqueeze(-2)
    else:
        T_align = 32
        if (pad_len := (T_align - T % T_align) % T_align) > 0:
            if not torch.compiler.is_compiling():
                warnings.warn(f"Expected T to be aligned with {T_align}. Automatic padding has been applied. This is NOT an ERROR but it may impact performance.")
            w_addr = torch.nn.functional.pad(w_addr, (0, pad_len), mode='constant', value=-torch.inf)
            r_addr = torch.nn.functional.pad(r_addr, (0, pad_len), mode='constant', value=-torch.inf)
            xs = torch.nn.functional.pad(xs, (0, 0, 0, pad_len), mode='constant', value=0)
            gamma = torch.nn.functional.pad(gamma, (0, pad_len), mode='constant', value=0)
            position_ids = torch.nn.functional.pad(position_ids, (0, pad_len), mode='constant', value=0)
            if mask is not None:
                mask = torch.nn.functional.pad(mask, (0, pad_len), mode='constant', value=False)

        addr_idx, addr_val = torch.ops.ramnet_ops.topk_addr_forward(
            w_addr, r_addr,
            embed_num, A, position_ids,
            w_top_k, r_top_k,
            w_top_k_re_norm, r_top_k_re_norm,
        )

        if not inplace:
            init_slot = init_slot.to(dtype=acc_type)
            init_mass = init_mass.to(dtype=acc_type)

        ys, post_slot, post_mass = _cuda_topk_parallel(
            addr_idx, addr_val,
            xs, gamma,
            init_slot, init_mass,
            mask,
            w_top_k, r_top_k,
            mass_aware, eps,
            inplace,
        )

        if ys.size(-2) > T:
            ys = ys[..., :T, :]

    return ys, post_slot, post_mass


def _cuda_topk_addr(
        addr: torch.Tensor,  # (B, H, U, P, T)

        embed_num: int,
        position_ids: torch.Tensor,  # (B, T)

        top_k: int,
        top_k_re_norm: bool,

        addr_val_neg: bool = False,
        encode_batch_idx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:  # shapes: int32 (B, H, K, T), float32 (B, H, K, T)
    """Decode one CUDA address tensor; gradients propagate through the values.

    Indices encode ``(b * H + h) * A + a_idx`` by default.
    Set ``encode_batch_idx=False`` to return ``h * A + a_idx`` in ``[0, H * A)``.
    CAPE offsets still apply in either mode.
    """
    assert addr.ndim == 5, f"Expected address shape (B, H, U, P, T), got {tuple(addr.shape)}."
    assert addr.is_cuda, f"Expected CUDA addresses, got {addr.device}."

    B, H, U, P, T = addr.shape
    A = P ** U
    assert A < ADDRESS_LIMIT, f"Address space overflow: A={A}, limit={ADDRESS_LIMIT}."
    assert isinstance(top_k, int) and 0 < top_k <= A, f"Expected top_k in [1, A={A}], got {top_k!r}."
    assert isinstance(embed_num, int) and 0 <= embed_num <= H, f"Expected embed_num in [0, {H}], got {embed_num!r}."
    assert isinstance(top_k_re_norm, bool), f"Expected boolean top_k_re_norm, got {top_k_re_norm!r}."
    assert isinstance(addr_val_neg, bool), f"Expected boolean addr_val_neg, got {addr_val_neg!r}."
    assert isinstance(encode_batch_idx, bool), f"Expected boolean encode_batch_idx, got {encode_batch_idx!r}."
    assert position_ids.shape == (B, T), f"Expected position_ids.shape={(B, T)}, got {tuple(position_ids.shape)}."
    padded_T = T + (32 - T % 32) % 32
    check_int_capacity(B, H, U, P, A, padded_T, embed_num, top_k,
                       B * H * U * P * padded_T, B * H * A, B * H * top_k * padded_T)
    position_ids = position_ids.to(device=addr.device, dtype=torch.int32, non_blocking=True)

    if (pad_len := (32 - T % 32) % 32) > 0:
        if not torch.compiler.is_compiling():
            warnings.warn(f"Expected T to be aligned with 32. Automatic padding has been applied. This is not an error, but it may impact performance.")
        addr = torch.nn.functional.pad(addr, (0, pad_len), mode='constant', value=-torch.inf)
        position_ids = torch.nn.functional.pad(position_ids, (0, pad_len), mode='constant', value=0)

    addr_idx, addr_val = torch.ops.ramnet_ops.topk_addr_single_forward(
        addr,
        embed_num, A, position_ids,
        top_k, top_k_re_norm,
        addr_val_neg, encode_batch_idx,
    )

    if addr_idx.size(-1) > T:
        addr_idx = addr_idx[..., :T]
    if addr_val.size(-1) > T:
        addr_val = addr_val[..., :T]

    return addr_idx, addr_val
