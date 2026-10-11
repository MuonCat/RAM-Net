"""RAM attention inputs, state carry, gradients, and comparison pairs."""

from dataclasses import dataclass, replace

import torch

from tests.helpers import select_valid_tokens

RamFwd = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
RamBwd = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
RamResult = tuple[RamFwd, RamBwd | None]

TOP_K = (8, 8)
TOP_K_SETTINGS = ((4, 4), (8, 8), (4, 12), (16, 16), (32, 32), (64, 64), (128, 128))
TOP_K_RE_NORM_SETTINGS = ((False, False), (True, True), (False, True), (True, False))
EMBED_NUM = 4
EPS = 1e-12


@dataclass(frozen=True)
class RamCase:
    w_addr: torch.Tensor
    r_addr: torch.Tensor
    values: torch.Tensor
    gamma: torch.Tensor
    init_slot: torch.Tensor
    init_mass: torch.Tensor
    d_y: torch.Tensor
    d_slot: torch.Tensor
    d_mass: torch.Tensor
    position_ids: torch.Tensor
    mask: torch.Tensor | None


def make_ram_case(
        *,
        batch: int = 1,
        heads: int = 8,
        order: int = 5,
        radix: int = 4,
        channels: int = 32,
        seq_len: int = 1024,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        masked: bool = True,
) -> RamCase:
    slots = radix ** order
    w_addr = torch.randn(batch, heads, order, radix, seq_len, device=device, dtype=dtype)
    r_addr = torch.randn(batch, heads, order, radix, seq_len, device=device, dtype=dtype)
    values = torch.randn(batch, heads, seq_len, channels, device=device, dtype=dtype)
    gamma = 0.2 * torch.randn(batch, heads, seq_len, device=device, dtype=dtype)
    init_slot = torch.randn(batch, heads, slots, channels, device=device, dtype=dtype)
    init_mass = torch.rand(batch, heads, slots, device=device, dtype=dtype) + 0.1
    d_y = torch.randn(batch, heads, seq_len, channels, device=device, dtype=dtype)
    d_slot = torch.randn(batch, heads, slots, channels, device=device, dtype=dtype)
    d_mass = torch.randn(batch, heads, slots, device=device, dtype=dtype)

    if masked:
        mask = torch.ones(batch, seq_len, device=device, dtype=torch.bool)
        pad_width = max(1, seq_len // 8)
        mid_start = seq_len // 2 - pad_width // 2
        mask[:, :pad_width] = False
        mask[:, mid_start:mid_start + pad_width] = False
        mask[:, -pad_width:] = False
        position_ids = mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(~mask, 0)
    else:
        mask = None
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len).expand(batch, -1)

    return RamCase(
        w_addr=w_addr,
        r_addr=r_addr,
        values=values,
        gamma=gamma,
        init_slot=init_slot,
        init_mass=init_mass,
        d_y=d_y,
        d_slot=d_slot,
        d_mass=d_mass,
        position_ids=position_ids,
        mask=mask,
    )


def cast_ram_case(case: RamCase, dtype: torch.dtype) -> RamCase:
    # State and its upstream gradients stay in the FP32 accumulation dtype.
    return replace(
        case,
        w_addr=case.w_addr.to(dtype),
        r_addr=case.r_addr.to(dtype),
        values=case.values.to(dtype),
        gamma=case.gamma.to(dtype),
        d_y=case.d_y.to(dtype),
    )


def compact_valid_tokens(case: RamCase) -> RamCase:
    assert case.mask is not None
    valid = case.mask[0]
    assert torch.equal(case.mask, valid.expand_as(case.mask))
    return replace(
        case,
        w_addr=case.w_addr[..., valid],
        r_addr=case.r_addr[..., valid],
        values=case.values[:, :, valid, :],
        gamma=case.gamma[..., valid],
        d_y=case.d_y[:, :, valid, :],
        position_ids=case.position_ids[:, valid],
        mask=None,
    )


def gather_ram_result(
        func,
        case: RamCase,
        *,
        top_k: tuple[int, int] = TOP_K,
        top_k_re_norm: tuple[bool, bool] = (False, False),
        chunk_size: int | None = None,
        grad: bool = True,
        mass_aware: bool = False,
        eps: float = EPS,

        inplace: bool = False,
) -> RamResult:
    if inplace and grad:
        raise ValueError("gather_ram_result requires grad=False when inplace=True.")

    init_slot = case.init_slot.detach().clone()
    init_mass = case.init_mass.detach().clone()
    if inplace:
        acc_type = torch.float32 if case.values.dtype in (torch.float16, torch.bfloat16) else case.values.dtype
        init_slot = init_slot.to(dtype=acc_type)
        init_mass = init_mass.to(dtype=acc_type)

    inputs = [
        case.w_addr.detach().requires_grad_(grad),
        case.r_addr.detach().requires_grad_(grad),
        case.values.detach().requires_grad_(grad),
        case.gamma.detach().requires_grad_(grad),
        init_slot.requires_grad_(grad),
        init_mass.requires_grad_(grad),
    ]
    w_addr, r_addr, values, gamma, slot, mass = inputs
    position_ids = case.position_ids
    mask = case.mask

    seq_len = values.size(-2)
    chunk_size = seq_len if chunk_size is None else chunk_size
    ys = []
    for start in range(0, seq_len, chunk_size):
        steps = slice(start, start + chunk_size)
        y, post_slot, post_mass = func(
            w_addr[..., steps], r_addr[..., steps],
            values[..., steps, :], gamma[..., steps],
            slot, mass,
            mask[:, steps] if mask is not None else None, EMBED_NUM, position_ids[:, steps],
            top_k, top_k_re_norm,
            mass_aware, eps,
            inplace,
        )
        if inplace and (post_slot.data_ptr() != slot.data_ptr() or post_mass.data_ptr() != mass.data_ptr()):
            raise AssertionError("inplace=True expected the backend to return tensors backed by the input slot and mass storage.")
        slot, mass = post_slot, post_mass
        ys.append(y)

    fwd = (torch.cat(ys, dim=-2), slot, mass)
    bwd = None

    if grad:
        upstream_y = case.d_y
        if mask is not None:
            upstream_y = upstream_y * mask[:, None, :, None].to(device=case.d_y.device, dtype=case.d_y.dtype)
        grads = torch.autograd.grad(fwd, inputs, (upstream_y, case.d_slot, case.d_mass))
        bwd = tuple(t.detach().double() for t in grads)

    return tuple(t.detach().double() for t in fwd), bwd


def ram_masked_pairs(
        res: RamResult,
        ref: RamResult,
        *,
        res_mask: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    res_fwd, res_bwd = res
    ref_fwd, ref_bwd = ref
    fwd_labels = ("y", "slot", "mass")
    pairs = [(label, val, cmp) for label, val, cmp in zip(fwd_labels, res_fwd, ref_fwd)]

    if res_bwd is not None and ref_bwd is not None:
        grad_labels = ("D_w", "D_r", "D_x", "D_gamma", "D_slot", "D_mass")
        pairs.extend((label, val, cmp) for label, val, cmp in zip(grad_labels, res_bwd, ref_bwd))

    masked: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    time_dims = {
        "y": -2,
        "D_w": -1,
        "D_r": -1,
        "D_x": -2,
        "D_gamma": -1,
    }
    for label, val, cmp in pairs:
        if label in time_dims:
            t_dim = time_dims[label]
            if res_mask is not None:
                val = select_valid_tokens(val, res_mask, t_dim=t_dim)
            if ref_mask is not None:
                cmp = select_valid_tokens(cmp, ref_mask, t_dim=t_dim)
            if res_mask is not None or ref_mask is not None:
                val = val.reshape(-1)
                cmp = cmp.reshape(-1)
        masked.append((label, val, cmp))
    return masked
