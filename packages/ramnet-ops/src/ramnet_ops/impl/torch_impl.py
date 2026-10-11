import torch
import einops
import functools
import numpy as np


def _torch_impl_gsu(
        ws: torch.Tensor,
        rs: torch.Tensor,

        xs: torch.Tensor,
        gamma: torch.Tensor,

        slot: torch.Tensor,
        mass: torch.Tensor,

        mass_aware: bool = False,
        eps: float = 1e-12,

        inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # shapes: (T, ..., C), (..., A, C), (..., A)
    """
    Compute the power-decay moving average:

    alpha_i = sigmoid( - logit(w_i) - g_i)
    mass_i = alpha_i * mass_{i-1} + w_i
    gain_i = w_i / (mass_i + eps)
    slot_i = slot_{i-1} + gain_i * (x_i - slot_{i-1})

    y_i = sum_a slot_i[a] * r_i[a]
        = sum_a slot_i[a] * sum_a{r_i[a]} * r_i[a] * mass_i[a] / (sum_a{r_i[a] * mass_i[a]} + eps)

    :param ws: Write weight tensor in (T, ..., A), where A is the memory channel.
    :param rs: Read weight tensor in (T, ..., A), where A is the memory channel.
    :param xs: Input tensor in (T, ..., C), where C is the value channel.
    :param gamma: Controls forgetting of previous data (T, ...)
                    <0: trend to remember
                    >0: trend to forget
    :param slot: Initial slot state in (..., A, C).
    :param mass: Initial mass state in (..., A).
    :param mass_aware: Weight reads by updated mass with epsilon-stabilized normalization.
    :param eps: Small value for numerical stability.
    :param inplace: Whether to update ``slot`` and ``mass`` in place.
    :return: ``(y, post_slot, post_mass)`` in shapes ``(T, ..., C)``, ``(..., A, C)``, and ``(..., A)``.
    """

    xs = xs.unsqueeze(dim=-2).to(slot.dtype)  # (T, ..., 1, C)
    ws = ws.unsqueeze(dim=-1).to(slot.dtype)  # (T, ..., A, 1)
    rs = rs.to(slot.dtype)  # (T, ..., A)
    mass_state = mass
    mass = mass.unsqueeze(dim=-1)  # (..., A, 1)
    ys = []
    for i in range(xs.size(0)):
        w, r, x, g = ws[i], rs[i], xs[i], gamma[i, ..., None, None].to(slot.dtype)  # (..., A, 1), (..., A), (..., 1, C), (..., 1, 1)

        comp_w = 1 - w
        alpha = comp_w / (comp_w + torch.exp(g) * w)
        if inplace:
            mass.mul_(alpha).add_(w)  # (..., A, 1)
        else:
            mass = alpha * mass + w  # (..., A, 1)
        gain = w / (mass + eps)
        if inplace:
            slot.add_(gain * (x - slot))  # (..., A, C)
        else:
            slot = slot + gain * (x - slot)  # (..., A, C)

        if mass_aware:
            mass_read = mass.squeeze(-1) * r
            r = r.sum(dim=-1, keepdim=True) * mass_read / (mass_read.sum(dim=-1, keepdim=True) + eps)

        y = torch.einsum('...ac, ...a-> ...c', slot, r)
        ys.append(y)

    return torch.stack(ys, dim=0), slot, mass_state if inplace else mass.squeeze(dim=-1)


def circular_shift_embed(xs: torch.Tensor, position_ids: torch.Tensor, embed_num: int = 0) -> torch.Tensor:
    """Apply CAPE shift to decoded address weights in ``(T, B, H, A)``.

    `_torch_ram_attn` decodes addresses to (T, B, H, A). Position ids stay in
    the backend input layout (B, T), so transpose them explicitly before
    broadcasting over heads and address slots. ``embed_num`` must be in
    ``[0, H]``; zero returns ``xs`` unchanged.
    """
    assert isinstance(xs, torch.Tensor) and isinstance(position_ids, torch.Tensor), f"Expected xs/position_ids tensors, got {type(xs).__name__}/{type(position_ids).__name__}."
    assert isinstance(embed_num, int) and embed_num >= 0, f"Expected embed_num >= 0, got {embed_num!r}."
    if embed_num == 0:
        return xs
    assert xs.ndim == 4, f"Expected xs in (T, B, H, A), but got shape {tuple(xs.shape)}."

    T, B, H, A = xs.shape
    assert embed_num <= H, f"embed_num must not exceed H={H}, but got {embed_num}."
    assert position_ids.shape == (B, T), f"Expected position_ids.shape={(B, T)}, got {tuple(position_ids.shape)}."

    position_ids = position_ids.to(device=xs.device, dtype=torch.long)
    position_ids = einops.rearrange(position_ids, 'b t -> t b 1 1')
    head_idx = einops.rearrange(torch.arange(H, device=xs.device, dtype=torch.long), 'h -> 1 1 h 1')
    slot_idx = einops.rearrange(torch.arange(A, device=xs.device, dtype=torch.long), 'a -> 1 1 1 a')
    shift = (head_idx < embed_num).long() * position_ids
    indices = (slot_idx + shift) % A
    return torch.gather(xs, -1, indices.expand_as(xs))


def kronecker_sum(xs: torch.Tensor) -> torch.Tensor:
    """Decode factorized log-address digits from (..., U, P) to (..., P ** U)."""
    assert isinstance(xs, torch.Tensor), f"Expected xs to be Tensor, got {type(xs).__name__}."
    assert xs.ndim >= 2 and xs.size(-2) > 0 and xs.size(-1) > 0, f"Expected a non-empty (..., U, P) tensor, but got shape {tuple(xs.shape)}."
    padding_idx = np.flip(np.arange(xs.size(-2)))  # [..., 3, 2, 1, 0]
    # noinspection PyTypeChecker
    combo_sum: torch.Tensor = functools.reduce(torch.add, [t.view(*list(t.size()[:-1]), *np.where(i == padding_idx, -1, 1)) for i, t in enumerate(xs.contiguous().unbind(-2))])
    return combo_sum.flatten(start_dim=xs.dim() - 2)


def mask_top_k(val: torch.Tensor, top_k: int | None = None, top_k_re_norm: bool = True) -> torch.Tensor:
    """Mask all but the largest ``top_k`` log-values.

    When filtering removes values, ``top_k_re_norm=True`` applies log-softmax
    to the retained values. ``top_k`` must be in ``[1, val.size(-1)]``.
    """
    assert isinstance(val, torch.Tensor), f"Expected val to be Tensor, got {type(val).__name__}."
    assert val.ndim > 0 and val.size(-1) > 0, f"Expected val with non-empty final dimension, got shape={tuple(val.shape)}."
    assert top_k is None or isinstance(top_k, int), f"Expected top_k to be int or None, got {type(top_k).__name__}."
    assert top_k is None or 0 < top_k <= val.size(-1), f"top_k must be in [1, {val.size(-1)}], but got {top_k}."
    if top_k is not None and top_k < val.size(-1):
        _, idx = torch.topk(val, top_k, dim=-1)
        mask = torch.zeros_like(val, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, value=True)
        val = val.masked_fill(~mask, -torch.inf)
        if top_k_re_norm:
            val = torch.log_softmax(val, dim=-1)
    return val


def addr_decode(addr: torch.Tensor, position_ids: torch.Tensor, embed_num: int = 0, top_k: int | None = None, top_k_re_norm: bool = False) -> torch.Tensor:
    """Decode logits from ``(T, B, H, U, P)`` to weights in ``(T, B, H, P ** U)``."""
    assert isinstance(addr, torch.Tensor) and isinstance(position_ids, torch.Tensor), f"Expected addr/position_ids tensors, got {type(addr).__name__}/{type(position_ids).__name__}."
    assert addr.ndim == 5, f"Expected addr in (T, B, H, U, P), but got shape {tuple(addr.shape)}."
    # Decode the factorized softmax in log-domain before exponentiating once at the end.
    addr = kronecker_sum(addr)  # ( ..., U, P ) ==> ( ..., P^U )

    # cyclic shift position embedding
    addr = circular_shift_embed(addr, position_ids=position_ids, embed_num=embed_num)

    # apply top-K mask
    addr = mask_top_k(addr, top_k=top_k, top_k_re_norm=top_k_re_norm)

    # log-domain ==> real
    return addr.exp()


def _torch_ram_attn(
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
    """Readable reference implementation used for correctness checks and CPU fallback."""
    acc_type = torch.float32 if xs.dtype in (torch.float16, torch.bfloat16) else xs.dtype
    if inplace:
        assert not any(t.requires_grad for t in [w_addr, r_addr, xs, gamma, init_slot, init_mass]), "inplace=True does not support backward."
        assert init_slot.dtype == init_mass.dtype == acc_type, f"Expected inplace state dtype={acc_type}, got slot={init_slot.dtype}, mass={init_mass.dtype}."

    w_top_k, r_top_k = top_k or (None, None)
    assert isinstance(top_k_re_norm, tuple) and len(top_k_re_norm) == 2 and all(isinstance(v, bool) for v in top_k_re_norm), f"Expected top_k_re_norm=(write_bool, read_bool), got {top_k_re_norm!r}."
    w_top_k_re_norm, r_top_k_re_norm = top_k_re_norm
    B, _, T, _ = xs.shape
    assert position_ids.shape == (B, T), f"Expected position_ids.shape={(B, T)}, got {tuple(position_ids.shape)}."
    position_ids = position_ids.to(device=xs.device, dtype=torch.long, non_blocking=True)

    # promotion type for address decoding
    w_addr = torch.log_softmax(w_addr, dim=-2, dtype=acc_type)
    r_addr = torch.log_softmax(r_addr, dim=-2, dtype=acc_type)

    # decode address
    w_addr = addr_decode(
        einops.rearrange(w_addr, '... u p t -> t ... u p'),
        position_ids=position_ids,
        embed_num=embed_num, top_k=w_top_k, top_k_re_norm=w_top_k_re_norm,
    )  # (T, B, H, A)
    r_addr = addr_decode(
        einops.rearrange(r_addr, '... u p t -> t ... u p'),
        position_ids=position_ids,
        embed_num=embed_num, top_k=r_top_k, top_k_re_norm=r_top_k_re_norm,
    )  # (T, B, H, A)

    if mask is not None:
        valid = einops.rearrange(mask.to(device=xs.device, dtype=w_addr.dtype), 'b t -> t b 1 1')
        w_addr = w_addr * valid  # no write on the invalid token

    # promotion type for accumulate safe
    if not inplace:
        init_slot = init_slot.to(dtype=acc_type)
        init_mass = init_mass.to(dtype=acc_type)

    # GSU
    ys, post_slot, post_mass = _torch_impl_gsu(
        w_addr, r_addr,
        einops.rearrange(xs, '... t c -> t ... c'), einops.rearrange(gamma, '... t -> t ...'),
        init_slot, init_mass,
        mass_aware, eps,
        inplace,
    )

    ys = einops.rearrange(ys, 't ... c -> ... t c').to(dtype=xs.dtype)

    return ys, post_slot, post_mass
