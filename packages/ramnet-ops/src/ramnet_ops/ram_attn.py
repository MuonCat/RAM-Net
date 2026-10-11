import torch
import warnings
from typing import Callable, Literal

from .impl.torch_impl import _torch_ram_attn

_triton_ram_attn = None
_cuda_load_error = None

try:
    from .impl.cuda_impl import _cuda_ram_attn
except Exception as e:
    _cuda_load_error = e
    warnings.warn(f"Failed to load the CUDA implementation: {e}. Install a compatible ramnet-ops CUDA wheel or build it with pip.")
    _cuda_ram_attn = None

ImplCallableType = Callable[
    [
        torch.Tensor,  # w_addr
        torch.Tensor,  # r_addr
        torch.Tensor,  # x
        torch.Tensor,  # gamma
        torch.Tensor,  # init_slot
        torch.Tensor,  # init_mass
        torch.Tensor | None,  # mask
        int,  # embed_num
        torch.Tensor,  # position_ids
        tuple[int, int] | None,  # top-k
        tuple[bool, bool],  # top-k renorm (write, read)
        bool,  # mass_aware
        float,  # eps
        bool,  # inplace
    ],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


def get_broadcast(*vals: int) -> int:
    """Return the non-singleton size shared by broadcast-compatible dimensions."""
    res = {v for v in vals if v != 1}
    assert len(res) <= 1, f"Cannot broadcast {vals}"
    return next(iter(res), 1)


def random_access_memory_attention(
        w_addr: torch.Tensor,
        r_addr: torch.Tensor,

        values: torch.Tensor,
        gamma: torch.Tensor | float = 2.197,

        init_ram: tuple[torch.Tensor | None, torch.Tensor | float | None] | None = None,

        mask: torch.Tensor | None = None,
        embed_num: int = 0,
        position_ids: torch.Tensor | None = None,

        top_k: tuple[int, int] | list[int] | int | None = 8,
        top_k_re_norm: bool | tuple[bool, bool] | list[bool] = False,

        mass_aware: bool = False,
        eps: float = 1e-12,

        t_dim: int = 1,
        inplace: bool = False,
        impl_backend: Literal['torch', 'triton', 'cuda'] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Random-access memory attention.

    Shape symbols:
        T: number of tokens (sequence length)
        H: number of heads
        U: order of address space (number of address digits)
        P: address radix (paper d_p), giving P ** U memory slots
        C: value channels

    :param w_addr: Write address logits in (..., T, ..., H, U, P).
    :param r_addr: Read address logits in (..., T, ..., H, U, P).
    :param values: Values to write into RAM (..., T, ..., H, C).
    :param gamma: Forget-rate control; <0 favors remembering and >0 favors forgetting, shape (..., T, ..., H).
    :param init_ram: Previous ``(slot, mass)`` RAM state. Slot may be None; mass may be a tensor, scalar, or None.
    :param mask: Valid-token mask, shape broadcastable to position_ids. True entries participate in RAM updates.
    :param embed_num: Number of heads using CAPE-style shift embedding (applied from head 0); must be in ``[0, H]``.
    :param position_ids: Token positions for cyclic address positional embedding, shape broadcastable to (..., T, ...).
    :param top_k: If set, keeps only the top-k write/read slots; a two-item tuple/list applies separate ``(write_k, read_k)`` values.
    :param top_k_re_norm: Whether to re-normalize the address vector after top-k filtering, forcing the top-k entries to sum to 1.0. A bool applies to both write and read; a two-item tuple/list specifies separate (write, read) bools.
    :param mass_aware: Weight reads by updated memory mass with epsilon-stabilized normalization.
    :param eps: Strictly positive value to avoid division by zero.
    :param t_dim: Index of the time dimension T.
    :param inplace: If True, update init_ram slot and mass tensors in-place. Supported by the Torch backend and CUDA top-k backend.
    :param impl_backend: Backend implementation: 'torch', 'triton', or 'cuda'. Use None for auto-selection.
    :return: ``(output, (slot, mass))`` with shapes ``(..., T, ..., H, C)``, ``(..., ..., H, P^U, C)``, and ``(..., ..., H, P^U)``.

    The CUDA backend supports float16, bfloat16, and float32 inputs in Top-K mode.
    """
    assert all(isinstance(tensor, torch.Tensor) for tensor in (r_addr, w_addr, values)), f"Expected r_addr/w_addr/values to be tensors, got {type(r_addr).__name__}, {type(w_addr).__name__}, {type(values).__name__}."

    if init_ram is None:
        init_slot, init_mass = None, None
    elif isinstance(init_ram, tuple) and len(init_ram) == 2:
        init_slot, init_mass = init_ram
    else:
        raise TypeError("init_ram must be a (slot, mass) tuple or None.")

    # check type
    assert init_slot is None or isinstance(init_slot, torch.Tensor), f"Expected init_slot to be Tensor or None, got {type(init_slot).__name__}."
    assert init_mass is None or isinstance(init_mass, (torch.Tensor, int, float)), f"Expected init_mass to be Tensor, scalar, or None, got {type(init_mass).__name__}."
    assert isinstance(gamma, (torch.Tensor, int, float)), f"Expected gamma to be Tensor or scalar, got {type(gamma).__name__}."
    assert position_ids is None or isinstance(position_ids, torch.Tensor), f"Expected position_ids to be Tensor or None, got {type(position_ids).__name__}."
    assert mask is None or isinstance(mask, torch.Tensor), f"Expected mask to be Tensor or None, got {type(mask).__name__}."
    assert isinstance(t_dim, int), f"Expected t_dim to be int, got {type(t_dim).__name__}."
    assert isinstance(embed_num, int) and embed_num >= 0, f"Expected embed_num >= 0, got {embed_num!r}."
    assert isinstance(eps, (int, float)) and eps > 0, f"Expected eps > 0, got {eps!r}."
    assert isinstance(mass_aware, bool), f"Expected mass_aware to be bool, got {type(mass_aware).__name__}."
    assert not inplace or isinstance(init_slot, torch.Tensor) and isinstance(init_mass, torch.Tensor), f"inplace=True requires Tensor state, got slot={type(init_slot).__name__}, mass={type(init_mass).__name__}."

    # check ndim
    min_ndims = (("w_addr", w_addr, 4), ("r_addr", r_addr, 4), ("values", values, 3))
    for name, tensor, min_ndim in min_ndims:
        assert tensor.ndim >= min_ndim, f"Expected {name}.ndim >= {min_ndim}, got shape={tuple(tensor.shape)}."
    assert not isinstance(gamma, torch.Tensor) or gamma.ndim >= 2, f"Expected gamma.ndim >= 2, got shape={tuple(gamma.shape)}."
    assert not isinstance(init_slot, torch.Tensor) or init_slot.ndim >= 3, f"Expected init_slot.ndim >= 3, got shape={tuple(init_slot.shape)}."
    assert not isinstance(init_mass, torch.Tensor) or init_mass.ndim >= 2, f"Expected init_mass.ndim >= 2, got shape={tuple(init_mass.shape)}."
    assert not isinstance(position_ids, torch.Tensor) or position_ids.ndim >= 1, f"Expected position_ids.ndim >= 1, got shape={tuple(position_ids.shape)}."
    assert not isinstance(mask, torch.Tensor) or mask.ndim >= 1, f"Expected mask.ndim >= 1, got shape={tuple(mask.shape)}."

    time_inputs = [("w_addr", w_addr, 3), ("r_addr", r_addr, 3), ("values", values, 2)]
    if isinstance(gamma, torch.Tensor):
        time_inputs.append(("gamma", gamma, 1))
    if isinstance(position_ids, torch.Tensor):
        time_inputs.append(("position_ids", position_ids, 0))
    if isinstance(mask, torch.Tensor):
        time_inputs.append(("mask", mask, 0))
    for name, tensor, trailing_dims in time_inputs:
        assert -tensor.ndim <= t_dim < tensor.ndim and t_dim % tensor.ndim < tensor.ndim - trailing_dims, f"t_dim={t_dim} does not select the time dimension of {name} with shape {tuple(tensor.shape)}."

    # check dtype
    assert w_addr.is_floating_point() and r_addr.is_floating_point() and values.is_floating_point(), f"Expected floating-point addresses/values, got {w_addr.dtype=}, {r_addr.dtype=}, {values.dtype=}."
    assert w_addr.dtype == r_addr.dtype == values.dtype, f"Expected matching dtypes, got {w_addr.dtype=}, {r_addr.dtype=}, {values.dtype=}."
    assert not isinstance(gamma, torch.Tensor) or gamma.dtype == values.dtype, f"Expected gamma.dtype={values.dtype}, got {gamma.dtype}."

    # Move the T dimension into the layout expected by the backends.
    w_addr = w_addr.movedim(t_dim, -1)  # (..., T, ..., H, U, P) ==> (..., ..., H, U, P, T)
    r_addr = r_addr.movedim(t_dim, -1)  # (..., T, ..., H, U, P) ==> (..., ..., H, U, P, T)
    values = values.movedim(t_dim, -2)  # (..., T, ..., H, C) ==> (..., ..., H, T, C)
    if isinstance(gamma, torch.Tensor):
        gamma = gamma.movedim(t_dim, -1)  # (..., T, ..., H) ==> (..., ..., H, T)
    if isinstance(position_ids, torch.Tensor):
        position_ids = position_ids.movedim(t_dim, -1)  # (..., T, ...) ==> (..., ..., T)
    if isinstance(mask, torch.Tensor):
        mask = mask.movedim(t_dim, -1)  # (..., T, ...) ==> (..., ..., T)

    # get the broadcast shape of inputs
    init_slot_shape = init_slot.shape if isinstance(init_slot, torch.Tensor) else (1, 1, 1)
    init_mass_shape = init_mass.shape if isinstance(init_mass, torch.Tensor) else (1, 1)
    gamma_shape = gamma.shape if isinstance(gamma, torch.Tensor) else (1, 1)
    position_shape = position_ids.shape if isinstance(position_ids, torch.Tensor) else (1,)
    mask_shape = mask.shape if isinstance(mask, torch.Tensor) else (1,)
    batch_shape = torch.broadcast_shapes(w_addr.shape[:-4], r_addr.shape[:-4], values.shape[:-3], gamma_shape[:-2], init_slot_shape[:-3], init_mass_shape[:-2], position_shape[:-1], mask_shape[:-1])
    H = get_broadcast(w_addr.size(-4), r_addr.size(-4), values.size(-3), gamma_shape[-2], init_slot_shape[-3], init_mass_shape[-2])
    U = get_broadcast(w_addr.size(-3), r_addr.size(-3))
    P = get_broadcast(w_addr.size(-2), r_addr.size(-2))
    A = get_broadcast(init_slot_shape[-2], init_mass_shape[-1], P ** U)
    C = get_broadcast(values.size(-1), init_slot_shape[-1])
    T = get_broadcast(w_addr.size(-1), r_addr.size(-1), values.size(-2), gamma_shape[-1], position_shape[-1], mask_shape[-1])
    assert all(size > 0 for size in (H, U, P, A, C, T)), f"RAM dimensions must be positive, got {H=}, {U=}, {P=}, {A=}, {C=}, {T=}."
    assert embed_num <= H, f"embed_num must not exceed H={H}, but got {embed_num}."
    if inplace:
        expected_slot_shape = (*batch_shape, H, A, C)
        expected_mass_shape = (*batch_shape, H, A)
        assert init_slot.shape == expected_slot_shape and init_mass.shape == expected_mass_shape, \
            f"inplace=True expects slot.shape={expected_slot_shape}, mass.shape={expected_mass_shape}; got {tuple(init_slot.shape)} and {tuple(init_mass.shape)}."
        assert init_slot.is_contiguous() and init_mass.is_contiguous(), \
            f"inplace=True requires contiguous init_ram tensors, got slot{init_slot.is_contiguous()=} and mass{init_mass.is_contiguous()=}."

    # auto set up the default value of previous if they are None
    if not isinstance(gamma, torch.Tensor):
        gamma = values.new_full(size=(*batch_shape, H, 1), fill_value=gamma).expand(*batch_shape, H, T)
    if not isinstance(init_slot, torch.Tensor):
        init_slot = values.new_zeros(*batch_shape, H, A, C)
    if not isinstance(init_mass, torch.Tensor):
        mass_val = 1.0 / (P ** U) if init_mass is None else init_mass
        init_mass = values.new_full(size=(*batch_shape, H, A), fill_value=mass_val)
    if not isinstance(position_ids, torch.Tensor):
        position_ids = torch.arange(T, device=values.device, dtype=torch.long).view(*([1] * len(batch_shape)), T)
    else:
        position_ids = position_ids.to(device=values.device)
    if isinstance(mask, torch.Tensor):
        mask = mask.to(device=values.device, dtype=torch.bool)

    # broadcast the inputs
    w_addr = w_addr.broadcast_to([*batch_shape, H, U, P, T])
    r_addr = r_addr.broadcast_to([*batch_shape, H, U, P, T])
    values = values.broadcast_to([*batch_shape, H, T, C])
    gamma = gamma.broadcast_to([*batch_shape, H, T])
    init_slot = init_slot.broadcast_to([*batch_shape, H, A, C])
    init_mass = init_mass.broadcast_to([*batch_shape, H, A])
    position_ids = position_ids.broadcast_to([*batch_shape, T])
    if isinstance(mask, torch.Tensor):
        mask = mask.broadcast_to([*batch_shape, T])

    # check device and input shapes
    assert w_addr.device == r_addr.device == values.device == gamma.device == init_slot.device == init_mass.device == position_ids.device, \
        f"Expected one device, got w={w_addr.device}, r={r_addr.device}, values={values.device}, gamma={gamma.device}, slot={init_slot.device}, mass={init_mass.device}, position_ids={position_ids.device}."
    assert mask is None or mask.device == position_ids.device, f"Expected mask.device={position_ids.device}, got {mask.device}."

    expected_shapes = (
        ("w_addr", w_addr, (*batch_shape, H, U, P, T)),
        ("r_addr", r_addr, (*batch_shape, H, U, P, T)),
        ("values", values, (*batch_shape, H, T, C)),
        ("gamma", gamma, (*batch_shape, H, T)),
        ("init_slot", init_slot, (*batch_shape, H, A, C)),
        ("init_mass", init_mass, (*batch_shape, H, A)),
        ("position_ids", position_ids, (*batch_shape, T)),
    )
    for name, tensor, expected_shape in expected_shapes:
        assert tensor.shape == expected_shape, f"Expected {name}.shape == {expected_shape}, but got {tuple(tensor.shape)}."
    expected_mask_shape = (*batch_shape, T)
    assert mask is None or mask.shape == expected_mask_shape, f"Expected mask.shape={expected_mask_shape}, got {tuple(mask.shape)}."

    # Pack all leading broadcast dimensions into one CUDA/PyTorch batch axis.
    w_addr = w_addr.reshape(-1, H, U, P, T)  # (..., ..., H, U, P, T) ==> (B, H, U, P, T)
    r_addr = r_addr.reshape(-1, H, U, P, T)  # (..., ..., H, U, P, T) ==> (B, H, U, P, T)
    values = values.reshape(-1, H, T, C)  # (..., ..., H, T, C) ==> (B, H, T, C)
    gamma = gamma.reshape(-1, H, T)  # (..., ..., H, T) ==> (B, H, T)
    init_slot = init_slot.reshape(-1, H, A, C)  # (..., ..., H, A, C) ==> (B, H, A, C)
    init_mass = init_mass.reshape(-1, H, A)  # (..., ..., H, A) ==> (B, H, A)
    position_ids = position_ids.reshape(-1, T)  # (..., ..., T) ==> (B, T)
    if isinstance(mask, torch.Tensor):
        mask = mask.reshape(-1, T)  # (..., ..., T) ==> (B, T)

    # check the top-k
    if isinstance(top_k, int):
        top_k = (top_k, top_k)
    elif isinstance(top_k, tuple | list):
        if len(top_k) != 2 or not all(isinstance(k, int) for k in top_k):
            raise TypeError("`top_k` must be int, a two-item tuple/list of ints, or None.")
        top_k = tuple(top_k)
    elif top_k is not None:
        raise TypeError("`top_k` must be int, a two-item tuple/list of ints, or None.")

    assert top_k is None or all(0 < k <= P ** U for k in top_k), f"top_k values must be in the range (0, {P ** U}], but got {top_k}"

    # check the top-k re-norm
    top_k_re_norm_type_error = "`top_k_re_norm` must be a bool or a two-item tuple/list of bools."
    if isinstance(top_k_re_norm, bool):
        top_k_re_norm = (top_k_re_norm, top_k_re_norm)
    elif isinstance(top_k_re_norm, tuple | list):
        if len(top_k_re_norm) != 2 or not all(isinstance(v, bool) for v in top_k_re_norm):
            raise TypeError(top_k_re_norm_type_error)
        top_k_re_norm = tuple(top_k_re_norm)
    else:
        raise TypeError(top_k_re_norm_type_error)

    if impl_backend not in (None, 'torch', 'triton', 'cuda'):
        raise ValueError(f"Unsupported impl_backend: {impl_backend!r}.")

    # select impl function
    if impl_backend is None:
        impl_backend = 'torch'
        if values.is_cuda:
            if _cuda_ram_attn is not None and top_k is not None and values.dtype in (torch.float16, torch.bfloat16, torch.float32):
                impl_backend = 'cuda'
            elif _triton_ram_attn is not None and top_k is None:
                impl_backend = 'triton'

            if impl_backend != 'cuda':
                warnings.warn(
                    f"CUDA backend is unavailable for this input/configuration; falling back to {impl_backend}. "
                    "Input is a CUDA tensor, so this may significantly reduce performance and increase GPU memory usage. "
                    "Check your ramnet-ops installation, or manually set 'impl_backend' to suppress this warning."
                )

    impl_func: ImplCallableType = {
        'torch': _torch_ram_attn,
        'cuda': _cuda_ram_attn,
        'triton': _triton_ram_attn,
    }[impl_backend]

    if inplace and impl_backend not in ('torch', 'cuda'):
        raise ValueError("inplace=True is only supported by the Torch backend and CUDA top-k backend.")
    if impl_backend == 'cuda' and _cuda_ram_attn is None:
        raise RuntimeError("CUDA RAM attention backend is unavailable.") from _cuda_load_error

    # calculate the ram_attn
    y, slot, mass = impl_func(
        w_addr, r_addr,
        values, gamma,
        init_slot, init_mass,
        mask, embed_num, position_ids,
        top_k, top_k_re_norm,
        mass_aware, eps,
        inplace,
    )

    # unpack the result
    y = y.reshape(*batch_shape, H, T, C)
    slot = slot.reshape(*batch_shape, H, A, C)
    mass = mass.reshape(*batch_shape, H, A)

    # reset the t_dim
    y = y.movedim(-2, t_dim)
    return y, (slot, mass)
