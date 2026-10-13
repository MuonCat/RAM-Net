import torch


@torch.library.register_fake("ramnet_ops::topk_addr_single_forward")
def _addr_single_forward_fake(
        addr,  # r (B, H, U, P, T)
        embed_num, A,
        position_ids,  # r (B, T)
        top_k, top_k_re_norm,
        addr_val_neg=False,
        encode_batch_idx=True,
):  # idx, val [ (B, H, K, T), (B, H, K, T) ]
    B, H, _, _, T = addr.shape
    shape = (B, H, top_k, T)
    return addr.new_empty(shape, dtype=torch.int32), addr.new_empty(shape, dtype=torch.float32)  # w , idx/val


@torch.library.register_fake("ramnet_ops::topk_addr_single_backward")
def _addr_single_backward_fake(
        idx,  # r (B, H, K, T)
        val,  # r (B, H, K, T)

        addr,  # r (B, H, U, P, T), optional logits

        D_val,  # r (B, H, K, T)

        embed_num,
        position_ids,  # r (B, T)

        top_k,
        top_k_re_norm,

        addr_val_neg,

        shape,  # (B, H, U, P, A, T)
        dtype,
):  # D_addr (B, H, U, P, T)
    B, H, U, P, _, T = shape
    return val.new_empty((B, H, U, P, T), dtype=dtype)


def _addr_single_setup(ctx, inputs, output):
    addr, embed_num, A, position_ids, top_k, top_k_re_norm, addr_val_neg, _ = inputs
    idx, val = output
    B, H, U, P, T = addr.shape

    ctx.embed_num = embed_num
    ctx.top_k = top_k
    ctx.top_k_re_norm = top_k_re_norm
    ctx.addr_val_neg = addr_val_neg
    ctx.shape = (B, H, U, P, A, T)
    ctx.dtype = addr.dtype

    ctx.save_for_backward(
        idx, val,
        position_ids,
        None if top_k_re_norm else addr,
    )
    ctx.mark_non_differentiable(idx)
    ctx.set_materialize_grads(False)


def _addr_single_backward(
        ctx,

        D_idx,  # r (B, H, K, T), non-differentiable (None)
        D_val  # r (B, H, K, T), or None
):
    if D_val is None:
        return (None,) * len(ctx.needs_input_grad)

    idx, val, position_ids, addr = ctx.saved_tensors
    D_addr = torch.ops.ramnet_ops.topk_addr_single_backward(
        idx, val,
        addr,
        D_val,
        ctx.embed_num, position_ids,
        ctx.top_k,
        ctx.top_k_re_norm, ctx.addr_val_neg,
        ctx.shape, ctx.dtype,
    )
    return (D_addr,) + (None,) * 7


torch.library.register_autograd(
    "ramnet_ops::topk_addr_single_forward",
    _addr_single_backward,
    setup_context=_addr_single_setup
)


@torch.library.register_fake("ramnet_ops::topk_addr_forward")
def _addr_forward_fake(
        w_addr,  # r (B, H, U, P, T)
        r_addr,  # r (B, H, U, P, T)

        embed_num, A,
        position_ids,  # r (B, T)

        w_top_k, r_top_k,
        w_re_norm, r_re_norm,
        encode_batch_idx=True,
):  # idx, val [ (B, H, Kw + Kr, T), (B, H, Kw + Kr, T) ]
    B, H, _, _, T = w_addr.shape
    shape = (B, H, w_top_k + r_top_k, T)
    return w_addr.new_empty(shape, dtype=torch.int32), w_addr.new_empty(shape, dtype=torch.float32)


@torch.library.register_fake("ramnet_ops::topk_addr_backward")
def _addr_backward_fake(
        idx,  # r (B, H, Kw + Kr, T)
        val,  # r (B, H, Kw + Kr, T)

        w_addr,  # r (B, H, U, P, T), optional logits
        r_addr,  # r (B, H, U, P, T), optional logits

        D_val,  # r (B, H, Kw + Kr, T), read gradients w.r.t. positive weights

        embed_num,
        position_ids,  # r (B, T)

        w_top_k, r_top_k,
        w_re_norm, r_re_norm,

        shape,  # (B, H, U, P, A, T)
        dtype,
):  # D_w, D_r [ (B, H, U, P, T), (B, H, U, P, T) ]
    B, H, U, P, _, T = shape
    return val.new_empty((B, H, U, P, T), dtype=dtype), val.new_empty((B, H, U, P, T), dtype=dtype)


def _addr_setup(ctx, inputs, output):
    w_addr, r_addr, embed_num, A, position_ids, w_top_k, r_top_k, w_re_norm, r_re_norm, _ = inputs
    idx, val = output
    B, H, U, P, T = w_addr.shape

    ctx.embed_num = embed_num
    ctx.w_top_k, ctx.r_top_k = w_top_k, r_top_k
    ctx.w_re_norm, ctx.r_re_norm = w_re_norm, r_re_norm
    ctx.shape = (B, H, U, P, A, T)
    ctx.dtype = w_addr.dtype

    ctx.save_for_backward(
        idx, val,
        position_ids,
        None if w_re_norm else w_addr, None if r_re_norm else r_addr,
    )
    ctx.mark_non_differentiable(idx)
    ctx.set_materialize_grads(False)


def _addr_backward(
        ctx,

        D_idx,  # r (B, H, Kw + Kr, T), non-differentiable (None)
        D_val  # r (B, H, Kw + Kr, T), read gradients w.r.t. positive weights, or None
):
    if D_val is None:
        return (None,) * len(ctx.needs_input_grad)

    idx, val, position_ids, w_addr, r_addr = ctx.saved_tensors
    D_w, D_r = torch.ops.ramnet_ops.topk_addr_backward(
        idx, val,
        w_addr, r_addr,
        D_val,
        ctx.embed_num, position_ids,
        ctx.w_top_k, ctx.r_top_k,
        ctx.w_re_norm, ctx.r_re_norm,
        ctx.shape, ctx.dtype,
    )

    return (D_w if ctx.needs_input_grad[0] else None, D_r if ctx.needs_input_grad[1] else None) + (None,) * 8


torch.library.register_autograd(
    "ramnet_ops::topk_addr_forward",
    _addr_backward,
    setup_context=_addr_setup
)


@torch.library.register_fake("ramnet_ops::topk_segment_forward")
def _segment_forward_fake(
        addr_val,  # r (B, H, Kw + Kr, T)

        seg_idx,  # r (S)
        seg_off,  # r (S)
        seg_len,  # r (S)
        event_i,  # r (B * H * T * (Kw + Kr))

        x,  # r (B, H, T, C)
        gamma,  # r (B, H, T)

        init_slot,  # r (B, H, A, C)
        init_mass,  # r (B, H, A)

        w_top_k,
        r_top_k,

        mass_aware,
        eps,
):  # y, slot, mass, event_v [ (B, H, T, C), (B, H, A, C), (B, H, A), (B * H * T * (Kw + Kr)) ]
    return x.new_empty(x.shape, dtype=torch.float32), init_slot.new_empty(init_slot.shape), init_mass.new_empty(init_mass.shape), addr_val.new_empty(event_i.shape)


@torch.library.register_fake("ramnet_ops::topk_segment_backward")
def _segment_backward_fake(
        seg_idx,  # r (S)
        seg_off,  # r (S)
        seg_len,  # r (S)
        event_i,  # r (B * H * T * (Kw + Kr))
        event_v,  # r (B * H * T * (Kw + Kr))

        x,  # r (B, H, T, C)
        gamma,  # r (B, H, T)

        init_slot,  # r (B, H, A, C)
        init_mass,  # r (B, H, A)

        D_y,  # r (B, H, T, C)
        D_post_slot,  # r (B, H, A, C)
        D_post_mass,  # r (B, H, A)

        w_top_k,
        r_top_k,

        mass_aware,
        eps,
):  # D_addr, D_x, D_gamma, D_slot, D_mass [ (B, H, Kw + Kr, T), (B, H, T, C), (B, H, T), (B, H, A, C), (B, H, A) ]
    B, H, T, _ = x.shape
    shape = (B, H, w_top_k + r_top_k, T)
    return (
        x.new_empty(shape, dtype=torch.float32),
        x.new_empty(x.shape, dtype=torch.float32), gamma.new_empty(gamma.shape, dtype=torch.float32),
        init_slot.new_empty(init_slot.shape), init_mass.new_empty(init_mass.shape)
    )


def _segment_setup(ctx, inputs, output):
    _, seg_idx, seg_off, seg_len, event_i, x, gamma, init_slot, init_mass, w_top_k, r_top_k, mass_aware, eps = inputs
    _, _, _, event_v = output

    ctx.w_top_k, ctx.r_top_k = w_top_k, r_top_k
    ctx.mass_aware = mass_aware
    ctx.eps = eps
    ctx.save_for_backward(
        seg_idx, seg_off, seg_len, event_i, event_v,
        x, gamma,
        init_slot, init_mass,
    )
    ctx.mark_non_differentiable(event_v)
    ctx.set_materialize_grads(False)


def _segment_backward(
        ctx,

        D_y,  # r (B, H, T, C), or None

        D_slot,  # r (B, H, A, C), or None
        D_mass,  # r (B, H, A), or None

        D_event_v  # r (B * H * T * (Kw + Kr)), non-differentiable (None)
):
    seg_idx, seg_off, seg_len, event_i, event_v, x, gamma, slot, mass = ctx.saved_tensors
    D_addr, D_x, D_gamma, D_slot, D_mass = torch.ops.ramnet_ops.topk_segment_backward(
        seg_idx, seg_off, seg_len, event_i, event_v,
        x, gamma,
        slot, mass,
        torch.zeros_like(x) if D_y is None else D_y,
        torch.zeros_like(slot) if D_slot is None else D_slot, torch.zeros_like(mass) if D_mass is None else D_mass,
        ctx.w_top_k, ctx.r_top_k,
        ctx.mass_aware, ctx.eps,
    )

    # Internal contract with _addr_backward: read gradients are w.r.t. positive read weights.
    return (D_addr if ctx.needs_input_grad[0] else None,) + (None,) * 4 + (
        D_x.to(x.dtype) if ctx.needs_input_grad[5] else None, D_gamma.to(gamma.dtype) if ctx.needs_input_grad[6] else None,
        D_slot if ctx.needs_input_grad[7] else None, D_mass if ctx.needs_input_grad[8] else None,
    ) + (None,) * 4


torch.library.register_autograd(
    "ramnet_ops::topk_segment_forward",
    _segment_backward,
    setup_context=_segment_setup
)


@torch.library.register_fake("ramnet_ops::topk_segment_forward_")
def _segment_inplace_fake(
        addr_val,  # r (B, H, Kw + Kr, T)

        seg_idx,  # r (S)
        seg_off,  # r (S)
        seg_len,  # r (S)
        event_i,  # r (B * H * T * (Kw + Kr))

        x,  # r (B, H, T, C)
        gamma,  # r (B, H, T)

        slot,  # r/w (B, H, A, C)
        mass,  # r/w (B, H, A)

        w_top_k,
        r_top_k,

        mass_aware,
        eps,
):  # (B, H, T, C)
    return x.new_empty(x.shape, dtype=torch.float32)


@torch.library.register_fake("ramnet_ops::topk_autoregressive_")
def _autoregressive_fake(
        w_addr,  # r (B, H, U, P)
        r_addr,  # r (B, H, U, P)

        x,  # r (B, H, C)
        gamma,  # r (B, H)

        slot,  # r/w (B, H, A, C)
        mass,  # r/w (B, H, A)

        mask,  # r (B), bool, or empty
        embed_num,
        position_ids,  # r (B)

        w_top_k,
        r_top_k,

        w_re_norm,
        r_re_norm,

        mass_aware,
        eps,
):  # (B, H, C)
    return x.new_empty(x.shape)
