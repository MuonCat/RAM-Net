#include "ram_attn_ops.h"
#include "ram_attn_topk_addr.cuh"
#include "ram_attn_segment_parallel.cuh"
#include "ram_attn_topk_autoregressive.cuh"

namespace ramnet_ops {
    namespace {
        using acc_t = torch_api::ScalarTypeToCPPTypeT<acc_dtype>;

        TopKAddrConfig address_config(
            const int64_t B, const int64_t H,
            const int64_t U, const int64_t P, const int64_t A,
            const int64_t T, const int64_t E,
            const int64_t K, const bool re_norm,
            const bool addr_val_neg
        ) {
            return {
                static_cast<int_t>(B), static_cast<int_t>(H),
                static_cast<int_t>(U), static_cast<int_t>(P), static_cast<int_t>(A),
                static_cast<int_t>(T), static_cast<int_t>(E),
                static_cast<int_t>(K), re_norm,
                addr_val_neg
            };
        }

        SegmentConfig get_segment_config(
            const Tensor &x, //    r (B, H, T, C)

            const Tensor &slot, // r (B, H, A, C)

            const int64_t w_top_k,
            const int64_t r_top_k,

            const bool mass_aware,
            const double eps
        ) {
            return {
                static_cast<int_t>(x.size(0)), static_cast<int_t>(x.size(1)),
                static_cast<int_t>(slot.size(2)),
                static_cast<int_t>(x.size(3)), static_cast<int_t>(x.size(2)),
                static_cast<int_t>(w_top_k), static_cast<int_t>(r_top_k),
                mass_aware, static_cast<float>(eps)
            };
        }

        Tensor channel_contiguous(const Tensor &value) {
            // r (..., C)
            return value.stride(-1) == 1 ? value : torch_api::contiguous(value);
        }

        Tensor gather_event_values(
            const Tensor &addr_val, // r (B, H, Kw + Kr, T)
            const Tensor &event_i //   r (B * H * T * (Kw + Kr))
        ) {
            // event_v (B * H * T * (Kw + Kr))
            const auto values = torch_api::reshape(torch_api::transpose(addr_val, -2, -1), {-1});
            return torch_api::index_select(values, 0, torch_api::to(event_i, torch_api::ScalarType::Long));
        }
    }

    TensorPair topk_addr_single_forward(
        const Tensor &addr, //         r (B, H, U, P, T)

        const int64_t embed_num,
        const int64_t A,
        const Tensor &position_ids, // r (B, T)

        const int64_t top_k,
        const bool top_k_re_norm,

        const bool addr_val_neg
    ) {
        const auto cfg = address_config(
            addr.size(0), addr.size(1),
            addr.size(2), addr.size(3), A,
            addr.size(4), embed_num,
            top_k, top_k_re_norm,
            addr_val_neg
        );

        auto idx = torch_api::empty({cfg.B, cfg.H, cfg.top_k, cfg.T}, int_dtype, addr.device()); // w (B, H, K, T)
        auto val = torch_api::empty({cfg.B, cfg.H, cfg.top_k, cfg.T}, acc_dtype, addr.device()); // w (B, H, K, T)

        topk_addr_forward_dispatcher(
            torch_api::contiguous(addr),
            TopKAddr{idx, val},
            torch_api::contiguous(position_ids),
            cfg
        );

        return {idx, val};
    }

    Tensor topk_addr_single_backward(
        const Tensor &idx, //                 r (B, H, K, T)
        const Tensor &val, //                 r (B, H, K, T)

        const std::optional<Tensor> &addr, // r (B, H, U, P, T), optional logits

        const Tensor &D_val, //               r (B, H, K, T)

        const int64_t embed_num,
        const Tensor &position_ids, //        r (B, T)

        const int64_t top_k,

        const bool top_k_re_norm,
        const bool addr_val_neg,

        const torch_api::IntArrayRef shape, // (B, H, U, P, A, T)
        const torch_api::ScalarType dtype
    ) {
        RAMNET_CHECK(shape.size() == 6, "Expected address dimensions (B, H, U, P, A, T).");
        RAMNET_CHECK(top_k_re_norm || addr.has_value(), "Address logits are required when top_k_re_norm is false.");
        const auto B = shape[0], H = shape[1], U = shape[2], P = shape[3], A = shape[4], T = shape[5];
        const auto cfg = address_config(B, H, U, P, A, T, embed_num, top_k, top_k_re_norm, addr_val_neg);


        auto D_addr = top_k_re_norm ? torch_api::empty({B, H, U, P, T}, dtype, val.device()) : torch_api::clone(*addr);
        auto grad = torch_api::to(D_val, acc_dtype); // r (B, H, K, T)
        if (addr_val_neg)
            grad = torch_api::neg(grad);

        topk_addr_backward_dispatcher(
            TopKAddr{idx, val},
            grad,
            D_addr,
            torch_api::contiguous(position_ids),
            cfg
        );

        return D_addr;
    }

    TensorPair topk_addr_forward(
        const Tensor &w_addr, // r (B, H, U, P, T)
        const Tensor &r_addr, // r (B, H, U, P, T)

        const int64_t embed_num,
        const int64_t A,
        const Tensor &position_ids, // r (B, T)

        const int64_t w_top_k,
        const int64_t r_top_k,

        const bool w_re_norm,
        const bool r_re_norm
    ) {
        const auto B = w_addr.size(0), H = w_addr.size(1), U = w_addr.size(2), P = w_addr.size(3), T = w_addr.size(4);
        const auto w_cfg = address_config(B, H, U, P, A, T, embed_num, w_top_k, w_re_norm, false);
        const auto r_cfg = address_config(B, H, U, P, A, T, embed_num, r_top_k, r_re_norm, true);

        auto idx = torch_api::empty({B, H, w_top_k + r_top_k, T}, int_dtype, w_addr.device()); // (B, H, Kw + Kr, T)
        auto val = torch_api::empty({B, H, w_top_k + r_top_k, T}, acc_dtype, w_addr.device()); // (B, H, Kw + Kr, T)

        const auto pos = torch_api::contiguous(position_ids); // r (B, T)

        topk_addr_forward_dispatcher(
            torch_api::contiguous(w_addr),
            TopKAddr{torch_api::narrow(idx, 2, 0, w_top_k), torch_api::narrow(val, 2, 0, w_top_k)},
            pos,
            w_cfg
        );
        topk_addr_forward_dispatcher(
            torch_api::contiguous(r_addr),
            TopKAddr{torch_api::narrow(idx, 2, w_top_k, r_top_k), torch_api::narrow(val, 2, w_top_k, r_top_k)},
            pos,
            r_cfg
        );

        return {idx, val};
    }

    TensorPair topk_addr_backward(
        const Tensor &idx, //                   r (B, H, Kw + Kr, T)
        const Tensor &val, //                   r (B, H, Kw + Kr, T)

        const std::optional<Tensor> &w_addr, // r (B, H, U, P, T), optional logits
        const std::optional<Tensor> &r_addr, // r (B, H, U, P, T), optional logits

        const Tensor &D_val, //                 r (B, H, Kw + Kr, T), read gradients w.r.t. positive weights

        const int64_t embed_num,
        const Tensor &position_ids, //          r (B, T)

        const int64_t w_top_k,
        const int64_t r_top_k,

        const bool w_re_norm,
        const bool r_re_norm,

        const torch_api::IntArrayRef shape, // (B, H, U, P, A, T)
        const torch_api::ScalarType dtype
    ) {
        RAMNET_CHECK(shape.size() == 6, "Expected address dimensions (B, H, U, P, A, T).");
        RAMNET_CHECK(w_re_norm || w_addr.has_value(), "Write logits are required when w_re_norm is false.");
        RAMNET_CHECK(r_re_norm || r_addr.has_value(), "Read logits are required when r_re_norm is false.");
        const auto B = shape[0], H = shape[1], U = shape[2], P = shape[3], A = shape[4], T = shape[5];
        const auto w_cfg = address_config(B, H, U, P, A, T, embed_num, w_top_k, w_re_norm, false);
        const auto r_cfg = address_config(B, H, U, P, A, T, embed_num, r_top_k, r_re_norm, true);

        auto D_w = w_re_norm ? torch_api::empty({B, H, U, P, T}, dtype, val.device()) : torch_api::clone(*w_addr);
        auto D_r = r_re_norm ? torch_api::empty({B, H, U, P, T}, dtype, val.device()) : torch_api::clone(*r_addr);
        const auto grad = torch_api::to(D_val, acc_dtype);
        const auto pos = torch_api::contiguous(position_ids);

        topk_addr_backward_dispatcher(
            TopKAddr{torch_api::narrow(idx, 2, 0, w_top_k), torch_api::narrow(val, 2, 0, w_top_k)},
            torch_api::narrow(grad, 2, 0, w_top_k),
            D_w,
            pos,
            w_cfg
        );
        // Internal addr/segment contract: read gradients are w.r.t. positive read weights.
        topk_addr_backward_dispatcher(
            TopKAddr{torch_api::narrow(idx, 2, w_top_k, r_top_k), torch_api::narrow(val, 2, w_top_k, r_top_k)},
            torch_api::narrow(grad, 2, w_top_k, r_top_k),
            D_r,
            pos,
            r_cfg
        );
        return {D_w, D_r};
    }

    TensorQuad topk_segment_forward(
        const Tensor &addr_val, //  r (B, H, Kw + Kr, T)

        const Tensor &seg_idx, //   r (S)
        const Tensor &seg_off, //   r (S)
        const Tensor &seg_len, //   r (S)
        const Tensor &event_i, //   r (B * H * T * (Kw + Kr))

        const Tensor &x, //         r (B, H, T, C)
        const Tensor &gamma, //     r (B, H, T)

        const Tensor &init_slot, // r (B, H, A, C)
        const Tensor &init_mass, // r (B, H, A)

        const int64_t w_top_k,
        const int64_t r_top_k,

        const bool mass_aware,
        const double eps
    ) {
        const auto cfg = get_segment_config(x, init_slot, w_top_k, r_top_k, mass_aware, eps);

        // gather address value of events
        auto event_v = gather_event_values(addr_val, event_i);

        auto post_slot = torch_api::clone(init_slot);
        auto post_mass = torch_api::clone(init_mass);
        auto y = torch_api::zeros(x.sizes(), acc_dtype, x.device());

        topk_segment_forward_dispatcher(
            SegmentMeta{seg_idx, seg_off, seg_len, event_i, event_v},
            InPack{channel_contiguous(x), gamma},
            State{channel_contiguous(init_slot), init_mass},
            y, State{post_slot, post_mass},
            cfg
        );

        // The auxiliary output (event_v) is saved by autograd without a second gather or event-gradient remapping.
        return {y, post_slot, post_mass, event_v};
    }

    TensorQuint topk_segment_backward(
        const Tensor &seg_idx, //     r (S)
        const Tensor &seg_off, //     r (S)
        const Tensor &seg_len, //     r (S)
        const Tensor &event_i, //     r (B * H * T * (Kw + Kr))
        const Tensor &event_v, //     r (B * H * T * (Kw + Kr))

        const Tensor &x, //           r (B, H, T, C)
        const Tensor &gamma, //       r (B, H, T)

        const Tensor &init_slot, //   r (B, H, A, C)
        const Tensor &init_mass, //   r (B, H, A)

        const Tensor &D_y, //         r (B, H, T, C)
        const Tensor &D_post_slot, // r (B, H, A, C)
        const Tensor &D_post_mass, // r (B, H, A)

        const int64_t w_top_k,
        const int64_t r_top_k,

        const bool mass_aware,
        const double eps
    ) {
        const auto cfg = get_segment_config(x, init_slot, w_top_k, r_top_k, mass_aware, eps);
        const auto device = x.device();

        auto D_slot = torch_api::clone(D_post_slot);
        auto D_mass = torch_api::clone(D_post_mass);
        auto D_addr = torch_api::zeros({cfg.B, cfg.H, w_top_k + r_top_k, cfg.T}, acc_dtype, device);
        auto D_x = torch_api::zeros(x.sizes(), acc_dtype, device);
        auto D_gamma = torch_api::zeros(gamma.sizes(), acc_dtype, device);

        topk_segment_backward_dispatcher(
            SegmentMeta{seg_idx, seg_off, seg_len, event_i, event_v},
            InPack{channel_contiguous(x), gamma},
            State{channel_contiguous(init_slot), init_mass},
            channel_contiguous(torch_api::to(D_y, x.scalar_type())),
            D_addr, InPack{D_x, D_gamma}, State{D_slot, D_mass},
            cfg
        );

        // Pass read gradients w.r.t. positive read weights directly to topk_addr_backward.
        return {D_addr, D_x, D_gamma, D_slot, D_mass};
    }

    Tensor topk_segment_forward_(
        const Tensor &addr_val, // r (B, H, Kw + Kr, T)

        const Tensor &seg_idx, // r (S)
        const Tensor &seg_off, // r (S)
        const Tensor &seg_len, // r (S)
        const Tensor &event_i, // r (B * H * T * (Kw + Kr))

        const Tensor &x, // r (B, H, T, C)
        const Tensor &gamma, // r (B, H, T)

        const Tensor &slot, // r/w (B, H, A, C)
        const Tensor &mass, // r/w (B, H, A)

        const int64_t w_top_k,
        const int64_t r_top_k,

        const bool mass_aware,
        const double eps
    ) {
        const auto cfg = get_segment_config(x, slot, w_top_k, r_top_k, mass_aware, eps);

        // gather address value of events
        const auto event_v = gather_event_values(addr_val, event_i);

        auto y = torch_api::zeros(x.sizes(), acc_dtype, x.device());

        topk_segment_forward_dispatcher(
            SegmentMeta{seg_idx, seg_off, seg_len, event_i, event_v},
            InPack{channel_contiguous(x), gamma},
            State{slot, mass},
            y, State{slot, mass},
            cfg
        );
        return y;
    }

    Tensor topk_autoregressive_(
        const Tensor &w_addr, // r (B, H, U, P)
        const Tensor &r_addr, // r (B, H, U, P)

        const Tensor &x, // r (B, H, C)
        const Tensor &gamma, // r (B, H)

        const Tensor &slot, // r/w (B, H, A, C)
        const Tensor &mass, // r/w (B, H, A)

        const Tensor &mask, // r (B), bool, or empty
        const int64_t embed_num,
        const Tensor &position_ids, // r (B)

        const int64_t w_top_k,
        const int64_t r_top_k,

        const bool w_re_norm,
        const bool r_re_norm,

        const bool mass_aware,
        const double eps
    ) {
        const AutoregressiveConfig cfg{
            static_cast<int_t>(x.size(0)), static_cast<int_t>(x.size(1)),
            static_cast<int_t>(w_addr.size(2)), static_cast<int_t>(w_addr.size(3)), static_cast<int_t>(slot.size(2)),
            static_cast<int_t>(x.size(2)), static_cast<int_t>(embed_num),
            static_cast<int_t>(w_top_k), static_cast<int_t>(r_top_k),
            w_re_norm, r_re_norm,
            mass_aware, static_cast<acc_t>(eps)
        };

        auto y = torch_api::zeros(x.sizes(), x.scalar_type(), x.device()); // w (B, H, C)

        ram_attn_topk_autoregressive_dispatcher(
            Address{torch_api::contiguous(w_addr), torch_api::contiguous(r_addr)},
            InPack{torch_api::contiguous(x), torch_api::contiguous(gamma)},
            State{slot, mass},
            y,
            torch_api::contiguous(mask), torch_api::contiguous(position_ids),
            cfg
        );
        return y;
    }
}
