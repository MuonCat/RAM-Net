#pragma once
#include "torch_compat.h"
#include <optional>
#include <tuple>

namespace ramnet_ops {
    using Tensor = torch_api::Tensor;
    using TensorPair = std::tuple<Tensor, Tensor>;
    using TensorQuad = std::tuple<Tensor, Tensor, Tensor, Tensor>;
    using TensorQuint = std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>;

    TensorPair topk_addr_single_forward(
        const Tensor &addr, //         r (B, H, U, P, T)

        int64_t embed_num,
        int64_t A,

        const Tensor &position_ids, // r (B, T)

        int64_t top_k,
        bool top_k_re_norm,

        bool addr_val_neg
    ); // w: idx/val (B, H, K, T)

    Tensor topk_addr_single_backward(
        const Tensor &idx, //                 r (B, H, K, T)
        const Tensor &val, //                 r (B, H, K, T)

        const std::optional<Tensor> &addr, // r (B, H, U, P, T), optional logits

        const Tensor &D_val, //               r (B, H, K, T)

        int64_t embed_num,
        const Tensor &position_ids, //        r (B, T)

        int64_t top_k,
        bool top_k_re_norm,

        bool addr_val_neg,

        torch_api::IntArrayRef shape, // (B, H, U, P, A, T)
        torch_api::ScalarType dtype
    ); // w: D_addr (B, H, U, P, T)

    TensorPair topk_addr_forward(
        const Tensor &w_addr, //       r (B, H, U, P, T)
        const Tensor &r_addr, //       r (B, H, U, P, T)

        int64_t embed_num,
        int64_t A,

        const Tensor &position_ids, // r (B, T)

        int64_t w_top_k,
        int64_t r_top_k,

        bool w_re_norm,
        bool r_re_norm
    ); // w: idx/val (B, H, Kw + Kr, T)

    TensorPair topk_addr_backward(
        const Tensor &idx, //                   r (B, H, Kw + Kr, T)
        const Tensor &val, //                   r (B, H, Kw + Kr, T)

        const std::optional<Tensor> &w_addr, // r (B, H, U, P, T), optional logits
        const std::optional<Tensor> &r_addr, // r (B, H, U, P, T), optional logits

        const Tensor &D_val, //                 r (B, H, Kw + Kr, T), read gradients w.r.t. positive weights

        int64_t embed_num,
        const Tensor &position_ids, //          r (B, T)

        int64_t w_top_k,
        int64_t r_top_k,

        bool w_re_norm,
        bool r_re_norm,

        torch_api::IntArrayRef shape, // (B, H, U, P, A, T)
        torch_api::ScalarType dtype
    ); // w: D_w/D_r (B, H, U, P, T)

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

        int64_t w_top_k,
        int64_t r_top_k,

        bool mass_aware,
        double eps
    ); // w: y (B, H, T, C), slot (B, H, A, C), mass (B, H, A), event_v (B * H * T * (Kw + Kr)), replay cache

    TensorQuint topk_segment_backward(
        const Tensor &seg_idx, //   r (S)
        const Tensor &seg_off, //   r (S)
        const Tensor &seg_len, //   r (S)
        const Tensor &event_i, //   r (B * H * T * (Kw + Kr))
        const Tensor &event_v, //   r (B * H * T * (Kw + Kr))

        const Tensor &x, //         r (B, H, T, C)
        const Tensor &gamma, //     r (B, H, T)

        const Tensor &init_slot, // r (B, H, A, C)
        const Tensor &init_mass, // r (B, H, A)

        const Tensor &D_y, //       r (B, H, T, C)
        const Tensor &D_post_slot, // r (B, H, A, C)
        const Tensor &D_post_mass, // r (B, H, A)

        int64_t w_top_k,
        int64_t r_top_k,

        bool mass_aware,
        double eps
    ); // w: D_addr (B, H, Kw + Kr, T), D_x (B, H, T, C), D_gamma (B, H, T), D_slot (B, H, A, C), D_mass (B, H, A)
    // The read part of D_addr is w.r.t. positive read weights.

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

        int64_t w_top_k,
        int64_t r_top_k,

        bool mass_aware,
        double eps
    ); // w: y (B, H, T, C)

    Tensor topk_autoregressive_(
        const Tensor &w_addr, //       r  (B, H, U, P)
        const Tensor &r_addr, //       r  (B, H, U, P)

        const Tensor &x, //            r  (B, H, C)
        const Tensor &gamma, //        r  (B, H)

        const Tensor &slot, //        r/w (B, H, A, C)
        const Tensor &mass, //        r/w (B, H, A)

        const Tensor &mask, //         r  (B), bool, or empty
        int64_t embed_num,
        const Tensor &position_ids, // r  (B)

        int64_t w_top_k,
        int64_t r_top_k,

        bool w_re_norm,
        bool r_re_norm,

        bool mass_aware,
        double eps
    ); // w: y (B, H, C)
}
