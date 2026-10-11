#pragma once
#include "ram_attn.cuh"

struct SegmentConfig {
    int_t B, H, A, C, T;

    int_t w_top_k, r_top_k;

    bool mass_aware;
    float eps;

    constexpr SegmentConfig(
        const int_t B_,
        const int_t H_,
        const int_t A_,
        const int_t C_,
        const int_t T_,

        const int_t w_top_k_,
        const int_t r_top_k_,

        const bool mass_aware_,
        const float eps_
    )
        : B(B_), H(H_), A(A_), C(C_), T(T_),
          w_top_k(w_top_k_), r_top_k(r_top_k_),
          mass_aware(mass_aware_), eps(eps_) {
    }
};

struct SegmentMeta {
    torch_api::Tensor seg_idx; // (S)
    torch_api::Tensor seg_off; // (S)
    torch_api::Tensor seg_len; // (S)
    torch_api::Tensor event_i; // (B*H*T*(Kw+Kr))
    torch_api::Tensor event_v; // (B*H*T*(Kw+Kr))

    explicit SegmentMeta(
        const torch_api::Tensor &seg_idx_,
        const torch_api::Tensor &seg_off_,
        const torch_api::Tensor &seg_len_,
        const torch_api::Tensor &event_i_,
        const torch_api::Tensor &event_v_
    )
        : seg_idx(seg_idx_), seg_off(seg_off_), seg_len(seg_len_), event_i(event_i_), event_v(event_v_) {
    }
};

void topk_segment_forward_dispatcher(
    const SegmentMeta &seg_meta, // r

    const InPack &inputs, //        r  (B, H, T, [C])

    const State &init_state, //     r  (B, H, A, [C])

    const torch_api::Tensor &ys, //    r/w (B, H, T, C)
    const State &post_state, //     w  (B, H, A, [C])

    const SegmentConfig &cfg
);

void topk_segment_backward_dispatcher(
    const SegmentMeta &seg_meta, //     r

    const InPack &inputs, //            r  (B, H, T, [C])

    const State &init_state, //         r  (B, H, A, [C])

    const torch_api::Tensor &D_y, //        r  (B, H, T, C)

    const torch_api::Tensor &D_addr_val, // w  (B, H, Kw + Kr, T)
    const InPack &D_inputs, //         r/w (B, H, T, [C])
    const State &D_state, //           r/w (B, H, A, [C])

    const SegmentConfig &cfg
);
