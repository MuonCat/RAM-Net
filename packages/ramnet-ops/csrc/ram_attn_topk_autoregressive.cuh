#pragma once
#include "ram_attn.cuh"

struct AutoregressiveConfig {
    int_t B, H, U, P, A, C, E;

    int_t w_top_k, r_top_k;
    bool w_top_k_re_norm, r_top_k_re_norm;

    bool mass_aware;
    float eps;

    constexpr AutoregressiveConfig(
        const int_t B_,
        const int_t H_,
        const int_t U_,
        const int_t P_,
        const int_t A_,
        const int_t C_,

        const int_t E_,

        const int_t w_top_k_,
        const int_t r_top_k_,

        const bool w_top_k_re_norm_,
        const bool r_top_k_re_norm_,

        const bool mass_aware_,
        const float eps_
    )
        : B(B_), H(H_), U(U_), P(P_), A(A_), C(C_), E(E_),
          w_top_k(w_top_k_), r_top_k(r_top_k_),
          w_top_k_re_norm(w_top_k_re_norm_), r_top_k_re_norm(r_top_k_re_norm_),
          mass_aware(mass_aware_), eps(eps_) {
    }
};

struct Address {
    torch_api::Tensor w;
    torch_api::Tensor r;
};


void ram_attn_topk_autoregressive_dispatcher(
    const Address &addr, //        r  (B, H, U, P)

    const InPack &inputs, //       r  (B, H, [C])

    const State &state, //        r/w (B, H, A, [C])

    const torch_api::Tensor &y, //     w  (B, H, C)

    const torch_api::Tensor &mask, //   r (B), bool, or empty
    const torch_api::Tensor &position_ids, // r (B)

    const AutoregressiveConfig &cfg
);
