#pragma once
#include "torch_compat.h"

using int_t = int32_t;
constexpr torch_api::ScalarType int_dtype = torch_api::ScalarType::Int;
constexpr torch_api::ScalarType acc_dtype = torch_api::ScalarType::Float;
constexpr int_t warp_size = 32;


struct TaskConfig {
    int_t B, H, U, P, A, C, T, E; // batch, heads, order, radix, slots, channels, timesteps, CAPE heads

    int_t w_top_k, r_top_k;
    bool w_top_k_re_norm, r_top_k_re_norm;

    bool mass_aware;
    float eps;

    constexpr TaskConfig(
        const int_t B_, const int_t H_,
        const int_t U_, const int_t P_, const int_t A_,
        const int_t C_,
        const int_t T_,
        const int_t E_,
        const int_t w_top_k_, const int_t r_top_k_,
        const bool w_top_k_re_norm_, const bool r_top_k_re_norm_,
        const bool mass_aware_,
        const float eps_
    )
        : B(B_), H(H_), U(U_), P(P_), A(A_), C(C_),
          T(T_), E(E_),
          w_top_k(w_top_k_), r_top_k(r_top_k_),
          w_top_k_re_norm(w_top_k_re_norm_), r_top_k_re_norm(r_top_k_re_norm_),
          mass_aware(mass_aware_), eps(eps_) {
    }
};

struct State {
    torch_api::Tensor slot;
    torch_api::Tensor mass;
};

struct InPack {
    torch_api::Tensor x;
    torch_api::Tensor gamma;
};
