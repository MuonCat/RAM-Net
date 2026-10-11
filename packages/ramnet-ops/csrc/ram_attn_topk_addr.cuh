#pragma once
#include "ram_attn.cuh"

struct TopKAddrConfig {
    int_t B, H, U, P, A, T, E;

    int_t top_k;
    bool top_k_re_norm;

    bool addr_val_neg;

    constexpr TopKAddrConfig(
        const int_t B_, const int_t H_,
        const int_t U_, const int_t P_, const int_t A_,
        const int_t T_, const int_t E_,
        const int_t top_k_, const bool top_k_re_norm_,
        const bool addr_val_neg_
    )
        : B(B_), H(H_), U(U_), P(P_), A(A_), T(T_), E(E_),
          top_k(top_k_), top_k_re_norm(top_k_re_norm_),
          addr_val_neg(addr_val_neg_) {
    }
};

struct TopKAddr {
    torch_api::Tensor idx; // (B, H, K, T)
    torch_api::Tensor val; // (B, H, K, T)
};

void topk_addr_forward_dispatcher(
    const torch_api::Tensor &addr, //         r (B, H, U, P, T)

    const TopKAddr &top_k_addr, //        w (B, H, K, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
);

void topk_addr_backward_dispatcher(
    const TopKAddr &top_k_addr, //        r (B, H, K, T)

    const torch_api::Tensor &D_addr_val, //   r (B, H, K, T)

    const torch_api::Tensor &D_addr, //       w (B, H, U, P, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
);
