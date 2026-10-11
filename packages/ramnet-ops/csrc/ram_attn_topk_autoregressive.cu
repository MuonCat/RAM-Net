/*
 * RAM Attention kernel type:
 * - Autoregressive inference (T=1), Top-K sparse path.
 *
 * Parallel decomposition:
 * - Global space: (B, H).
 * - Grid/warp mapping: one warp owns one (b, h) sample.
 *
 * Warp/lane boundary:
 * - Warp: coordinates top-k address handling and slot-level read/write flow for that (b, h).
 * - Lane: parallelizes channel work (C dimension) and participates in warp collectives for top-k.
 *
 * Structure hint:
 * - Address handling + state update + readout are fused in one kernel for single-step decode.
 */

#include "ram_attn_topk_autoregressive.cuh"
#include "cuda_utils.cuh"
#include "vector_ops.cuh"

template<typename T, bool ReadOnly = false>
struct AddressPtr {
    using ptr_t = std::conditional_t<ReadOnly, const T *, T *>;
    ptr_t w;
    ptr_t r;

    explicit AddressPtr(const Address &a)
        : w(torch_api::data_ptr<T, ReadOnly>(a.w)), r(torch_api::data_ptr<T, ReadOnly>(a.r)) {
    }
};

inline void check_address(const Address &addr, const torch_api::ScalarType dtype, const torch_api::IntArrayRef sizes) {
    check_tensor("w", addr.w, dtype, CheckContiguity::Whole, sizes);
    check_tensor("r", addr.r, dtype, CheckContiguity::Whole, sizes);
}

template<typename scalar_t, typename acc_t, int_t warp_per_block, int_t c_loop, int_t max_k, int_t max_p, int_t max_u, bool MASS_AWARE, bool native_vec = false>
__global__ void ram_attn_topk_autoregressive_kernel(
    const AddressPtr<scalar_t, true> addr_ptr, //    r  (B, H, U, P)

    const InPackPtr<scalar_t, acc_t> input_ptr, //   r  (B, H, [C])

    const StatePtr<acc_t> state_ptr, //             r/w (B, H, A, [C])

    scalar_t *const global_y_ptr, //                 w  (B, H, C)

    const bool *const __restrict__ mask, //          r  (B), optional
    const int_t *const __restrict__ position_ids, // r  (B)

    const AutoregressiveConfig cfg
) {
    using acc_vec = Vec<acc_t, c_loop, native_vec>;
    using scalar_vec = Vec<scalar_t, c_loop, native_vec>;
    constexpr unsigned full_mask = ~0u;
    static_assert(warp_per_block <= warp_size);
    static_assert(max_u <= warp_size / 2);

    const int_t thread_idx = threadIdx.x;
    const int_t lane_idx = thread_idx % warp_size;
    const int_t warp_idx = thread_idx / warp_size;
    const int_t bh_idx = blockIdx.x * warp_per_block + warp_idx;
    const int_t b_idx = bh_idx / cfg.H;
    const int_t h_idx = bh_idx % cfg.H;

    if (bh_idx >= cfg.B * cfg.H || (mask != nullptr && !mask[b_idx])) return;

    const scalar_t *const __restrict__ local_w_ptr = addr_ptr.w + bh_idx * cfg.U * cfg.P; // (U, P)
    const scalar_t *const __restrict__ local_r_ptr = addr_ptr.r + bh_idx * cfg.U * cfg.P; // (U, P)
    const scalar_t *const __restrict__ local_x_ptr = input_ptr.x + bh_idx * cfg.C; // (C)
    const scalar_t *const __restrict__ local_gamma_ptr = input_ptr.gamma + bh_idx; // ()
    acc_t *const __restrict__ local_slot_ptr = state_ptr.slot + bh_idx * cfg.A * cfg.C; // (A, C)
    acc_t *const __restrict__ local_mass_ptr = state_ptr.mass + bh_idx * cfg.A; // (A)
    scalar_t *const __restrict__ local_y_ptr = global_y_ptr + bh_idx * cfg.C; // (C)
    const acc_t gamma = type_cast<scalar_t, acc_t>(*local_gamma_ptr);

    __shared__ acc_t block_w_cache[warp_per_block][max_p * max_u];
    __shared__ acc_t block_r_cache[warp_per_block][max_p * max_u];
    auto &warp_w_cache = block_w_cache[warp_idx];
    auto &warp_r_cache = block_r_cache[warp_idx];
    // read (U, P) into shared memory
    for (int_t i = lane_idx; i < cfg.U * cfg.P; i += warp_size) {
        warp_w_cache[i] = type_cast<scalar_t, acc_t>(local_w_ptr[i]);
        warp_r_cache[i] = type_cast<scalar_t, acc_t>(local_r_ptr[i]);
    }
    __syncwarp();

    // load the value local
    const int_t u_idx = lane_idx % (warp_size / 2);
    const acc_t *const addr_cache = lane_idx < warp_size / 2 ? warp_w_cache : warp_r_cache;
    const bool addr_re_norm = lane_idx < warp_size / 2 ? cfg.w_top_k_re_norm : cfg.r_top_k_re_norm;


    __shared__ IdxValList<acc_t, max_k, warp_per_block * warp_size> local_topk;
    local_topk.init(thread_idx);

    // calculate the top-k of current u
    do {
        IdxValList<acc_t, max_p> local_input;
        local_input.init();
#pragma unroll
        for (int_t p = 0; p < max_p; ++p)
            if (p < cfg.P && u_idx < cfg.U)
                local_input.set(p, p, addr_cache[u_idx * cfg.P + p]);
        // sort local
        local_input.odd_even_sort();

        // calculate log-softmax
        if (!addr_re_norm)
            local_input.topk_log_softmax(cfg.P);

        // copy top-k from inputs
        local_topk.copy_from(local_input, thread_idx);
    } while (false);

    // merge cross lane
    for (int_t offset = 1, i = 0, stride = cfg.P; offset < warp_size / 2; offset <<= 1, ++i, stride *= stride) {
        IdxValList<acc_t, max_k> peer_topk;
        peer_topk.init();
#pragma unroll topk_unroll
        for (int_t k = 0; k < max_k; ++k)
            peer_topk.set(
                k,
                __shfl_down_sync(full_mask, local_topk.get_idx(k, thread_idx), offset),
                __shfl_down_sync(full_mask, local_topk.get_val(k, thread_idx), offset)
            );

        if (u_idx + offset < cfg.U)
            local_topk.sum_merge_from(peer_topk, stride, thread_idx);
    }

    // calculate log-softmax
    if (addr_re_norm)
        local_topk.topk_log_softmax(lane_idx < warp_size / 2 ? cfg.w_top_k : cfg.r_top_k, thread_idx);

    // write
    const int_t a_offset = h_idx < cfg.E ? position_ids[b_idx] % cfg.A : 0;
#pragma unroll
    for (int_t k = 0; k < max_k; ++k)
        if (k < cfg.w_top_k) {
            const acc_t w = __shfl_sync(full_mask, get_exp<acc_t>(local_topk.get_val(k, thread_idx)), 0);
            if (w == 0) continue;

            const int_t idx = (__shfl_sync(full_mask, local_topk.get_idx(k, thread_idx), 0) + cfg.A - a_offset) % cfg.A;

            acc_vec slot(local_slot_ptr + idx * cfg.C, cfg.C, lane_idx);
            acc_vec x_val = scalar_vec{local_x_ptr, cfg.C, lane_idx}.template cast<acc_t>();
            const auto [gain, new_mass] = get_gain<acc_t>(w, gamma, local_mass_ptr[idx], cfg.eps);
            slot.lerp_(gain, x_val);
            slot.store(local_slot_ptr + idx * cfg.C, cfg.C, lane_idx);

            if (lane_idx == 0)
                local_mass_ptr[idx] = new_mass;
        }

    __syncwarp();

    // read
    acc_vec out{0};
    if constexpr (MASS_AWARE) {
        acc_t sum_r = 0, sum_rm = 0;
#pragma unroll
        for (int_t k = max_k - 1; k >= 0; --k)
            if (k < cfg.r_top_k) {
                const acc_t r = __shfl_sync(full_mask, get_exp<acc_t>(local_topk.get_val(k, thread_idx)), warp_size / 2);
                if (r == 0) continue;

                const int_t idx = (__shfl_sync(full_mask, local_topk.get_idx(k, thread_idx), warp_size / 2) + cfg.A - a_offset) % cfg.A;

                acc_vec slot(local_slot_ptr + idx * cfg.C, cfg.C, lane_idx);
                const acc_t rm = r * local_mass_ptr[idx];
                sum_r += r, sum_rm += rm;
                slot *= rm;
                out += slot;
            }
        out *= sum_r / (sum_rm + cfg.eps);
    }
    else {
#pragma unroll
        for (int_t k = max_k - 1; k >= 0; --k)
            if (k < cfg.r_top_k) {
                const acc_t r = __shfl_sync(full_mask, get_exp<acc_t>(local_topk.get_val(k, thread_idx)), warp_size / 2);
                if (r == 0) continue;

                const int_t idx = (__shfl_sync(full_mask, local_topk.get_idx(k, thread_idx), warp_size / 2) + cfg.A - a_offset) % cfg.A;

                acc_vec slot(local_slot_ptr + idx * cfg.C, cfg.C, lane_idx);
                slot *= r;
                out += slot;
            }
    }
    out.template cast<scalar_t>().store(local_y_ptr, cfg.C, lane_idx);
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t warp_per_block, int_t c_loop, int_t max_k, int_t max_p, int_t max_u, bool native_vec = false>
inline void ram_attn_topk_autoregressive_kernel_launcher(
    const Address &addr, //               r  (B, H, U, P)

    const InPack &inputs, //              r  (B, H, [C])

    const State &state, //               r/w (B, H, A, [C])

    const torch_api::Tensor &y, //            w  (B, H, C)

    const torch_api::Tensor &mask, //         r  (B), bool, or empty
    const torch_api::Tensor &position_ids, // r  (B)

    const AutoregressiveConfig &cfg
) {
    using scalar_t = torch_api::ScalarTypeToCPPTypeT<scalar_dtype>;
    using acc_t = torch_api::ScalarTypeToCPPTypeT<acc_dtype>;

    RAMNET_CHECK(cfg.C <= c_loop * warp_size, "Expect C <= ", c_loop * warp_size, " but got ", cfg.C);
    RAMNET_CHECK(0 < cfg.w_top_k && cfg.w_top_k <= max_k, "Expect 0 < w_top_k <= ", max_k, " but got ", cfg.w_top_k);
    RAMNET_CHECK(0 < cfg.r_top_k && cfg.r_top_k <= max_k, "Expect 0 < r_top_k <= ", max_k, " but got ", cfg.r_top_k);
    RAMNET_CHECK(cfg.U <= max_u, "Expect U <= ", max_u, " but got ", cfg.U);
    RAMNET_CHECK(cfg.P <= max_p, "Expect P <= ", max_p, " but got ", cfg.P);
    check_address(addr, scalar_dtype, {cfg.B, cfg.H, cfg.U, cfg.P});
    check_tensor("x", inputs.x, scalar_dtype, CheckContiguity::Whole, {cfg.B, cfg.H, cfg.C});
    check_tensor("gamma", inputs.gamma, scalar_dtype, CheckContiguity::Whole, {cfg.B, cfg.H});
    check_tensor("position_ids", position_ids, int_dtype, CheckContiguity::Whole, {cfg.B});
    if (mask.numel() > 0)
        check_tensor("mask", mask, torch_api::ScalarType::Bool, CheckContiguity::Whole, {cfg.B});
    check_tensor("slot", state.slot, acc_dtype, CheckContiguity::Whole, {cfg.B, cfg.H, cfg.A, cfg.C});
    check_tensor("mass", state.mass, acc_dtype, CheckContiguity::Whole, {cfg.B, cfg.H, cfg.A});
    check_tensor("y", y, scalar_dtype, CheckContiguity::Whole, {cfg.B, cfg.H, cfg.C});


    const auto blocks = cdiv(cfg.B * cfg.H, warp_per_block);

    const auto device = inputs.x.device();
    torch_api::CUDAGuard device_guard(device);
    const auto stream = torch_api::current_cuda_stream(device);
    const torch_api::CUDAStreamGuard stream_guard(stream, device);

    const auto kernel = cfg.mass_aware
                            ? ram_attn_topk_autoregressive_kernel<scalar_t, acc_t, warp_per_block, c_loop, max_k, max_p, max_u, true, native_vec>
                            : ram_attn_topk_autoregressive_kernel<scalar_t, acc_t, warp_per_block, c_loop, max_k, max_p, max_u, false, native_vec>;
    kernel<<<blocks, warp_per_block * warp_size, 0, stream>>>(
        AddressPtr<scalar_t, true>{addr},
        InPackPtr<scalar_t, acc_t>(inputs),
        StatePtr<acc_t>{state},
        torch_api::data_ptr<scalar_t>(y),
        mask.numel() > 0 ? torch_api::data_ptr<bool, true>(mask) : nullptr, torch_api::data_ptr<int_t, true>(position_ids),
        cfg
    );
    RAMNET_CUDA_KERNEL_LAUNCH_CHECK();
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t warp_per_block, int_t max_k, int_t max_p, int_t max_u>
inline void ram_attn_topk_autoregressive_chs_dispatcher(
    const Address &addr, //               r  (B, H, U, P)

    const InPack &inputs, //              r  (B, H, [C])

    const State &state, //               r/w (B, H, A, [C])

    const torch_api::Tensor &y, //            w  (B, H, C)

    const torch_api::Tensor &mask, //         r  (B), bool, or empty
    const torch_api::Tensor &position_ids, // r  (B)

    const AutoregressiveConfig &cfg
) {
    constexpr static struct {
        bool (*check)(int_t);

        decltype(&ram_attn_topk_autoregressive_kernel_launcher<scalar_dtype, acc_dtype, warp_per_block, 1, max_k, max_p, max_u>) func;
    } kernels[] = {
        {[](const int_t c) -> bool { return c <= 32; }, ram_attn_topk_autoregressive_kernel_launcher<scalar_dtype, acc_dtype, warp_per_block, 1, max_k, max_p, max_u>},
        {[](const int_t c) -> bool { return c <= 64 && c % 2 == 0; }, ram_attn_topk_autoregressive_kernel_launcher<scalar_dtype, acc_dtype, warp_per_block, 2, max_k, max_p, max_u, true>},
        // {[](const int_t c) -> bool { return c <= 64; }, ram_attn_topk_autoregressive_kernel_launcher<scalar_dtype, acc_dtype, warp_per_block, 2, max_k, max_p, max_u>},
        {[](const int_t c) -> bool { return c <= 128; }, ram_attn_topk_autoregressive_kernel_launcher<scalar_dtype, acc_dtype, warp_per_block, 4, max_k, max_p, max_u>},

    };

    for (const auto &[check, kernel]: kernels)
        if (check(cfg.C)) {
            kernel(
                addr,
                inputs,
                state,
                y,
                mask, position_ids,
                cfg
            );
            return;
        }
    RAMNET_CHECK(false, "Unsupported size: C=", cfg.C);
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype>
inline void ram_attn_topk_autoregressive_size_dispatcher(
    const Address &addr, //               r  (B, H, U, P)

    const InPack &inputs, //              r  (B, H, [C])

    const State &state, //               r/w (B, H, A, [C])

    const torch_api::Tensor &y, //            w  (B, H, C)

    const torch_api::Tensor &mask, //         r  (B), bool, or empty
    const torch_api::Tensor &position_ids, // r  (B)

    const AutoregressiveConfig &cfg
) {
    constexpr static struct {
        bool (*check)(int_t, int_t, int_t);

        decltype(&ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 4, 4, 8>) func;
    } kernels[] = {
        // {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 4 && p <= 4 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 4, 4, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 8 && p <= 4 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 8, 4, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 16 && p <= 4 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 16, 4, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 32 && p <= 4 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 32, 4, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 64 && p <= 4 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 1, 64, 4, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 128 && p <= 4 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 1, 128, 4, 8>},

        // {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 4 && p <= 8 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 4, 8, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 8 && p <= 8 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 8, 8, 8>},
        // {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 16 && p <= 8 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 16, 8, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 32 && p <= 8 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 2, 32, 8, 8>},
        // {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 64 && p <= 8 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 1, 64, 8, 8>},
        {[](const int_t k, const int_t u, const int_t p) -> bool { return k <= 128 && p <= 8 && u <= 8; }, ram_attn_topk_autoregressive_chs_dispatcher<scalar_dtype, acc_dtype, 1, 128, 8, 8>},
    };

    for (const auto &[check, kernel]: kernels)
        if (check(max(cfg.w_top_k, cfg.r_top_k), cfg.U, cfg.P)) {
            kernel(
                addr,
                inputs,
                state,
                y,
                mask, position_ids,
                cfg
            );
            return;
        }
    RAMNET_CHECK(false, "Unsupported size: w_top-k=", cfg.w_top_k, " r_top-k=", cfg.r_top_k, ", U=", cfg.U, ", P=", cfg.P);
}


void ram_attn_topk_autoregressive_dispatcher(
    const Address &addr, //               r  (B, H, U, P)

    const InPack &inputs, //              r  (B, H, [C])

    const State &state, //               r/w (B, H, A, [C])

    const torch_api::Tensor &y, //            w  (B, H, C)

    const torch_api::Tensor &mask, //         r  (B), bool, or empty
    const torch_api::Tensor &position_ids, // r  (B)

    const AutoregressiveConfig &cfg
) {
    decltype(&ram_attn_topk_autoregressive_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float>) kernel = nullptr;
    // ReSharper disable once CppIncompleteSwitchStatement
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (inputs.x.scalar_type()) {
        case torch_api::ScalarType::Half:
            kernel = ram_attn_topk_autoregressive_size_dispatcher<torch_api::ScalarType::Half, torch_api::ScalarType::Float>;
            break;
        case torch_api::ScalarType::BFloat16:
            kernel = ram_attn_topk_autoregressive_size_dispatcher<torch_api::ScalarType::BFloat16, torch_api::ScalarType::Float>;
            break;
        case torch_api::ScalarType::Float:
            kernel = ram_attn_topk_autoregressive_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float>;
            break;
    }

    RAMNET_CHECK(kernel!=nullptr, "Unsupported scalar type: ", inputs.x.scalar_type());
    kernel(
        addr,
        inputs,
        state,
        y,
        mask, position_ids,
        cfg
    );
}
