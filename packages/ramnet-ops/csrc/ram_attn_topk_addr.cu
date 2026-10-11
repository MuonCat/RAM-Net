/*
 * RAM Attention helper kernel type:
 * - Fuses address decoding (across U, P dimensions) and Top-K selection.
 * - Builds sparse events (addr_idx/addr_val) from factorized addresses.
 * - Handles optional CAPE offsets and re-normalization after Top-K.
 *
 * Parallel decomposition:
 * - Global space: (B, H, T).
 * - Grid/warp mapping: each warp processes one (b, h) tile over T.
 * - Lane mapping: each lane corresponds to one timestep inside the warp-sized T tile.
 *
 * Warp/lane boundary:
 * - Warp: manages one top-k address list for its (b, h, t-tile) region.
 * - Lane: performs per-timestep top-k selection work and writes one column of outputs.
 */

#include "ram_attn_topk_addr.cuh"
#include "cuda_utils.cuh"

struct TopKAddrStride {
    int64_t b, h, k, t;

    explicit TopKAddrStride(const torch_api::Tensor &tensor)
        : b(tensor.stride(0)), h(tensor.stride(1)), k(tensor.stride(2)), t(tensor.stride(3)) {
    }

    __host__ __device__ __forceinline__
    int64_t offset(const int_t b_idx, const int_t h_idx, const int_t k_idx, const int_t t_idx) const {
        return b_idx * b + h_idx * h + k_idx * k + t_idx * t;
    }
};

template<typename acc_t, bool ReadOnly = false>
struct TopKAddrPtr {
    std::conditional_t<ReadOnly, const int_t *, int_t *> idx;
    std::conditional_t<ReadOnly, const acc_t *, acc_t *> val;
    TopKAddrStride idx_stride, val_stride;

    explicit TopKAddrPtr(const TopKAddr &a)
        : idx(torch_api::data_ptr<int_t, ReadOnly>(a.idx)), val(torch_api::data_ptr<acc_t, ReadOnly>(a.val)),
          idx_stride(a.idx), val_stride(a.val) {
    }
};


inline void check_top_k_addr(const TopKAddr &top_k_addr, const torch_api::ScalarType acc_dtype, const TopKAddrConfig &cfg) {
    check_tensor("addr_idx", top_k_addr.idx, int_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.top_k, cfg.T});
    check_tensor("addr_val", top_k_addr.val, acc_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.top_k, cfg.T});
}

/* Select Top-K addresses and encode their weights with the requested sign. */
template<typename scalar_t, typename acc_t, int_t max_k, int_t max_p, int_t warp_per_block>
__global__ void topk_addr_forward_kernel(
    const scalar_t *const __restrict__ addr_ptr, //  r (B, H, U, P, T)

    const TopKAddrPtr<acc_t> top_k_addr_ptr, //      w (B, H, K, T)

    const int_t *const __restrict__ position_ids, // r (B, T)

    const TopKAddrConfig cfg
) {
    const int_t thread_idx = threadIdx.x;
    const int_t warp_idx = thread_idx / warp_size;
    const int_t lane_idx = thread_idx % warp_size;
    const int_t t_blocks = cfg.T / warp_size;
    const int_t global_warp_idx = blockIdx.x * warp_per_block + warp_idx;
    const int_t bh_idx = global_warp_idx / t_blocks;
    const int_t t_offs = global_warp_idx % t_blocks * warp_size;
    const int_t t_idx = t_offs + lane_idx;
    const int_t b_idx = bh_idx / cfg.H;
    const int_t h_idx = bh_idx % cfg.H;

    if (bh_idx >= cfg.B * cfg.H) return;

    const scalar_t *const __restrict__ local_addr_ptr = addr_ptr + bh_idx * cfg.U * cfg.P * cfg.T; // (U, P, T)

    /* get the top-k select */
    __shared__ IdxValList<acc_t, max_k, warp_per_block * warp_size> main;
    main.init(thread_idx);

    for (int_t u = 0, stride = 1; u < cfg.U; ++u, stride *= cfg.P) {
        IdxValList<acc_t, max_p> peer; // local sub-vector (address digit selection)
        peer.init();

        /* load the sub-vector and sort */
#pragma unroll
        for (int_t p = 0; p < max_p; ++p)
            if (p < cfg.P)
                peer.set(p, p, type_cast<scalar_t, acc_t>(local_addr_ptr[(u * cfg.P + p) * cfg.T + t_idx]));
        peer.odd_even_sort();

        if (!cfg.top_k_re_norm)
            peer.topk_log_softmax(cfg.P); // log-softmax on sub-vector

        /* merge new sub-vector into the main Top-K */
        if (u == 0)
            main.copy_from(peer, thread_idx);
        else
            main.sum_merge_from(peer, stride, thread_idx);
    }

    /* re-norm the selected Top-K in log domain */
    if (cfg.top_k_re_norm)
        main.topk_log_softmax(cfg.top_k, thread_idx);

    /* CAPE offset is time-based for the first E heads; batch-head id is folded into segment address. */
    int_t *const __restrict__ local_idx_ptr = top_k_addr_ptr.idx + top_k_addr_ptr.idx_stride.offset(b_idx, h_idx, 0, t_idx);
    acc_t *const __restrict__ local_val_ptr = top_k_addr_ptr.val + top_k_addr_ptr.val_stride.offset(b_idx, h_idx, 0, t_idx);
    const int_t a_offset = h_idx < cfg.E ? position_ids[b_idx * cfg.T + t_idx] % cfg.A : 0;
    const acc_t addr_val_sign = cfg.addr_val_neg ? -1 : +1;
#pragma unroll
    for (int_t k = 0; k < max_k; ++k)
        if (k < cfg.top_k) {
            local_idx_ptr[k * top_k_addr_ptr.idx_stride.k] = bh_idx * cfg.A + (main.get_idx(k, thread_idx) + cfg.A - a_offset) % cfg.A; // unique address index (bh_idx * cfg.a + a_idx)
            local_val_ptr[k * top_k_addr_ptr.val_stride.k] = addr_val_sign * get_exp<acc_t>(main.get_val(k, thread_idx)); // Encode the positive weight with the requested sign.
        }
}

template<typename scalar_t, typename acc_t, int_t max_k, int_t max_p, int_t warp_per_block>
__global__ void topk_addr_backward_kernel(
    const TopKAddrPtr<acc_t, true> top_k_addr_ptr, //       r (B, H, K, T)

    const acc_t *const __restrict__ global_D_top_k_addr, // r (B, H, K, T)

    scalar_t *const __restrict__ D_addr_ptr, //             w (B, H, U, P, T)

    const TopKAddrStride D_topk_stride,

    const int_t *const __restrict__ position_ids, //        r (B, T)

    const TopKAddrConfig cfg
) {
    const int_t thread_idx = threadIdx.x;
    const int_t warp_idx = thread_idx / warp_size;
    const int_t lane_idx = thread_idx % warp_size;
    const int_t t_blocks = cfg.T / warp_size;
    const int_t global_warp_idx = blockIdx.x * warp_per_block + warp_idx;
    const int_t bh_idx = global_warp_idx / t_blocks;
    const int_t t_offs = global_warp_idx % t_blocks * warp_size;
    const int_t t_idx = t_offs + lane_idx;
    const int_t b_idx = bh_idx / cfg.H;
    const int_t h_idx = bh_idx % cfg.H;

    if (bh_idx >= cfg.B * cfg.H) return;

    const int_t *const __restrict__ local_idx_ptr = top_k_addr_ptr.idx + top_k_addr_ptr.idx_stride.offset(b_idx, h_idx, 0, t_idx);
    const acc_t *const __restrict__ local_val_ptr = top_k_addr_ptr.val + top_k_addr_ptr.val_stride.offset(b_idx, h_idx, 0, t_idx);
    const acc_t *const __restrict__ local_D_topk_ptr = global_D_top_k_addr + D_topk_stride.offset(b_idx, h_idx, 0, t_idx);
    scalar_t *const __restrict__ local_D_addr_ptr = D_addr_ptr + bh_idx * cfg.U * cfg.P * cfg.T; // (U, P, T)
    const acc_t addr_val_sign = cfg.addr_val_neg ? -1 : +1;

    acc_t topk_val[max_k], D_topk[max_k];

    /* load addr_val and the gradient */
#pragma unroll topk_unroll
    for (int_t k = 0; k < max_k; ++k)
        if (k < cfg.top_k) {
            topk_val[k] = addr_val_sign * local_val_ptr[k * top_k_addr_ptr.val_stride.k];
            // The sign only encodes values; upstream gradients refer to positive weights.
            D_topk[k] = local_D_topk_ptr[k * D_topk_stride.k];
        }

    if (cfg.top_k_re_norm) {
        /* backward the softmax if re-norm */
        acc_t D_topk_sum = 0;
#pragma unroll topk_unroll
        for (int_t k = max_k - 1; k >= 0; --k)
            if (k < cfg.top_k)
                D_topk_sum += topk_val[k] * D_topk[k];

#pragma unroll topk_unroll
        for (int_t k = 0; k < max_k; ++k)
            if (k < cfg.top_k)
                D_topk[k] = topk_val[k] * (D_topk[k] - D_topk_sum);
    }
    else {
        /* backward the exp if not re-norm */
#pragma unroll topk_unroll
        for (int_t k = 0; k < max_k; ++k)
            if (k < cfg.top_k)
                D_topk[k] = topk_val[k] * D_topk[k];
    }

    /* load slot index and remove CAPE */
    int_t addr_idx[max_k];
    const int_t a_offset = h_idx < cfg.E ? position_ids[b_idx * cfg.T + t_idx] % cfg.A : 0;
#pragma unroll topk_unroll
    for (int_t k = 0; k < max_k; ++k)
        if (k < cfg.top_k)
            addr_idx[k] = (local_idx_ptr[k * top_k_addr_ptr.idx_stride.k] + a_offset + cfg.A) % cfg.A;

    /* distribute the gradient into the sub-vector */
    for (int_t u = 0, stride = 1; u < cfg.U; ++u, stride *= cfg.P) {
        acc_t D_addr[max_p]; // gradient on sub-vector
#pragma unroll
        for (int_t p = 0; p < max_p; ++p)
            D_addr[p] = 0;

        /* distribute gradient of top-k to sub-vector selection */
#pragma unroll topk_unroll
        for (int_t k = 0; k < max_k; ++k)
            if (k < cfg.top_k)
                D_addr[(addr_idx[k] / stride) % cfg.P] += D_topk[k]; // accumulate in register-array first

        /* if not re-norm, re-distribute gradient inside the sub-vector (backward log-softmax) */
        if (!cfg.top_k_re_norm) {
            // rebuild the sorted sub-vector (bounded by memory, rebuild save up one write out)
            IdxValList<acc_t, max_p> addr;
            addr.init();

#pragma unroll
            for (int_t p = 0; p < max_p; ++p)
                if (p < cfg.P)
                    addr.set(p, p, type_cast<scalar_t, acc_t>(local_D_addr_ptr[(u * cfg.P + p) * cfg.T + t_idx]));

            addr.odd_even_sort();
            addr.topk_log_softmax(cfg.P);

            acc_t D_sum = 0; // sum of gradient for backward log-softmax

#pragma unroll
            for (int_t p = 0; p < max_p; ++p)
                D_sum += D_addr[p];

#pragma unroll
            for (int_t k = 0; k < max_p; ++k)
                if (addr.get_idx(k) >= 0) // in case of the padding
                    D_addr[addr.get_idx(k)] -= get_exp<acc_t>(addr.get_val(k)) * D_sum;
        }

        /* save the result to gmem */
#pragma unroll
        for (int_t p = 0; p < max_p; ++p)
            if (p < cfg.P)
                local_D_addr_ptr[(u * cfg.P + p) * cfg.T + t_idx] = type_cast<acc_t, scalar_t>(D_addr[p]);
    }
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t max_k, int_t max_p, int_t warp_per_block>
inline void topk_addr_forward_launcher(
    const torch_api::Tensor &addr, //         r (B, H, U, P, T)

    const TopKAddr &top_k_addr, //        w (B, H, K, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
) {
    using scalar_t = torch_api::ScalarTypeToCPPTypeT<scalar_dtype>;
    using acc_t = torch_api::ScalarTypeToCPPTypeT<acc_dtype>;

    RAMNET_CHECK(cfg.P <= max_p, "Expect P <= ", max_p, " but got ", cfg.P);
    RAMNET_CHECK(cfg.T % warp_size == 0, "T must align by ", warp_size, ", but get T=", cfg.T);
    RAMNET_CHECK(0 < cfg.top_k && cfg.top_k <= max_k, "Expect 0 < top_k <= ", max_k, " but got ", cfg.top_k);
    check_tensor("addr", addr, scalar_dtype, CheckContiguity::Whole, {cfg.B, cfg.H, cfg.U, cfg.P, cfg.T});
    check_tensor("position_ids", position_ids, int_dtype, CheckContiguity::Whole, {cfg.B, cfg.T});
    check_top_k_addr(top_k_addr, acc_dtype, cfg);

    const auto device = addr.device();
    torch_api::CUDAGuard device_guard(device);
    const auto stream = torch_api::current_cuda_stream(device);
    const torch_api::CUDAStreamGuard stream_guard(stream, device);

    const int_t addr_blocks = cdiv(cfg.B * cfg.H * (cfg.T / warp_size), warp_per_block);
    topk_addr_forward_kernel<scalar_t, acc_t, max_k, max_p, warp_per_block><<<addr_blocks, warp_per_block * warp_size, 0, stream>>>(
        torch_api::data_ptr<scalar_t, true>(addr),
        TopKAddrPtr<acc_t>{top_k_addr},
        torch_api::data_ptr<int_t, true>(position_ids),
        cfg
    );
    RAMNET_CUDA_KERNEL_LAUNCH_CHECK();
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t max_k, int_t max_p, int_t warp_per_block>
inline void topk_addr_backward_launcher(
    const TopKAddr &top_k_addr, //        r (B, H, K, T)

    const torch_api::Tensor &D_addr_val, //   r (B, H, K, T)

    const torch_api::Tensor &D_addr, //       w (B, H, U, P, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
) {
    using scalar_t = torch_api::ScalarTypeToCPPTypeT<scalar_dtype>;
    using acc_t = torch_api::ScalarTypeToCPPTypeT<acc_dtype>;

    RAMNET_CHECK(cfg.P <= max_p, "Expect P <= ", max_p, " but got ", cfg.P);
    RAMNET_CHECK(cfg.T % warp_size == 0, "T must align by ", warp_size, ", but get T=", cfg.T);
    RAMNET_CHECK(0 < cfg.top_k && cfg.top_k <= max_k, "Expect 0 < top_k <= ", max_k, " but got ", cfg.top_k);
    check_top_k_addr(top_k_addr, acc_dtype, cfg);
    check_tensor("position_ids", position_ids, int_dtype, CheckContiguity::Whole, {cfg.B, cfg.T});
    check_tensor("D_addr_val", D_addr_val, acc_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.top_k, cfg.T});
    check_tensor("D_addr", D_addr, scalar_dtype, CheckContiguity::Whole, {cfg.B, cfg.H, cfg.U, cfg.P, cfg.T});

    const auto device = D_addr.device();
    torch_api::CUDAGuard device_guard(device);
    const auto stream = torch_api::current_cuda_stream(device);
    const torch_api::CUDAStreamGuard stream_guard(stream, device);

    const int_t addr_blocks = cdiv(cfg.B * cfg.H * (cfg.T / warp_size), warp_per_block);
    topk_addr_backward_kernel<scalar_t, acc_t, max_k, max_p, warp_per_block><<<addr_blocks, warp_per_block * warp_size, 0, stream>>>(
        TopKAddrPtr<acc_t, true>{top_k_addr},
        torch_api::data_ptr<acc_t, true>(D_addr_val),
        torch_api::data_ptr<scalar_t>(D_addr),
        TopKAddrStride{D_addr_val},
        torch_api::data_ptr<int_t, true>(position_ids),
        cfg
    );
    RAMNET_CUDA_KERNEL_LAUNCH_CHECK();
}


template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype>
inline void topk_addr_forward_size_dispatcher(
    const torch_api::Tensor &addr, //         r (B, H, U, P, T)

    const TopKAddr &top_k_addr, //        w (B, H, K, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
) {
    constexpr static struct {
        bool (*check)(const int_t, const int_t);

        decltype(&topk_addr_forward_launcher<scalar_dtype, acc_dtype, 4, 4, 8>) func;
    } kernels[] = {
        // {[](const int_t k, const int_t p) -> bool { return k <= 4 && p <= 4; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 4, 4, 8>},
        {[](const int_t k, const int_t p) -> bool { return k <= 8 && p <= 4; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 8, 4, 8>},
        {[](const int_t k, const int_t p) -> bool { return k <= 16 && p <= 4; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 16, 4, 2>},
        {[](const int_t k, const int_t p) -> bool { return k <= 32 && p <= 4; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 32, 4, 2>},
        {[](const int_t k, const int_t p) -> bool { return k <= 64 && p <= 4; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 64, 4, 1>},
        {[](const int_t k, const int_t p) -> bool { return k <= 128 && p <= 4; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 128, 4, 1>},
        // {[](const int_t k, const int_t p) -> bool { return k <= 4 && p <= 8; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 4, 8, 8>},
        {[](const int_t k, const int_t p) -> bool { return k <= 8 && p <= 8; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 8, 8, 8>},
        // {[](const int_t k, const int_t p) -> bool { return k <= 16 && p <= 8; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 16, 8, 2>},
        {[](const int_t k, const int_t p) -> bool { return k <= 32 && p <= 8; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 32, 8, 2>},
        // {[](const int_t k, const int_t p) -> bool { return k <= 64 && p <= 8; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 64, 8, 1>},
        {[](const int_t k, const int_t p) -> bool { return k <= 128 && p <= 8; }, topk_addr_forward_launcher<scalar_dtype, acc_dtype, 128, 8, 1>},
    };

    for (const auto &[check, kernel]: kernels)
        if (check(cfg.top_k, cfg.P)) {
            kernel(
                addr,
                top_k_addr,
                position_ids,
                cfg
            );
            return;
        }
    RAMNET_CHECK(false, "No matching kernel found for top_k=", cfg.top_k, ", P=", cfg.P)
}

void topk_addr_forward_dispatcher(
    const torch_api::Tensor &addr, //         r (B, H, U, P, T)

    const TopKAddr &top_k_addr, //        w (B, H, K, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
) {
    decltype(&topk_addr_forward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float>) kernel = nullptr;
    // ReSharper disable once CppIncompleteSwitchStatement
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (addr.scalar_type()) {
        case torch_api::ScalarType::Half:
            kernel = topk_addr_forward_size_dispatcher<torch_api::ScalarType::Half, torch_api::ScalarType::Float>;
            break;
        case torch_api::ScalarType::BFloat16:
            kernel = topk_addr_forward_size_dispatcher<torch_api::ScalarType::BFloat16, torch_api::ScalarType::Float>;
            break;
        case torch_api::ScalarType::Float:
            kernel = topk_addr_forward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float>;
            break;
    }

    RAMNET_CHECK(kernel!=nullptr, "Unsupported scalar type: ", addr.scalar_type());
    kernel(
        addr,
        top_k_addr,
        position_ids,
        cfg
    );
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype>
inline void topk_addr_backward_size_dispatcher(
    const TopKAddr &top_k_addr, //        r (B, H, K, T)

    const torch_api::Tensor &D_addr_val, //   r (B, H, K, T)

    const torch_api::Tensor &D_addr, //       w (B, H, U, P, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
) {
    constexpr static struct {
        bool (*check)(const int_t, const int_t);

        decltype(&topk_addr_backward_launcher<scalar_dtype, acc_dtype, 4, 4, 8>) func;
    } kernels[] = {
        // {[](const int_t k, const int_t p) -> bool { return k <= 4 && p <= 4; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 4, 4, 8>},
        {[](const int_t k, const int_t p) -> bool { return k <= 8 && p <= 4; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 8, 4, 8>},
        {[](const int_t k, const int_t p) -> bool { return k <= 16 && p <= 4; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 16, 4, 2>},
        {[](const int_t k, const int_t p) -> bool { return k <= 32 && p <= 4; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 32, 4, 2>},
        {[](const int_t k, const int_t p) -> bool { return k <= 64 && p <= 4; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 64, 4, 1>},
        {[](const int_t k, const int_t p) -> bool { return k <= 128 && p <= 4; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 128, 4, 1>},
        // {[](const int_t k, const int_t p) -> bool { return k <= 4 && p <= 8; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 4, 8, 8>},
        {[](const int_t k, const int_t p) -> bool { return k <= 8 && p <= 8; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 8, 8, 8>},
        // {[](const int_t k, const int_t p) -> bool { return k <= 16 && p <= 8; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 16, 8, 2>},
        {[](const int_t k, const int_t p) -> bool { return k <= 32 && p <= 8; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 32, 8, 2>},
        // {[](const int_t k, const int_t p) -> bool { return k <= 64 && p <= 8; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 64, 8, 1>},
        {[](const int_t k, const int_t p) -> bool { return k <= 128 && p <= 8; }, topk_addr_backward_launcher<scalar_dtype, acc_dtype, 128, 8, 1>},
    };

    for (const auto &[check, kernel]: kernels)
        if (check(cfg.top_k, cfg.P)) {
            kernel(
                top_k_addr,
                D_addr_val,
                D_addr,
                position_ids,
                cfg
            );
            return;
        }
    RAMNET_CHECK(false, "No matching kernel found for top_k=", cfg.top_k, ", P=", cfg.P)
}

void topk_addr_backward_dispatcher(
    const TopKAddr &top_k_addr, //        r (B, H, K, T)

    const torch_api::Tensor &D_addr_val, //   r (B, H, K, T)

    const torch_api::Tensor &D_addr, //       w (B, H, U, P, T)

    const torch_api::Tensor &position_ids, // r (B, T)

    const TopKAddrConfig &cfg
) {
    decltype(&topk_addr_backward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float>) kernel = nullptr;
    // ReSharper disable once CppIncompleteSwitchStatement
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (D_addr.scalar_type()) {
        case torch_api::ScalarType::Half:
            kernel = topk_addr_backward_size_dispatcher<torch_api::ScalarType::Half, torch_api::ScalarType::Float>;
            break;
        case torch_api::ScalarType::BFloat16:
            kernel = topk_addr_backward_size_dispatcher<torch_api::ScalarType::BFloat16, torch_api::ScalarType::Float>;
            break;
        case torch_api::ScalarType::Float:
            kernel = topk_addr_backward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float>;
            break;
    }

    RAMNET_CHECK(kernel!=nullptr, "Unsupported scalar type: ", D_addr.scalar_type());
    kernel(
        top_k_addr,
        D_addr_val,
        D_addr,
        position_ids,
        cfg
    );
}
