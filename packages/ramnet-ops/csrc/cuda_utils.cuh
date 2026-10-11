#pragma once
#include "ram_attn.cuh"

#include <cmath>
#include <cstdio>
#include <cuda_fp16.h>      // __half
#include <cuda_bf16.h>      // __nv_bfloat16
#include <thrust/pair.h>
#include <cuda_runtime.h>
#include "torch_compat_cuda.cuh"

constexpr int_t topk_unroll = 32;

template<typename Kernel>
void print_kernel_block_resources(const Kernel k) {
    cudaFuncAttributes attr{};
    cudaError_t err = cudaFuncGetAttributes(&attr, k);
    if (err != cudaSuccess) {
        printf("cudaFuncGetAttributes failed: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("Kernel block resource usage:\n");
    printf("  regs per thread      : %d\n", attr.numRegs);
    printf("  static shmem (bytes) : %zu\n", attr.sharedSizeBytes);
    printf("  const memory (bytes) : %zu\n", attr.constSizeBytes);
    printf("  local memory (bytes) : %zu\n", attr.localSizeBytes);
    printf("  max threads per block: %d\n", attr.maxThreadsPerBlock);
}

template<typename int_t>
__host__ __device__ __forceinline__
int_t cdiv(const int_t x, const int_t y) { return (x + y - 1) / y; }

template<typename int_t>
int_t fast_pow(int_t base, int_t exponent) {
    int_t res = 1;
    while (exponent > 0) {
        if (exponent & 1) res *= base;
        base *= base, exponent >>= 1;
    }
    return res;
}

template<typename scalar_t>
__forceinline__ __device__ scalar_t get_exp(const scalar_t x) {
    if constexpr (std::is_same_v<scalar_t, float>)
        return expf(x); // FP32
    else if constexpr (std::is_same_v<scalar_t, double>)
        return exp(x); // FP64
    else if constexpr (std::is_same_v<scalar_t, __half>)
        return __half(expf(__half2float(x))); // FP16
    else if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>)
        return __nv_bfloat16(expf(__bfloat162float(x))); // BF16
    else
        static_assert(sizeof(scalar_t) == 0, "Unsupported scalar type for get_exp");
    return std::nan("");
}

template<typename scalar_t>
__forceinline__ __device__ scalar_t get_log(const scalar_t x) {
    if constexpr (std::is_same_v<scalar_t, float>)
        return logf(x); // FP32
    else if constexpr (std::is_same_v<scalar_t, double>)
        return log(x); // FP64
    else if constexpr (std::is_same_v<scalar_t, __half>)
        return __float2half(logf(__half2float(x))); // FP16
    else if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>)
        return __float2bfloat16(logf(__bfloat162float(x))); // BF16
    else
        static_assert(sizeof(scalar_t) == 0, "Unsupported scalar type for get_log");
    return nan("");
}

template<typename scalar_t>
__forceinline__ __device__ scalar_t get_pow(const scalar_t x, const scalar_t y) {
    if constexpr (std::is_same_v<scalar_t, float>)
        return powf(x, y); // FP32
    else if constexpr (std::is_same_v<scalar_t, double>)
        return pow(x, y); // FP64
    else if constexpr (std::is_same_v<scalar_t, __half>)
        return __half(powf(__half2float(x), __half2float(y))); // FP16
    else if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>)
        return __nv_bfloat16(powf(__bfloat162float(x), __bfloat162float(y))); // BF16
    else
        static_assert(sizeof(scalar_t) == 0, "Unsupported scalar type for get_pow");
    return std::nan("");
}

template<typename from_t, typename to_t>
__device__ __forceinline__ to_t type_cast(const from_t &v) { return static_cast<to_t>(v); }

template<typename T>
__device__ __forceinline__ T type_cast(const T &v) { return v; }

template<>
__device__ __forceinline__ float type_cast<__half, float>(const __half &v) { return __half2float(v); }

template<>
__device__ __forceinline__ float type_cast<__nv_bfloat16, float>(const __nv_bfloat16 &v) { return __bfloat162float(v); }

template<>
__device__ __forceinline__ __half type_cast<float, __half>(const float &v) { return __float2half(v); }

template<>
__device__ __forceinline__ __nv_bfloat16 type_cast<float, __nv_bfloat16>(const float &v) { return __float2bfloat16(v); }

template<>
__device__ __forceinline__ float2 type_cast<__half2, float2>(const __half2 &v) { return __half22float2(v); }

template<>
__device__ __forceinline__ float2 type_cast<__nv_bfloat162, float2>(const __nv_bfloat162 &v) { return make_float2(__bfloat162float(v.x), __bfloat162float(v.y)); }

template<>
__device__ __forceinline__ __half2 type_cast<float2, __half2>(const float2 &v) { return __float22half2_rn(v); }

template<>
__device__ __forceinline__ __nv_bfloat162 type_cast<float2, __nv_bfloat162>(const float2 &v) { return __floats2bfloat162_rn(v.x, v.y); }

template<typename acc_t, int_t width = warp_size>
__device__ __forceinline__ void warp_affine_scan_L2R(acc_t &k, acc_t &b, const int_t lane) {
    static_assert(width <= warp_size);
#pragma unroll
    for (unsigned offset = 1; offset < width; offset <<= 1) {
        constexpr unsigned full_mask = ~0u;
        const acc_t L_k = __shfl_up_sync(full_mask, k, offset);
        const acc_t L_b = __shfl_up_sync(full_mask, b, offset);
        if (lane >= offset) {
            b += k * L_b;
            k *= L_k;
        }
    }
}

template<typename acc_t, int_t width = warp_size>
__device__ __forceinline__ void warp_affine_scan_R2L(acc_t &k, acc_t &b, const int_t lane) {
    static_assert(width <= warp_size);
#pragma unroll
    for (unsigned offset = 1; offset < width; offset <<= 1) {
        constexpr unsigned full_mask = ~0u;
        const acc_t R_k = __shfl_down_sync(full_mask, k, offset);
        const acc_t R_b = __shfl_down_sync(full_mask, b, offset);
        if (lane + offset < width) {
            b += k * R_b;
            k *= R_k;
        }
    }
}

template<typename acc_t>
__device__ __forceinline__ acc_t warp_shift_L2R(const acc_t val, const acc_t padding, const int_t lane) {
    constexpr unsigned full_mask = ~0u;
    acc_t L_val = __shfl_up_sync(full_mask, val, 1);
    return lane == 0 ? padding : L_val;
}

template<typename acc_t>
__device__ __forceinline__ acc_t warp_shift_R2L(const acc_t val, const acc_t padding, const int_t lane) {
    constexpr unsigned full_mask = ~0u;
    acc_t R_val = __shfl_down_sync(full_mask, val, 1);
    return lane == warp_size - 1 ? padding : R_val;
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_affine_scan_forward(scalar_t k, scalar_t b, const int_t lane, const scalar_t init_val) {
    // apply affine scan:
    // y_i = k_i * y_{i-1} + b_i s.t. y_{-1} = init_val and i is lane idx
    warp_affine_scan_L2R(k, b, lane);
    return k * init_val + b;
}

template<typename scalar_t>
__device__ __forceinline__ thrust::pair<scalar_t, scalar_t> warp_affine_scan_backward(const scalar_t k, const scalar_t b, const int_t lane, const scalar_t init_val) {
    // apply reverse affine scan (suffix scan):
    // y_i = k_i * x_i s.t. x_i = (y_{i+1} + b_i), and y_{warp_size} = init_val
    // return y_i and x_i
    scalar_t B = k * b;
    warp_affine_scan_R2L(k, B, lane);
    const scalar_t y = k * init_val + B;
    return {y, warp_shift_R2L(y, init_val, lane) + b};
}

template<typename acc_t, int_t N, int_t G = 1>
struct IdxValList {
    // SoA for sort
    int_t idx[N][G];
    acc_t val[N][G];

    __device__ __forceinline__ void set(const int_t k, const int_t new_idx, const acc_t new_val, const int_t lane = 0) {
        idx[k][lane] = new_idx;
        val[k][lane] = new_val;
    }

    __device__ __forceinline__ int_t get_idx(const int_t k, const int_t lane = 0) const {
        return idx[k][lane];
    }

    __device__ __forceinline__ acc_t get_val(const int_t k, const int_t lane = 0) const {
        return val[k][lane];
    }

    __device__ __forceinline__ void init(const int_t lane = 0) {
#pragma unroll (G == 1 ? N : 8)
        for (int_t i = 0; i < N; ++i)
            set(i, -1, -std::numeric_limits<acc_t>::infinity(), lane);
    }

    __device__ __forceinline__ void cmp_swap(const int_t i, const int_t j, const int_t lane = 0) {
        if (get_val(i, lane) < get_val(j, lane)) {
            const acc_t tmp_val = get_val(i, lane);
            const int_t tmp_idx = get_idx(i, lane);
            set(i, get_idx(j, lane), get_val(j, lane), lane);
            set(j, tmp_idx, tmp_val, lane);
        }
    }

    template<int_t M>
    __device__ __forceinline__ void copy_from(const IdxValList<acc_t, M> &other, const int_t lane = 0, const int_t other_lane = 0) {
#pragma unroll
        for (int_t i = 0; i < M; ++i)
            if (i < N)
                set(i, other.get_idx(i, other_lane), other.get_val(i, other_lane), lane);
    }

    __device__ __forceinline__ void odd_even_sort(const int_t lane = 0) {
#pragma unroll
        for (int_t i = 0; i < N; ++i) {
#pragma unroll
            for (int_t j = 1 + (i & 1); j < N; j += 2)
                cmp_swap(j - 1, j, lane);
        }
    }

    __device__ __forceinline__ void topk_log_softmax(const int_t max_size = N, const int_t lane = 0) {
        // apply log-softmax on the topk, where topk must be sorted
        const acc_t max_log = val[0][lane];

        if (max_log <= -std::numeric_limits<acc_t>::infinity())
            return;

        acc_t sum_exp = 0;
#pragma unroll
        for (int_t i = N - 1; i >= 0; --i) // scan form small to large: better numerical stability due to float rounding
            if (i < max_size)
                sum_exp += get_exp<acc_t>(val[i][lane] - max_log);

        const acc_t log_sum_exp = get_log<acc_t>(sum_exp);
#pragma unroll
        for (int_t i = 0; i < N; ++i)
            val[i][lane] = (val[i][lane] - max_log) - log_sum_exp; // remove the max first
    }

    template<int_t M>
    __device__ __forceinline__ void sum_merge_from(const IdxValList<acc_t, M> &peer, const int idx_stride, const int_t lane = 0, const int_t other_lane = 0) {
        // assume both main and peer are sorted
        // get the top-k of {main[i] + peer[j]}
        // the new index is (i + j * idx_stride)
        IdxValList<acc_t, N> res;
        int_t frontier[M];
#pragma unroll
        for (int_t i = 0; i < M; ++i)
            frontier[i] = 0;

#pragma unroll 8
        for (int_t k = 0; k < N; ++k) {
            acc_t best_val = -std::numeric_limits<acc_t>::infinity();
            int_t best_r = -1, best_idx = -1;
#pragma unroll
            for (int_t r = 0; r < M; ++r)
                if (const int_t main_idx = frontier[r]; main_idx < N)
                    if (acc_t curr_val = get_val(main_idx, lane) + peer.get_val(r, other_lane); curr_val > best_val) {
                        best_val = curr_val;
                        best_r = r;
                        best_idx = get_idx(main_idx, lane) + idx_stride * peer.get_idx(r, other_lane);
                    }

            res.set(k, best_idx, best_val);
#pragma unroll
            for (int i = 0; i < M; ++i)
                if (i == best_r)
                    ++frontier[i];
        }

        copy_from(res, lane);
    }
};

template<typename acc_t>
struct GainResult {
    acc_t gain;
    acc_t new_mass;
};

template<typename acc_t>
__device__ __forceinline__ GainResult<acc_t> get_gain(const acc_t w, const acc_t gamma, const acc_t mass, const acc_t eps = 0) {
    // Stable alpha forms:
    //   gamma <  0: alpha = (1 - w) / (1 - w + exp(gamma) * w) <== origin formula.
    //   gamma >= 0: alpha = exp(-gamma) * (1 - w) / (w + exp(-gamma) * (1 - w)) <== rescales the numerator and denominator by exp(-gamma),
    // keeping the exponential bounded by one without changing alpha.

    const acc_t exp_gamma = get_exp<acc_t>(gamma >= 0 ? -gamma : gamma);
    const acc_t inv_den = 1 / (gamma >= 0 ? (w + exp_gamma * (1 - w)) : (1 - w + exp_gamma * w));
    const acc_t alpha = (gamma >= 0 ? (1 - w) * exp_gamma : (1 - w)) * inv_den;

    const acc_t new_mass = alpha * mass + w;
    return {w / (new_mass + eps), new_mass};
}

template<typename acc_t>
struct GainBackwardResult {
    acc_t gain;
    acc_t D_w;
    acc_t D_gamma;
    acc_t D_prev_mass;
};

template<typename acc_t>
__device__ __forceinline__ GainBackwardResult<acc_t> get_Dw_Dgamma_Dmass(
    const acc_t D_gain,
    const acc_t D_mass,
    const acc_t w,
    const acc_t gamma,
    const acc_t prev_mass,
    const acc_t eps = 0
) {
    const acc_t exp_gamma = get_exp<acc_t>(gamma >= 0 ? -gamma : gamma);
    const acc_t inv_den = 1 / (gamma >= 0 ? (w + exp_gamma * (1 - w)) : (1 - w + exp_gamma * w));
    const acc_t alpha = (gamma >= 0 ? (1 - w) * exp_gamma : (1 - w)) * inv_den;

    const acc_t inv_new_mass = 1 / (alpha * prev_mass + w + eps);
    const acc_t D_mass_total = D_mass - D_gain * w * inv_new_mass * inv_new_mass; // D L / D (alpha * prev_mass + w)

    const acc_t d_alpha_dw = -exp_gamma * inv_den * inv_den;
    const acc_t d_alpha_dg = -alpha * (1 - alpha);
    const acc_t D_alpha = prev_mass * D_mass_total;

    return {
        w * inv_new_mass, //                                            gain
        inv_new_mass * D_gain + d_alpha_dw * D_alpha + D_mass_total, // D_w
        d_alpha_dg * D_alpha, //                                        D_gamma
        alpha * D_mass_total //                                         D_mass
    };
}


template<typename T, bool ReadOnly = false>
struct StatePtr {
    using ptr_t = std::conditional_t<ReadOnly, const T *, T *>;
    ptr_t slot;
    ptr_t mass;

    StatePtr()
        : slot(nullptr), mass(nullptr) {
    }

    explicit StatePtr(const State &s)
        : slot(torch_api::data_ptr<T, ReadOnly>(s.slot)), mass(torch_api::data_ptr<T, ReadOnly>(s.mass)) {
    }
};

template<typename scalar_t, typename acc_t, bool ReadOnly = true>
struct InPackPtr {
    using s_ptr_t = std::conditional_t<ReadOnly, const scalar_t *, scalar_t *>;
    using a_ptr_t = std::conditional_t<ReadOnly, const acc_t *, acc_t *>;
    s_ptr_t x;
    s_ptr_t gamma;

    explicit InPackPtr(const InPack &inputs)
        : x(torch_api::data_ptr<scalar_t, ReadOnly>(inputs.x)), gamma(torch_api::data_ptr<scalar_t, ReadOnly>(inputs.gamma)) {
    }
};


enum class CheckContiguity { None, Whole, LastDim };

inline void check_tensor(const char *const name, const torch_api::Tensor &t, const torch_api::ScalarType dtype,
                         const CheckContiguity contiguity = CheckContiguity::Whole,
                         const torch_api::IntArrayRef sizes = {}) {
    RAMNET_CHECK(t.is_cuda(), "Tensor ", name, " must be a CUDA tensor");

    if (contiguity == CheckContiguity::Whole)
        RAMNET_CHECK(t.is_contiguous(), "Tensor ", name, " must be contiguous");
    if (contiguity == CheckContiguity::LastDim)
        RAMNET_CHECK(t.stride(-1) == 1, "Tensor ", name, " last dim must be contiguous");

    RAMNET_CHECK(t.scalar_type() == dtype, "Tensor ", name, " type mismatch: expect ", torch_api::toString(dtype), ", but got ", torch_api::toString(t.scalar_type()));

    if (!sizes.empty())
        RAMNET_CHECK(t.sizes().equals(sizes), "Tensor ", name, " shape mismatch: expect ", torch_api::shape_string(sizes), ", but got ", torch_api::shape_string(t.sizes()));
}
