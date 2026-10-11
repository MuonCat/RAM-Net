#pragma once
#include "ram_attn.cuh"

#include <cuda_fp16.h>      // __half
#include <cuda_bf16.h>      // __nv_bfloat16
#include <cuda_runtime.h>

template<typename>
inline constexpr bool always_false_v = false;

// Map scalar types to native vector types
template<typename T, int_t N>
struct NativeTraits {
    static_assert(always_false_v<T>, "NativeTraits<T, N> not supported for this type or vector width");
};

template<>
struct NativeTraits<float, 2> {
    using type = float2;
    __device__ static __forceinline__ type full(const float val) { return make_float2(val, val); }
};

template<>
struct NativeTraits<torch_api::Half, 2> {
    using type = half2;
    __device__ static __forceinline__ type full(const float val) { return make_half2(__float2half(val), __float2half(val)); }
};

template<>
struct NativeTraits<torch_api::BFloat16, 2> {
    using type = nv_bfloat162;
    __device__ static __forceinline__ type full(const float val) { return __floats2bfloat162_rn(val, val); }
};


template<typename T, int_t c_loop, bool native_vec = false>
struct Vec {
    T data[c_loop];

    // Default constructor (does not initialize to save registers)
    __device__ __forceinline__ Vec() {
    }

    __device__ __forceinline__ explicit Vec(const float val) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = val;
    }

    __device__ __forceinline__ Vec(const T *const src, const int_t C, const int lane_idx, const T padding = 0) {
        load(src, C, lane_idx, padding);
    }

    Vec(const Vec &) = default;

    __device__ __forceinline__ void load(const T *const src, const int_t C, const int lane_idx, const T padding = 0) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c) {
            if (const int_t c_idx = c * warp_size + lane_idx; c_idx < C)
                data[c] = src[c_idx];
            else
                data[c] = padding;
        }
    }

    __device__ __forceinline__ void store(T *const __restrict__ dst, const int_t C, const int lane_idx) const {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            if (const int_t c_idx = c * warp_size + lane_idx; c_idx < C)
                dst[c_idx] = data[c];
    }

    __device__ __forceinline__ void atomic_add(T *const __restrict__ dst, const int_t C, const int lane_idx) const {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            if (const int_t c_idx = c * warp_size + lane_idx; c_idx < C)
                atomicAdd(dst + c_idx, data[c]);
    }


    template<typename DstT>
    __device__ __forceinline__ Vec<DstT, c_loop, native_vec> cast() const {
        Vec<DstT, c_loop, native_vec> res;
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            res.data[c] = type_cast<T, DstT>(data[c]);
        return res;
    }

    __device__ __forceinline__ void scale_add_(const Vec &k, const Vec &b) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = __fmaf_rn(k.data[c], data[c], b.data[c]);
    }

    __device__ __forceinline__ void scale_add_(const T k, const Vec &b) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = __fmaf_rn(k, data[c], b.data[c]);
    }

    __device__ __forceinline__ void scale_add_(const Vec &k, const T b) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = __fmaf_rn(k.data[c], data[c], b);
    }

    __device__ __forceinline__ void scale_add_(const T k, const T b) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = __fmaf_rn(k, data[c], b);
    }

    __device__ __forceinline__ void lerp_(const T alpha, const Vec &x) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = __fmaf_rn(alpha, x.data[c] - data[c], data[c]);
    }

    __device__ __forceinline__ void operator*=(const T s) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] *= s;
    }

    __device__ __forceinline__ void operator*=(const Vec &other) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] *= other.data[c];
    }

    __device__ __forceinline__ Vec operator*(const T s) const {
        Vec res;
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            res.data[c] = data[c] * s;
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec &other) const {
        Vec res;
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            res.data[c] = data[c] * other.data[c];
        return res;
    }

    __device__ __forceinline__ void operator+=(const T s) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] += s;
    }


    __device__ __forceinline__ void operator+=(const Vec &other) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] += other.data[c];
    }

    __device__ __forceinline__ Vec operator+(const T s) const {
        Vec res;
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            res.data[c] = data[c] + s;
        return res;
    }

    __device__ __forceinline__ Vec operator+(const Vec &other) const {
        Vec res;
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            res.data[c] = data[c] + other.data[c];
        return res;
    }

    __device__ __forceinline__ void operator-=(const T s) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] -= s;
    }


    __device__ __forceinline__ void operator-=(const Vec &other) {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] -= other.data[c];
    }

    __device__ __forceinline__ Vec operator-(const Vec &other) const {
        Vec res;
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            res.data[c] = data[c] - other.data[c];
        return res;
    }

    __device__ __forceinline__ void sigmoid() {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = 1 / (1 + get_exp<T>(-data[c]));
    }

    __device__ __forceinline__ void mul_comp_() {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            data[c] = data[c] * (1 - data[c]);
    }

    __device__ __forceinline__ T dot(const Vec &other, T val = 0) const {
#pragma unroll
        for (int c = 0; c < c_loop; ++c)
            val += data[c] * other.data[c];
        return val;
    }
};

template<typename T, int_t c_loop>
struct Vec<T, c_loop, true> {
    using traits = NativeTraits<T, c_loop>;
    using native_t = typename traits::type;
    native_t data;

    // Default constructor (does not initialize to save registers)
    __device__ __forceinline__ Vec() {
    }

    __device__ __forceinline__ explicit Vec(const float val) {
        data = traits::full(val);
    }

    __device__ __forceinline__ Vec(const T *const src, const int_t C, const int lane_idx, const T padding = 0) {
        load(src, C, lane_idx, padding);
    }

    Vec(const Vec &) = default;

    __device__ __forceinline__ void load(const T *const src, const int_t C, const int lane_idx, const T padding = 0) {
        if (lane_idx < C / c_loop)
            data = reinterpret_cast<const native_t *>(src)[lane_idx];
        else
            data = traits::full(padding);
    }

    __device__ __forceinline__ void store(T *const __restrict__ dst, const int_t C, const int lane_idx) const {
        if (lane_idx < C / c_loop)
            reinterpret_cast<native_t *>(dst)[lane_idx] = data;
    }

    __device__ __forceinline__ void atomic_add(T *const __restrict__ dst, const int_t C, const int lane_idx) const {
        if constexpr (std::is_same_v<T, float>) {
            if constexpr (c_loop == 2) {
                constexpr unsigned full_mask = ~0u;
                constexpr unsigned width = warp_size / c_loop;

                /* apply warp transpose by shuffling for coalesced memory access */
                const float ex = __shfl_xor_sync(full_mask, (lane_idx < width ? data.y : data.x), width); // y[:16], x[16:] ==> x[16:], y[:16]
                const float lo = lane_idx < width ? data.x : ex; // x[:16], y[:16]
                const float hi = lane_idx < width ? ex : data.y; // x[16:], y[16:]

                const int_t src_lane = lane_idx % 2 * width + lane_idx / 2; // 0, 16, 1, 17, 2, 18, ...
                const float val_1 = __shfl_sync(full_mask, lo, src_lane);
                if (0 * warp_size + lane_idx < C)
                    atomicAdd(dst + 0 * warp_size + lane_idx, val_1);
                const float val_2 = __shfl_sync(full_mask, hi, src_lane);
                if (1 * warp_size + lane_idx < C)
                    atomicAdd(dst + 1 * warp_size + lane_idx, val_2);
            }
            else
                static_assert(always_false_v<T>, "atomic_add only support c_loop==2 for float");
        }
        else {
            if (lane_idx < C / c_loop)
                atomicAdd(reinterpret_cast<const native_t *>(dst) + lane_idx, data);
        }
    }


    template<typename DstT>
    __device__ __forceinline__ Vec<DstT, c_loop, true> cast() const {
        Vec<DstT, c_loop, true> out;
        out.data = type_cast<native_t, typename NativeTraits<DstT, c_loop>::type>(data);
        return out;
    }

    __device__ __forceinline__ void scale_add_(const Vec &k, const Vec &b) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = __fmaf_rn(k.data.x, data.x, b.data.x);
            data.y = __fmaf_rn(k.data.y, data.y, b.data.y);
        }
        else
            data = __hfma2(k, data, b);
    }

    __device__ __forceinline__ void scale_add_(const T k, const Vec &b) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = __fmaf_rn(k, data.x, b.data.x);
            data.y = __fmaf_rn(k, data.y, b.data.y);
        }
        else
            data = __hfma2(traits::full(k), data, b);
    }

    __device__ __forceinline__ void scale_add_(const Vec &k, const T b) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = __fmaf_rn(k.data.x, data.x, b);
            data.y = __fmaf_rn(k.data.y, data.y, b);
        }
        else
            data = __hfma2(k, data, traits::full(b));
    }

    __device__ __forceinline__ void scale_add_(const T k, const T b) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = __fmaf_rn(k, data.x, b);
            data.y = __fmaf_rn(k, data.y, b);
        }
        else
            data = __hfma2(traits::full(k), data, traits::full(b));
    }

    __device__ __forceinline__ void lerp_(const T alpha, const Vec &x) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = __fmaf_rn(alpha, x.data.x - data.x, data.x);
            data.y = __fmaf_rn(alpha, x.data.y - data.y, data.y);
        }
        else
            data = __hfma2(traits::full(alpha), x.data - data, data);
    }

    __device__ __forceinline__ void operator*=(const T s) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = s * data.x;
            data.y = s * data.y;
        }
        else
            data = __hmul2(native_t(s), data);
    }

    __device__ __forceinline__ void operator*=(const Vec &other) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = other.data.x * data.x;
            data.y = other.data.y * data.y;
        }
        else
            data = __hmul2(other.data, data);
    }

    __device__ __forceinline__ Vec operator*(const T s) const {
        Vec res;
        if constexpr (std::is_same_v<T, float>) {
            res.data.x = data.x * s;
            res.data.y = data.y * s;
        }
        else {
            res.data = __hmul2(data, traits::full(s));
        }
        return res;
    }

    __device__ __forceinline__ Vec operator*(const Vec &other) const {
        Vec res;
        if constexpr (std::is_same_v<T, float>) {
            res.data.x = data.x * other.data.x;
            res.data.y = data.y * other.data.y;
        }
        else {
            res.data = __hmul2(data, other.data);
        }
        return res;
    }

    __device__ __forceinline__ void operator+=(const T s) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = data.x + s;
            data.y = data.y + s;
        }
        else
            data = __hadd2(data, native_t(s));
    }

    __device__ __forceinline__ void operator+=(const Vec &other) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = data.x + other.data.x;
            data.y = data.y + other.data.y;
        }
        else
            data = __hadd2(data, other.data);
    }

    __device__ __forceinline__ Vec operator+(const T s) const {
        Vec res;
        if constexpr (std::is_same_v<T, float>) {
            res.data.x = data.x + s;
            res.data.y = data.y + s;
        }
        else {
            res.data = __hadd2(data, traits::full(s));
        }
        return res;
    }

    __device__ __forceinline__ Vec operator+(const Vec &other) const {
        Vec res;
        if constexpr (std::is_same_v<T, float>) {
            res.data.x = data.x + other.data.x;
            res.data.y = data.y + other.data.y;
        }
        else {
            res.data = __hadd2(data, other.data);
        }
        return res;
    }

    __device__ __forceinline__ void operator-=(const T s) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = data.x - s;
            data.y = data.y - s;
        }
        else
            data = __hsub2(data, native_t(s));
    }

    __device__ __forceinline__ void operator-=(const Vec &other) {
        if constexpr (std::is_same_v<T, float>) {
            data.x = data.x - other.data.x;
            data.y = data.y - other.data.y;
        }
        else
            data = __hsub2(data, other.data);
    }

    __device__ __forceinline__ Vec operator-(const Vec &other) const {
        Vec res;
        if constexpr (std::is_same_v<T, float>) {
            res.data.x = data.x - other.data.x;
            res.data.y = data.y - other.data.y;
        }
        else {
            res.data = __hsub2(data, other.data);
        }
        return res;
    }

    __device__ __forceinline__ void sigmoid() {
        data.x = 1 / (1 + get_exp<T>(-data.x));
        data.y = 1 / (1 + get_exp<T>(-data.y));
    }

    __device__ __forceinline__ void mul_comp_() {
        data.x = data.x * (1 - data.x);
        data.y = data.y * (1 - data.y);
    }

    __device__ __forceinline__ T dot(const Vec &other, T val = 0) const {
        if constexpr (std::is_same_v<T, float>)
            return data.x * other.data.x + data.y * other.data.y;
        else {
            const auto m = __hmul2(data, other.data);
            return m.x + m.y;
        }
    }
};
