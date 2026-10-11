/*
 * RAM Attention kernel type:
 * - Parallel sequence kernel (T>1), Top-K sparse path.
 * - Segment-parallel execution over active slots/events.
 *
 * Parallel decomposition:
 * - Global work space: active segments S (each segment maps to one (b, h, a) slot).
 * - Grid mapping: one warp handles one segment.
 * - Time/event mapping: events within a segment are processed in order.
 *
 * Warp/lane boundary:
 * - Warp: owns one segment and the recurrent state flow for that segment.
 * - Lane: parallelizes channel work (vector lanes over C) and cooperates on reductions.
 *
 * Structure hint:
 * - Forward and backward are organized as directional scans over the same segment layout.
 */

// Include before PyTorch headers to avoid cuda/c10::cuda ambiguity with CUDA 11.8 on MSVC.
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "ram_attn_segment_parallel.cuh"
#include "cuda_utils.cuh"
#include "vector_ops.cuh"

namespace cg = cooperative_groups;

struct ValueStride {
    int64_t b, h, t;

    ValueStride() : b(0), h(0), t(0) {
    }

    explicit ValueStride(const torch_api::Tensor &tensor)
        : b(tensor.stride(0)), h(tensor.stride(1)), t(tensor.stride(2)) {
    }

    __host__ __device__ inline
    int64_t offset(const int_t b_idx, const int_t h_idx, const int_t t_idx) const {
        return b_idx * b + h_idx * h + t_idx * t;
    }

    __host__ __device__ inline
    bool time_varying() const {
        return t != 0;
    }
};

struct SlotStride {
    int64_t b, h, a;

    explicit SlotStride(const torch_api::Tensor &tensor)
        : b(tensor.stride(0)), h(tensor.stride(1)), a(tensor.stride(2)) {
    }

    __host__ __device__ inline
    int64_t offset(const int_t b_idx, const int_t h_idx, const int_t a_idx) const {
        return b_idx * b + h_idx * h + a_idx * a;
    }
};

struct StateStride {
    SlotStride slot;
    SlotStride mass;

    explicit StateStride(const State &s)
        : slot(s.slot), mass(s.mass) {
    }
};

struct InputStride {
    ValueStride x;
    ValueStride gamma;

    explicit InputStride(const InPack &inputs)
        : x(inputs.x), gamma(inputs.gamma) {
    }
};

template<typename acc_t>
struct SegmentMetaPtr {
    const int_t *seg_idx; // Sum S <= B * H * A
    const int_t *seg_off; // Sum S <= B * H * A
    const int_t *seg_len; // Sum S <= B * H * A
    const int_t *event_i; // B * H * (Kw + Kr) * T
    const acc_t *event_v; // B * H * (Kw + Kr) * T

    explicit SegmentMetaPtr(const SegmentMeta &m)
        : seg_idx(torch_api::data_ptr<int_t, true>(m.seg_idx)), seg_off(torch_api::data_ptr<int_t, true>(m.seg_off)), seg_len(torch_api::data_ptr<int_t, true>(m.seg_len)),
          event_i(torch_api::data_ptr<int_t, true>(m.event_i)), event_v(torch_api::data_ptr<acc_t, true>(m.event_v)) {
    }
};

struct SegmentInterState {
    torch_api::Tensor event_slot;
    torch_api::Tensor event_mass;
    torch_api::Tensor post_slot;
};

template<typename T, bool ReadOnly = false>
struct SegmentInterStatePtr {
    using ptr_t = std::conditional_t<ReadOnly, const T *, T *>;
    ptr_t event_slot;
    ptr_t event_mass;
    ptr_t post_slot;

    SegmentInterStatePtr()
        : event_slot(nullptr), event_mass(nullptr), post_slot(nullptr) {
    }

    explicit SegmentInterStatePtr(const SegmentInterState &s)
        : event_slot(torch_api::data_ptr<T, ReadOnly>(s.event_slot)),
          event_mass(torch_api::data_ptr<T, ReadOnly>(s.event_mass)),
          post_slot(torch_api::data_ptr<T, ReadOnly>(s.post_slot)) {
    }
};

struct ReadSums {
    torch_api::Tensor sum_r;
    torch_api::Tensor sum_rm;

    ReadSums() = default;

    ReadSums(
        const SegmentConfig &cfg,
        const torch_api::ScalarType dtype,
        const torch_api::Device device
    ) : sum_r(torch_api::zeros({cfg.B, cfg.H, cfg.T}, dtype, device)),
        sum_rm(torch_api::zeros({cfg.B, cfg.H, cfg.T}, dtype, device)) {
    }

    void normalize_(const torch_api::Tensor &y, const float eps) const {
        const auto scale = torch_api::div(sum_r, torch_api::add(sum_rm, eps));
        torch_api::mul_(y, torch_api::unsqueeze(scale, -1));
    }

    torch_api::Tensor backward_(const torch_api::Tensor &y, const torch_api::Tensor &D_y, const float eps) {
        const auto denom = torch_api::add(sum_rm, eps);
        const auto scale = torch_api::div(sum_r, denom);
        const auto D_y_acc = torch_api::to(D_y, sum_r.scalar_type());

        // y is the unscaled sum(r * mass * slot). Replace the sums with their gradients.
        sum_r = torch_api::div(torch_api::sum(torch_api::mul(D_y_acc, y), -1), denom);
        sum_rm = torch_api::mul(torch_api::neg(sum_r), scale);
        return torch_api::to(torch_api::mul(D_y_acc, torch_api::unsqueeze(scale, -1)), D_y.scalar_type());
    }
};

template<typename T, bool ReadOnly = false>
struct ReadSumsPtr {
    using ptr_t = std::conditional_t<ReadOnly, const T *, T *>;
    ptr_t sum_r;
    ptr_t sum_rm;

    ReadSumsPtr() : sum_r(nullptr), sum_rm(nullptr) {
    }

    explicit ReadSumsPtr(const ReadSums &s)
        : sum_r(torch_api::data_ptr<T, ReadOnly>(s.sum_r)), sum_rm(torch_api::data_ptr<T, ReadOnly>(s.sum_rm)) {
    }
};

inline void check_segment_meta(const SegmentMeta &seg_meta, const torch_api::ScalarType acc_dtype, const SegmentConfig &cfg) {
    check_tensor("seg_idx", seg_meta.seg_idx, int_dtype, CheckContiguity::Whole);
    check_tensor("seg_off", seg_meta.seg_off, int_dtype, CheckContiguity::Whole);
    check_tensor("seg_len", seg_meta.seg_len, int_dtype, CheckContiguity::Whole);
    check_tensor("event_i", seg_meta.event_i, int_dtype, CheckContiguity::Whole);
    check_tensor("event_v", seg_meta.event_v, acc_dtype, CheckContiguity::Whole);

    auto check_1d = [](const char *name, const torch_api::Tensor &t) {
        RAMNET_CHECK(t.dim() == 1, name, " must be 1-D, but got dim=", t.dim(), " shape=", torch_api::shape_string(t.sizes()));
    };

    check_1d("seg_idx", seg_meta.seg_idx);
    check_1d("seg_off", seg_meta.seg_off);
    check_1d("seg_len", seg_meta.seg_len);
    check_1d("event_i", seg_meta.event_i);
    check_1d("event_v", seg_meta.event_v);

    RAMNET_CHECK(
        seg_meta.seg_idx.numel() == seg_meta.seg_off.numel() && seg_meta.seg_off.numel() == seg_meta.seg_len.numel(),
        "seg tensors size mismatch: "
        "seg_idx=", seg_meta.seg_idx.numel(),
        ", seg_off=", seg_meta.seg_off.numel(),
        ", seg_len=", seg_meta.seg_len.numel()
    );

    RAMNET_CHECK(
        seg_meta.event_i.numel() == seg_meta.event_v.numel(),
        "event tensors size mismatch: "
        "event_i=", seg_meta.event_i.numel(),
        ", event_v=", seg_meta.event_v.numel()
    );

    const int64_t event_count = static_cast<int64_t>(cfg.B) * cfg.H * cfg.T * (cfg.w_top_k + cfg.r_top_k);
    RAMNET_CHECK(
        seg_meta.event_i.numel() <= event_count,
        "Too many segment events: "
        "numel=", seg_meta.event_i.numel(),
        ", maximum ==", event_count,
        " (B=", cfg.B,
        ", H=", cfg.H,
        ", T=", cfg.T,
        ", w_top_k=", cfg.w_top_k,
        ", r_top_k=", cfg.r_top_k, ")"
    );
}

template<typename acc_t>
struct segment_info {
    int_t seg_off, seg_end, bh_idx, a_idx, b_idx, h_idx;

    __host__ __device__
    explicit segment_info(const SegmentMetaPtr<acc_t> seg_meta_ptr, const int_t s_idx, const SegmentConfig &cfg) {
        const int_t *const __restrict__ seg_idx_ptr = seg_meta_ptr.seg_idx;
        const int_t *const __restrict__ seg_off_ptr = seg_meta_ptr.seg_off;
        const int_t *const __restrict__ seg_len_ptr = seg_meta_ptr.seg_len;

        seg_off = seg_off_ptr[s_idx];
        seg_end = seg_off + seg_len_ptr[s_idx];

        const int_t seg_idx = seg_idx_ptr[s_idx];
        bh_idx = seg_idx / cfg.A;
        a_idx = seg_idx % cfg.A;
        b_idx = bh_idx / cfg.H;
        h_idx = bh_idx % cfg.H;
    }
};

template<unsigned max_bytes, unsigned group = 4>
__device__ __forceinline__ void ptx_prefetch_global_l1(const void *const ptr, const int_t nbytes, const int_t lane_idx) {
    const auto base = static_cast<const uint8_t *>(ptr);
#pragma unroll
    for (int_t i = 0; i < max_bytes; i += warp_size * group)
        if (const int_t off = i + lane_idx * group; off < nbytes)
            asm volatile("prefetch.global.L1 [%0];" : : "l"(base + off));
}

/* scan the segment from left ==> right, IS_FORWARD control the mode (for forward or backward) */
template<typename scalar_t, typename acc_t, int_t c_loop, int_t warp_per_block, bool IS_BWD, bool MASS_AWARE, bool native_vec = false, unsigned ptx_prefetch = 16>
__global__ void topk_segment_l2r_kernel(
    const SegmentMetaPtr<acc_t> seg_meta_ptr, //          r

    const InPackPtr<scalar_t, acc_t> input_ptr, //        r  (B, H, T, [C])

    const StatePtr<acc_t, true> init_state_ptr, //        r  (B, H, A, [C])

    acc_t *const __restrict__ global_y_ptr, //           r/w (B, H, T, C), optional replay output
    const StatePtr<acc_t> post_state_ptr, //              w  (B, H, A, [C]), forward only

    // read accumulation for forward or mass-aware backward recomputation.
    const ReadSumsPtr<acc_t> read_sums, //               r/w (B, H, T), optional

    // Backward (store the event info for R2L scan)
    // for write event: save update delta and previous mass
    const SegmentInterStatePtr<acc_t> inter_state_ptr, // w, backward replay cache

    const int_t s_size,
    const InputStride in_stride,
    const StateStride init_state_stride,
    const StateStride post_state_stride,
    const ValueStride y_stride,
    const SegmentConfig cfg
) {
    static_assert(ptx_prefetch < warp_size);
    constexpr bool IS_FWD = !IS_BWD;
    using acc_vec = Vec<acc_t, c_loop, native_vec>;
    using scalar_vec = Vec<scalar_t, c_loop, native_vec>;
    constexpr int_t coalesced_align_mask = ~(warp_size - 1);

    // get index of current warp
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<warp_size>(block);
    const int_t warp_idx = block.thread_rank() / warp.size();
    const int_t lane_idx = warp.thread_rank();
    const int_t s_idx = blockIdx.x * warp_per_block + warp_idx;
    if (s_idx >= s_size || seg_meta_ptr.seg_len[s_idx] == 0) return;

    // load the segment information
    const acc_t *const __restrict__ event_v_ptr = seg_meta_ptr.event_v;
    const int_t *const __restrict__ event_i_ptr = seg_meta_ptr.event_i;
    const segment_info seg_info{seg_meta_ptr, s_idx, cfg};

    const scalar_t *const __restrict__ x_ptr = input_ptr.x; // (B, H, T, C)
    const scalar_t *const __restrict__ gamma_ptr = input_ptr.gamma; // (B, H, T)
    const acc_t *const init_slot = init_state_ptr.slot + init_state_stride.slot.offset(seg_info.b_idx, seg_info.h_idx, seg_info.a_idx); // (C)
    const acc_t *const init_mass = init_state_ptr.mass + init_state_stride.mass.offset(seg_info.b_idx, seg_info.h_idx, seg_info.a_idx); // ()
    acc_t *const post_slot = post_state_ptr.slot + post_state_stride.slot.offset(seg_info.b_idx, seg_info.h_idx, seg_info.a_idx); // (C)
    acc_t *const post_mass = post_state_ptr.mass + post_state_stride.mass.offset(seg_info.b_idx, seg_info.h_idx, seg_info.a_idx); // ()

    // load initial memory state
    acc_vec slot(init_slot, cfg.C, lane_idx);
    acc_t mass = *init_mass;

    // pre-fetch data
    scalar_vec next_cache;
    scalar_t next_gamma_cache;
    // pre-fetch meta
    acc_t val_cache, next_val_cache;
    int_t t_idx_cache, next_t_idx_cache;

    // update the metadata (val, idx) of next chunk
    const auto update_meta = [&](const int_t off) {
        // update current
        val_cache = next_val_cache;
        t_idx_cache = next_t_idx_cache;

        // load next chunk
        if (const int_t seq_pos = off + lane_idx; seg_info.seg_off <= seq_pos && seq_pos < seg_info.seg_end) {
            next_val_cache = event_v_ptr[seq_pos];
            const int_t next_idx = event_i_ptr[seq_pos];
            next_t_idx_cache = next_idx / (cfg.w_top_k + cfg.r_top_k) % cfg.T;
        }
        else {
            next_val_cache = 0; // nop-node
        }
    };
    // pre-fetch the data of next event
    const auto pre_fetch_data = [&](const int_t rank) {
        const acc_t next_val = rank < warp_size ? warp.shfl(val_cache, rank) : warp.shfl(next_val_cache, rank - warp_size);
        const int_t next_t_idx = rank < warp_size ? warp.shfl(t_idx_cache, rank) : warp.shfl(next_t_idx_cache, rank - warp_size);

        if (next_val > 0) {
            next_cache.load(x_ptr + in_stride.x.offset(seg_info.b_idx, seg_info.h_idx, next_t_idx), cfg.C, lane_idx);
            if (lane_idx == 0)
                next_gamma_cache = gamma_ptr[in_stride.gamma.offset(seg_info.b_idx, seg_info.h_idx, next_t_idx)];
        }
    };
    // hint the L1 cache of next event
    const auto ptx_L1_hint = [&](const int_t rank) {
        const acc_t next_val = rank < warp_size ? warp.shfl(val_cache, rank) : warp.shfl(next_val_cache, rank - warp_size);
        const int_t next_t_idx = rank < warp_size ? warp.shfl(t_idx_cache, rank) : warp.shfl(next_t_idx_cache, rank - warp_size);

        if (next_val > 0)
            ptx_prefetch_global_l1<c_loop * warp_size * sizeof(scalar_t)>(x_ptr + in_stride.x.offset(seg_info.b_idx, seg_info.h_idx, next_t_idx), cfg.C * sizeof(scalar_t), lane_idx);
    };

    update_meta(seg_info.seg_off & coalesced_align_mask);
    for (int_t s = 0; s < ptx_prefetch; ++s)
        ptx_L1_hint(warp_size + s);
    pre_fetch_data(warp_size);
    for (int_t chunk_off = seg_info.seg_off & coalesced_align_mask; chunk_off < seg_info.seg_end; chunk_off += warp_size) {
        update_meta(chunk_off + warp_size);
        for (int_t chunk_pos = 0; chunk_pos < warp_size; ++chunk_pos) {
            // read out value from pre-fetch
            const acc_t curr_val = warp.shfl(val_cache, chunk_pos);
            const scalar_vec input_cache = next_cache;
            const scalar_t gamma_cache = next_gamma_cache;

            // pre-fetch for next iteration
            if constexpr (ptx_prefetch > 0)
                ptx_L1_hint(chunk_pos + ptx_prefetch);
            pre_fetch_data(chunk_pos + 1);

            if (curr_val == 0) continue; // val == 0: nop node

            if (const acc_t w = +curr_val; w > 0) {
                // write node
                acc_vec x_val = input_cache.template cast<acc_t>();
                if constexpr (IS_BWD) {
                    acc_t *const __restrict__ event_slot = inter_state_ptr.event_slot; // (N_events, C)
                    const int_t seq_pos = chunk_off + chunk_pos;
                    // save previous slot and mass for writing backward
                    (x_val - slot).store(event_slot + static_cast<uint64_t>(seq_pos) * static_cast<uint64_t>(cfg.C), cfg.C, lane_idx);
                    if (lane_idx == chunk_pos)
                        val_cache = mass; // re-use the reg: val_cache stores previous mass for lane_idx <= chunk_pos, and keeps r/w value for lane_idx > chunk_pos
                }

                // write update
                const acc_t gamma = warp.shfl(type_cast<scalar_t, acc_t>(gamma_cache), 0);
                const auto [gain, new_mass] = get_gain<acc_t>(w, gamma, mass, cfg.eps);
                mass = new_mass;
                slot.lerp_(gain, x_val);
            }
            if (const acc_t r = -curr_val; r > 0) {
                // read node
                if constexpr (IS_FWD || MASS_AWARE) {
                    const int_t t_idx = warp.shfl(t_idx_cache, chunk_pos);
                    // accumulate the read value on the output (cross K)
                    acc_vec out = slot;
                    if constexpr (MASS_AWARE) {
                        out *= r * mass;
                        if (lane_idx == chunk_pos) {
                            const int_t idx = seg_info.bh_idx * cfg.T + t_idx;
                            atomicAdd(read_sums.sum_r + idx, r);
                            atomicAdd(read_sums.sum_rm + idx, r * mass);
                        }
                    }
                    else
                        out *= r;
                    out.atomic_add(global_y_ptr + y_stride.offset(seg_info.b_idx, seg_info.h_idx, t_idx), cfg.C, lane_idx);
                }
                if constexpr (IS_BWD && MASS_AWARE)
                    if (lane_idx == chunk_pos)
                        val_cache = mass;
            }
        }

        // coalesced store the mass event
        if constexpr (IS_BWD) {
            acc_t *const __restrict__ event_mass = inter_state_ptr.event_mass; // (N_events)
            if (const int_t evt_idx = chunk_off + lane_idx; seg_info.seg_off <= evt_idx && evt_idx < seg_info.seg_end)
                event_mass[evt_idx] = val_cache;
        }
    }

    // store the memory state after
    if constexpr (IS_FWD) {
        slot.store(post_slot, cfg.C, lane_idx);
        if (lane_idx == 0)
            *post_mass = mass;
    }
    else
        slot.store(inter_state_ptr.post_slot + static_cast<uint64_t>(s_idx) * static_cast<uint64_t>(cfg.C), cfg.C, lane_idx);
}

template<typename scalar_t, typename acc_t, int_t c_loop, int_t warp_per_block, bool MASS_AWARE, bool native_vec = false, unsigned ptx_prefetch = 8>
__global__ void topk_segment_backward_r2l_kernel(
    const SegmentMetaPtr<acc_t> seg_meta_ptr, //                r

    const InPackPtr<scalar_t, acc_t> input_ptr, //              r  (B, H, T, [C])

    const SegmentInterStatePtr<acc_t, true> inter_state_ptr, // r, backward replay cache

    const scalar_t *const __restrict__ global_D_y, //           r  (B, H, T, C)
    const ReadSumsPtr<acc_t, true> D_read_sums, //              r  (B, H, T), optional

    acc_t *const __restrict__ global_D_addr_val, //             w  (B, H, Kw + Kr, T)
    const InPackPtr<acc_t, acc_t, false> D_input_ptr, //       r/w (B, H, T, [C])
    const StatePtr<acc_t> D_state_ptr, //                      r/w (B, H, A, [C])

    const int_t s_size,
    const InputStride in_stride,
    const StateStride D_state_stride,
    const InputStride Din_stride,
    const ValueStride Dy_stride,

    const SegmentConfig cfg
) {
    static_assert(ptx_prefetch < warp_size);
    using acc_vec = Vec<acc_t, c_loop, native_vec>;
    using scalar_vec = Vec<scalar_t, c_loop, native_vec>;
    constexpr int_t coalesced_align_mask = ~(warp_size - 1);

    // get index of current warp
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<warp_size>(block);
    const int_t warp_idx = block.thread_rank() / warp.size();
    const int_t lane_idx = warp.thread_rank();
    const int_t s_idx = blockIdx.x * warp_per_block + warp_idx;
    if (s_idx >= s_size || seg_meta_ptr.seg_len[s_idx] == 0) return;

    // load the segment information
    const acc_t *const __restrict__ event_v_ptr = seg_meta_ptr.event_v;
    const int_t *const __restrict__ event_i_ptr = seg_meta_ptr.event_i;
    const segment_info seg_info{seg_meta_ptr, s_idx, cfg};

    const scalar_t *const __restrict__ gamma_ptr = input_ptr.gamma; // (B, H, T)
    const acc_t *const __restrict__ post_slot_state = inter_state_ptr.post_slot + static_cast<uint64_t>(s_idx) * static_cast<uint64_t>(cfg.C); // (C)
    acc_t *const __restrict__ D_slot_state = D_state_ptr.slot + D_state_stride.slot.offset(seg_info.b_idx, seg_info.h_idx, seg_info.a_idx); // (C)
    acc_t *const __restrict__ D_mass_state = D_state_ptr.mass + D_state_stride.mass.offset(seg_info.b_idx, seg_info.h_idx, seg_info.a_idx); // ()
    const acc_t *const __restrict__ event_slot = inter_state_ptr.event_slot; // (N_events, C)
    const acc_t *const __restrict__ event_mass = inter_state_ptr.event_mass; // (N_events)
    acc_t *const __restrict__ local_D_w = global_D_addr_val + seg_info.bh_idx * (cfg.w_top_k + cfg.r_top_k) * cfg.T; // (K, T)
    acc_t *const __restrict__ local_D_r = global_D_addr_val + seg_info.bh_idx * (cfg.w_top_k + cfg.r_top_k) * cfg.T + cfg.w_top_k * cfg.T; // (K, T)
    acc_t *const __restrict__ D_x_ptr = D_input_ptr.x; // (B, H, T, C)
    acc_t *const __restrict__ D_gamma_ptr = D_input_ptr.gamma; // (B, H, T)

    // load the post-state primal slot, then roll it back during the R2L scan
    acc_vec slot(post_slot_state, cfg.C, lane_idx);

    // load the gradient on the post state
    acc_vec D_slot(D_slot_state, cfg.C, lane_idx);
    acc_t D_mass = *D_mass_state;

    // pre-fetch data
    scalar_t next_gamma_cache;
    acc_vec next_event_slot_cache;
    scalar_vec next_Dy_cache;
    // pre-fetch meta
    acc_t val_cache, next_val_cache;
    acc_t mass_cache, next_mass_cache;
    int_t t_idx_cache, next_t_idx_cache;
    int_t k_idx_cache, next_k_idx_cache;

    // update the metadata (val, idx) of next chunk
    const auto update_meta = [&](const int_t off) {
        // update current
        val_cache = next_val_cache;
        mass_cache = next_mass_cache;
        t_idx_cache = next_t_idx_cache;
        k_idx_cache = next_k_idx_cache;

        // load next chunk
        if (const int_t seq_pos = off + lane_idx; seg_info.seg_off <= seq_pos && seq_pos < seg_info.seg_end) {
            next_val_cache = event_v_ptr[seq_pos];
            next_mass_cache = MASS_AWARE || next_val_cache > 0 ? event_mass[seq_pos] : 0;
            const int_t next_idx = event_i_ptr[seq_pos];
            next_t_idx_cache = next_idx / (cfg.w_top_k + cfg.r_top_k) % cfg.T;
            next_k_idx_cache = next_idx % (cfg.w_top_k + cfg.r_top_k);
        }
        else
            next_val_cache = 0; // nop-node
    };
    // pre-fetch the data of next event
    const auto pre_fetch_data = [&](const int_t off, const int_t rank) {
        const acc_t next_val = rank >= 0 ? warp.shfl(val_cache, rank) : warp.shfl(next_val_cache, warp_size + rank);
        const int_t next_t_idx = rank >= 0 ? warp.shfl(t_idx_cache, rank) : warp.shfl(next_t_idx_cache, warp_size + rank);
        if (next_val > 0) {
            next_event_slot_cache.load(event_slot + static_cast<uint64_t>(off + rank) * static_cast<uint64_t>(cfg.C), cfg.C, lane_idx, 0);
            if (lane_idx == 0)
                next_gamma_cache = gamma_ptr[in_stride.gamma.offset(seg_info.b_idx, seg_info.h_idx, next_t_idx)];
        }
        if (next_val < 0)
            next_Dy_cache.load(global_D_y + Dy_stride.offset(seg_info.b_idx, seg_info.h_idx, next_t_idx), cfg.C, lane_idx, 0);
    };
    // hint the L1 cache of next event
    const auto ptx_L1_hint = [&](const int_t off, const int_t rank) {
        const acc_t next_val = rank >= 0 ? warp.shfl(val_cache, rank) : warp.shfl(next_val_cache, warp_size + rank);
        const int_t next_t_idx = rank >= 0 ? warp.shfl(t_idx_cache, rank) : warp.shfl(next_t_idx_cache, warp_size + rank);
        if (next_val > 0)
            ptx_prefetch_global_l1<c_loop * warp_size * sizeof(acc_t)>(event_slot + static_cast<uint64_t>(off + rank) * static_cast<uint64_t>(cfg.C), cfg.C * sizeof(acc_t), lane_idx);
        if (next_val < 0)
            ptx_prefetch_global_l1<c_loop * warp_size * sizeof(scalar_t)>(global_D_y + Dy_stride.offset(seg_info.b_idx, seg_info.h_idx, next_t_idx), cfg.C * sizeof(scalar_t), lane_idx);
    };

    const int_t seg_end_aligned = (seg_info.seg_end - 1) & coalesced_align_mask;
    update_meta(seg_end_aligned);
    for (int_t s = 0; s < ptx_prefetch; ++s)
        ptx_L1_hint(seg_end_aligned + warp_size, -1 - s);
    pre_fetch_data(seg_end_aligned + warp_size, -1);
    for (int_t chunk_off = seg_end_aligned; seg_info.seg_off < chunk_off + warp_size; chunk_off -= warp_size) {
        update_meta(chunk_off - warp_size);
        for (int_t chunk_pos = warp_size - 1; chunk_pos >= 0; --chunk_pos) {
            // read out value from pre-fetch
            const acc_t curr_val = warp.shfl(val_cache, chunk_pos);
            const acc_vec event_slot_cache = next_event_slot_cache;
            const scalar_vec Dy_cache = next_Dy_cache;
            const scalar_t gamma_cache = next_gamma_cache;

            // pre-fetch data for next iteration
            if constexpr (ptx_prefetch > 0)
                ptx_L1_hint(chunk_off, chunk_pos - ptx_prefetch);
            pre_fetch_data(chunk_off, chunk_pos - 1);

            if (curr_val == 0) continue; // nop node

            if (const acc_t w = +curr_val; w > 0) {
                // write node: (D_slot, D_mass) ==> (D_w, D_x)
                const int_t t_idx = warp.shfl(t_idx_cache, chunk_pos);
                const int_t k_idx = warp.shfl(k_idx_cache, chunk_pos);

                // gradient of inputs
                acc_t D_gain = D_slot.dot(event_slot_cache); // event_slot_cache == x - prev_slot
                D_gain = cg::reduce(warp, D_gain, cg::plus<acc_t>()); // sum D_gain along C

                const acc_t gamma = warp.shfl(type_cast<scalar_t, acc_t>(gamma_cache), 0);
                const acc_t prev_mass = warp.shfl(mass_cache, chunk_pos);
                const auto [gain, D_w_term, D_gamma_term, D_prev_mass] = get_Dw_Dgamma_Dmass<acc_t>(D_gain, D_mass, w, gamma, prev_mass, cfg.eps);

                acc_vec D_x = D_slot * gain;
                D_x.atomic_add(D_x_ptr + Din_stride.x.offset(seg_info.b_idx, seg_info.h_idx, t_idx), cfg.C, lane_idx);

                // only the one lane write
                if (lane_idx == chunk_pos) {
                    local_D_w[k_idx * cfg.T + t_idx] = D_w_term;
                    atomicAdd(&D_gamma_ptr[Din_stride.gamma.offset(seg_info.b_idx, seg_info.h_idx, t_idx)], D_gamma_term);
                }

                // gradient of previous states
                D_slot *= 1 - gain;
                D_mass = D_prev_mass;
                slot -= event_slot_cache * gain;
            }
            if (const acc_t r = -curr_val; r > 0) {
                // read node: accumulate the gradient of output on D_slot
                const int_t t_idx = warp.shfl(t_idx_cache, chunk_pos);
                const int_t k_idx = warp.shfl(k_idx_cache, chunk_pos) - cfg.w_top_k;

                acc_vec Dy = Dy_cache.template cast<acc_t>();

                // sum D_r along C
                acc_t D_r_term = slot.dot(Dy);
                D_r_term = cg::reduce(warp, D_r_term, cg::plus<acc_t>());
                if constexpr (MASS_AWARE) {
                    const int_t idx = seg_info.bh_idx * cfg.T + t_idx;
                    const acc_t mass = warp.shfl(mass_cache, chunk_pos);
                    // The launcher differentiates the normalization with eps before this scan.
                    D_slot += Dy * (r * mass);
                    D_mass += r * (D_r_term + D_read_sums.sum_rm[idx]);
                    D_r_term = mass * (D_r_term + D_read_sums.sum_rm[idx]) + D_read_sums.sum_r[idx];
                }
                else
                    D_slot += Dy * r;
                if (lane_idx == chunk_pos)
                    local_D_r[k_idx * cfg.T + t_idx] = D_r_term;
            }
        }
    }

    // save the gradient on the init_state
    D_slot.store(D_slot_state, cfg.C, lane_idx);
    if (lane_idx == 0)
        *D_mass_state = D_mass;
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t c_loop, int_t warp_per_block, bool native_vec = false>
inline void topk_segment_forward_launcher(
    const SegmentMeta &seg_meta, // r

    const InPack &inputs, //        r  (B, H, T, [C])

    const State &init_state, //     r  (B, H, A, [C])

    const torch_api::Tensor &ys, //    r/w (B, H, T, C)
    const State &post_state, //     w (B, H, A, [C])

    const SegmentConfig &cfg
) {
    using scalar_t = torch_api::ScalarTypeToCPPTypeT<scalar_dtype>;
    using acc_t = torch_api::ScalarTypeToCPPTypeT<acc_dtype>;

    RAMNET_CHECK(0 < cfg.w_top_k, "Expect 0 < w_top_k but got ", cfg.w_top_k);
    RAMNET_CHECK(0 < cfg.r_top_k, "Expect 0 < r_top_k but got ", cfg.w_top_k);
    RAMNET_CHECK(cfg.C <= c_loop * warp_size, "Expect C <= ", c_loop * warp_size, " but got ", cfg.C);
    check_segment_meta(seg_meta, acc_dtype, cfg);
    check_tensor("x", inputs.x, scalar_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.T, cfg.C});
    check_tensor("gamma", inputs.gamma, scalar_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.T});
    check_tensor("init_slot", init_state.slot, acc_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.A, cfg.C});
    check_tensor("init_mass", init_state.mass, acc_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.A});
    check_tensor("post_slot", post_state.slot, acc_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.A, cfg.C});
    check_tensor("post_mass", post_state.mass, acc_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.A});
    check_tensor("ys", ys, acc_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.T, cfg.C});

    const InputStride in_stride{inputs};
    const StateStride init_state_stride{init_state};
    const StateStride post_state_stride{post_state};
    const ValueStride y_stride{ys};

    const auto device = inputs.x.device();
    torch_api::CUDAGuard device_guard(device);
    const auto stream = torch_api::current_cuda_stream(device);
    const torch_api::CUDAStreamGuard stream_guard(stream, device);

    const auto s_size = static_cast<int_t>(seg_meta.seg_idx.size(0));
    if (s_size == 0)
        return;
    const int_t blocks = cdiv(s_size, warp_per_block);

    ReadSums read_sums;
    if (cfg.mass_aware)
        read_sums = ReadSums(cfg, acc_dtype, device);

    const auto kernel = cfg.mass_aware
                            ? topk_segment_l2r_kernel<scalar_t, acc_t, c_loop, warp_per_block, false, true, native_vec>
                            : topk_segment_l2r_kernel<scalar_t, acc_t, c_loop, warp_per_block, false, false, native_vec>;
    kernel<<<blocks, warp_per_block * warp_size, 0, stream>>>(
        SegmentMetaPtr<acc_t>{seg_meta},
        InPackPtr<scalar_t, acc_t>{inputs},
        StatePtr<acc_t, true>{init_state},
        torch_api::data_ptr<acc_t>(ys), StatePtr<acc_t>{post_state},
        cfg.mass_aware ? ReadSumsPtr<acc_t>{read_sums} : ReadSumsPtr<acc_t>{}, SegmentInterStatePtr<acc_t>{},
        s_size, in_stride, init_state_stride, post_state_stride, y_stride,
        cfg
    );
    RAMNET_CUDA_KERNEL_LAUNCH_CHECK();
    if (cfg.mass_aware)
        read_sums.normalize_(ys, cfg.eps);
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t c_loop, int_t warp_per_block, bool native_vec = false>
inline void topk_segment_backward_launcher(
    const SegmentMeta &seg_meta, //     r

    const InPack &inputs, //            r  (B, H, T, [C])

    const State &init_state, //         r  (B, H, A, [C])

    const torch_api::Tensor &D_y, //        r  (B, H, T, C)

    const torch_api::Tensor &D_addr_val, // w  (B, H, Kw + Kr, T)
    const InPack &D_inputs, //         r/w (B, H, T, [C])
    const State &D_state, //           r/w (B, H, A, [C])

    const SegmentConfig &cfg
) {
    using scalar_t = torch_api::ScalarTypeToCPPTypeT<scalar_dtype>;
    using acc_t = torch_api::ScalarTypeToCPPTypeT<acc_dtype>;

    RAMNET_CHECK(0 < cfg.w_top_k, "Expect 0 < w_top_k but got ", cfg.w_top_k);
    RAMNET_CHECK(0 < cfg.r_top_k, "Expect 0 < r_top_k but got ", cfg.w_top_k);
    RAMNET_CHECK(cfg.C <= c_loop * warp_size, "Expect C <= ", c_loop * warp_size, " but got ", cfg.C);
    check_segment_meta(seg_meta, acc_dtype, cfg);
    check_tensor("x", inputs.x, scalar_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.T, cfg.C});
    check_tensor("gamma", inputs.gamma, scalar_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.T});
    check_tensor("init_slot", init_state.slot, acc_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.A, cfg.C});
    check_tensor("init_mass", init_state.mass, acc_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.A});
    check_tensor("D_slot", D_state.slot, acc_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.A, cfg.C});
    check_tensor("D_mass", D_state.mass, acc_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.A});
    check_tensor("D_y", D_y, scalar_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.T, cfg.C});
    check_tensor("D_addr_val", D_addr_val, acc_dtype, CheckContiguity::Whole, {cfg.B, cfg.H, cfg.w_top_k + cfg.r_top_k, cfg.T});
    check_tensor("D_x", D_inputs.x, acc_dtype, CheckContiguity::LastDim, {cfg.B, cfg.H, cfg.T, cfg.C});
    check_tensor("D_gamma", D_inputs.gamma, acc_dtype, CheckContiguity::None, {cfg.B, cfg.H, cfg.T});

    const InputStride in_stride{inputs};
    const StateStride state_stride{init_state};
    const StateStride D_state_stride{D_state};
    const InputStride Din_stride{D_inputs};

    const auto device = inputs.x.device();
    torch_api::CUDAGuard device_guard(device);
    const auto stream = torch_api::current_cuda_stream(device);
    const torch_api::CUDAStreamGuard stream_guard(stream, device);

    const auto s_size = static_cast<int_t>(seg_meta.seg_idx.size(0));
    if (s_size == 0)
        return;
    const int_t blocks = cdiv(s_size, warp_per_block);

    // event_slot/event_mass use the full event index space for writes and mass-aware read masses.
    // This keeps seq_pos as a direct index in both scans and
    // avoids a write compaction map plus the extra remapping/gather overhead in the backward hot path.
    const SegmentInterState inter_state{
        torch_api::empty({seg_meta.event_i.numel(), cfg.C}, acc_dtype, device),
        torch_api::empty({seg_meta.event_i.numel()}, acc_dtype, device),
        torch_api::empty({s_size, cfg.C}, acc_dtype, device),
    };

    torch_api::Tensor y;
    ReadSums read_sums;
    if (cfg.mass_aware) {
        // Reuse the existing L2R replay to reconstruct normalization intermediates.
        y = torch_api::zeros({cfg.B, cfg.H, cfg.T, cfg.C}, acc_dtype, device);
        read_sums = ReadSums(cfg, acc_dtype, device);
    }
    const auto l2r_kernel = cfg.mass_aware
                                ? topk_segment_l2r_kernel<scalar_t, acc_t, c_loop, warp_per_block, true, true, native_vec>
                                : topk_segment_l2r_kernel<scalar_t, acc_t, c_loop, warp_per_block, true, false, native_vec>;
    l2r_kernel<<<blocks, warp_per_block * warp_size, 0, stream>>>(
        SegmentMetaPtr<acc_t>{seg_meta},
        InPackPtr<scalar_t, acc_t>{inputs},
        StatePtr<acc_t, true>{init_state},
        cfg.mass_aware ? torch_api::data_ptr<acc_t>(y) : nullptr, StatePtr<acc_t>{},
        cfg.mass_aware ? ReadSumsPtr<acc_t>{read_sums} : ReadSumsPtr<acc_t>{}, SegmentInterStatePtr<acc_t>{inter_state},
        s_size, in_stride, state_stride, state_stride, cfg.mass_aware ? ValueStride{y} : ValueStride{},
        cfg
    );
    RAMNET_CUDA_KERNEL_LAUNCH_CHECK();

    const torch_api::Tensor D_read = cfg.mass_aware ? read_sums.backward_(y, D_y, cfg.eps) : D_y;
    const auto r2l_kernel = cfg.mass_aware
                                ? topk_segment_backward_r2l_kernel<scalar_t, acc_t, c_loop, warp_per_block, true, native_vec>
                                : topk_segment_backward_r2l_kernel<scalar_t, acc_t, c_loop, warp_per_block, false, native_vec>;
    r2l_kernel<<<blocks, warp_per_block * warp_size, 0, stream>>>(
        SegmentMetaPtr<acc_t>{seg_meta},
        InPackPtr<scalar_t, acc_t>{inputs},
        SegmentInterStatePtr<acc_t, true>{inter_state},
        torch_api::data_ptr<scalar_t, true>(D_read), cfg.mass_aware ? ReadSumsPtr<acc_t, true>{read_sums} : ReadSumsPtr<acc_t, true>{},
        torch_api::data_ptr<acc_t>(D_addr_val), InPackPtr<acc_t, acc_t, false>{D_inputs}, StatePtr<acc_t>(D_state),
        s_size, in_stride, D_state_stride, Din_stride, ValueStride{D_read},
        cfg
    );
    RAMNET_CUDA_KERNEL_LAUNCH_CHECK();
}


template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t warp_per_block>
inline void topk_segment_forward_size_dispatcher(
    const SegmentMeta &seg_meta, // r

    const InPack &inputs, //        r  (B, H, T, [C])

    const State &init_state, //     r  (B, H, A, [C])

    const torch_api::Tensor &ys, //    r/w (B, H, T, C)
    const State &post_state, //     w  (B, H, A, [C])

    const SegmentConfig &cfg
) {
    constexpr static struct {
        bool (*check)(int_t);

        decltype(&topk_segment_forward_launcher<scalar_dtype, acc_dtype, 1, warp_per_block>) func;
    } kernels[] = {
        {[](const int_t c) -> bool { return c <= 32; }, topk_segment_forward_launcher<scalar_dtype, acc_dtype, 1, warp_per_block>},
        {[](const int_t c) -> bool { return c <= 64 && c % 2 == 0; }, topk_segment_forward_launcher<scalar_dtype, acc_dtype, 2, warp_per_block, true>},
        {[](const int_t c) -> bool { return c <= 64; }, topk_segment_forward_launcher<scalar_dtype, acc_dtype, 2, warp_per_block>},
        {[](const int_t c) -> bool { return c <= 128; }, topk_segment_forward_launcher<scalar_dtype, acc_dtype, 4, warp_per_block>},
    };

    for (const auto &[check, kernel]: kernels)
        if (check(cfg.C)) {
            kernel(
                seg_meta,
                inputs,
                init_state,
                ys, post_state,
                cfg
            );
            return;
        }
    RAMNET_CHECK(false, "No matching kernel found for C=", cfg.C)
}

void topk_segment_forward_dispatcher(
    const SegmentMeta &seg_meta, // r

    const InPack &inputs, //        r  (B, H, T, [C])

    const State &init_state, //     r  (B, H, A, [C])

    const torch_api::Tensor &ys, //    r/w (B, H, T, C)
    const State &post_state, //     w  (B, H, A, [C])

    const SegmentConfig &cfg
) {
    constexpr int_t wpb = 2;

    decltype(&topk_segment_forward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float, wpb>) kernel = nullptr;
    // ReSharper disable once CppIncompleteSwitchStatement
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (inputs.x.scalar_type()) {
        case torch_api::ScalarType::Half:
            kernel = topk_segment_forward_size_dispatcher<torch_api::ScalarType::Half, torch_api::ScalarType::Float, wpb>;
            break;
        case torch_api::ScalarType::BFloat16:
            kernel = topk_segment_forward_size_dispatcher<torch_api::ScalarType::BFloat16, torch_api::ScalarType::Float, wpb>;
            break;
        case torch_api::ScalarType::Float:
            kernel = topk_segment_forward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float, wpb>;
            break;
    }

    RAMNET_CHECK(kernel!=nullptr, "Unsupported scalar type: ", inputs.x.scalar_type());
    kernel(
        seg_meta,
        inputs,
        init_state,
        ys, post_state,
        cfg
    );
}

template<torch_api::ScalarType scalar_dtype, torch_api::ScalarType acc_dtype, int_t warp_per_block>
inline void topk_segment_backward_size_dispatcher(
    const SegmentMeta &seg_meta, //     r

    const InPack &inputs, //            r  (B, H, T, [C])

    const State &init_state, //         r  (B, H, A, [C])

    const torch_api::Tensor &D_y, //        r  (B, H, T, C)

    const torch_api::Tensor &D_addr_val, // w  (B, H, Kw + Kr, T)
    const InPack &D_inputs, //         r/w (B, H, T, [C])
    const State &D_state, //           r/w (B, H, A, [C])

    const SegmentConfig &cfg
) {
    constexpr static struct {
        bool (*check)(int_t);

        decltype(&topk_segment_backward_launcher<scalar_dtype, acc_dtype, 1, warp_per_block>) func;
    } kernels[] = {
        {[](const int_t c) -> bool { return c <= 32; }, topk_segment_backward_launcher<scalar_dtype, acc_dtype, 1, warp_per_block>},
        {[](const int_t c) -> bool { return c <= 64 && c % 2 == 0; }, topk_segment_backward_launcher<scalar_dtype, acc_dtype, 2, warp_per_block, true>},
        {[](const int_t c) -> bool { return c <= 64; }, topk_segment_backward_launcher<scalar_dtype, acc_dtype, 2, warp_per_block>},
        {[](const int_t c) -> bool { return c <= 128; }, topk_segment_backward_launcher<scalar_dtype, acc_dtype, 4, warp_per_block>},
    };

    for (const auto &[check, kernel]: kernels)
        if (check(cfg.C)) {
            kernel(
                seg_meta,
                inputs,
                init_state,
                D_y,
                D_addr_val, D_inputs, D_state,
                cfg
            );
            return;
        }
    RAMNET_CHECK(false, "No matching kernel found for C=", cfg.C)
}

void topk_segment_backward_dispatcher(
    const SegmentMeta &seg_meta, //     r

    const InPack &inputs, //            r (B, H, T, [C])

    const State &init_state, //         r (B, H, A, [C])

    const torch_api::Tensor &D_y, //        r (B, H, T, C)

    const torch_api::Tensor &D_addr_val, // w (B, H, Kw + Kr, T)
    const InPack &D_inputs, //         r/w (B, H, T, [C])
    const State &D_state, //           r/w (B, H, A, [C])

    const SegmentConfig &cfg
) {
    constexpr int_t wpb = 2;
    decltype(&topk_segment_backward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float, wpb>) kernel = nullptr;
    // ReSharper disable once CppIncompleteSwitchStatement
    // ReSharper disable once CppDefaultCaseNotHandledInSwitchStatement
    switch (inputs.x.scalar_type()) {
        case torch_api::ScalarType::Half:
            kernel = topk_segment_backward_size_dispatcher<torch_api::ScalarType::Half, torch_api::ScalarType::Float, wpb>;
            break;
        case torch_api::ScalarType::BFloat16:
            kernel = topk_segment_backward_size_dispatcher<torch_api::ScalarType::BFloat16, torch_api::ScalarType::Float, wpb>;
            break;
        case torch_api::ScalarType::Float:
            kernel = topk_segment_backward_size_dispatcher<torch_api::ScalarType::Float, torch_api::ScalarType::Float, wpb>;
            break;
    }

    RAMNET_CHECK(kernel!=nullptr, "Unsupported scalar type: ", inputs.x.scalar_type());
    kernel(
        seg_meta,
        inputs,
        init_state,
        D_y,
        D_addr_val, D_inputs, D_state,
        cfg
    );
}
