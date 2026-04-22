#pragma once

#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int HIDDEN_SIZE       = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS       = 16;
constexpr int NUM_KV_HEADS      = 8;
constexpr int HEAD_DIM          = 128;
constexpr int Q_SIZE            = NUM_Q_HEADS * HEAD_DIM;
constexpr int KV_SIZE           = NUM_KV_HEADS * HEAD_DIM;
constexpr int VOCAB_SIZE        = 151936;
constexpr float RMS_NORM_EPS    = 1e-6f;

// ── Batch support ─────────────────────────────────────────────────────────────
// The kernel loops internally over [0, batch_size) for every activation buffer.
// Weights are NEVER replicated — they are loaded once and reused across the batch.
// Cooperative grid size stays fixed at LDG_NUM_BLOCKS regardless of batch size.
constexpr int MAX_BATCH_SIZE = 8;

constexpr int LDG_NUM_BLOCKS  = 18;
constexpr int LDG_BLOCK_SIZE  = 256;
constexpr int WARP_SIZE       = 32;
constexpr int LDG_NUM_WARPS   = LDG_BLOCK_SIZE / WARP_SIZE;

constexpr int LDG_LM_NUM_BLOCKS  = 1184;
constexpr int LDG_LM_BLOCK_SIZE  = 256;

// ── Reduction helpers ─────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < LDG_NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    if (wid == 0 && lane == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < LDG_NUM_WARPS) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }
    __syncthreads();
    if (wid == 0 && lane == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float ldg_silu(float x) {
    return x / (1.0f + expf(-x));
}