/**
 * Batched Fused Decode Kernel
 *
 * Key change vs single-sequence version:
 *   - Every activation buffer has a leading batch dimension
 *   - Weight matrices are loaded ONCE per row, then reused across all B sequences
 *   - KV cache layout: [batch, NUM_KV_HEADS, max_seq_len, HEAD_DIM] per layer
 *   - Cooperative grid size stays fixed (LDG_NUM_BLOCKS) — no relaunch needed
 *   - Bandwidth cost per token: model_size_GB / batch_size  (the whole point)
 *
 * Buffer strides (all activation buffers are flat, batch-major):
 *   hidden_buffer     : [batch, HIDDEN_SIZE]           bf16
 *   g_activations     : [batch, HIDDEN_SIZE]           f32
 *   g_residual        : [batch, HIDDEN_SIZE]           f32
 *   g_normalized      : [batch, HIDDEN_SIZE]           f32
 *   g_q               : [batch, Q_SIZE]                f32
 *   g_k               : [batch, KV_SIZE]               f32
 *   g_v               : [batch, KV_SIZE]               f32
 *   g_attn_out        : [batch, Q_SIZE]                f32
 *   g_mlp_intermediate: [batch, INTERMEDIATE_SIZE]     f32
 *   k_cache / v_cache : [batch, NUM_KV_HEADS, max_seq_len, HEAD_DIM]  bf16  (per layer)
 */

#include "config.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr float BATCHED_RMS_EPS = 1e-6f;

struct BatchedLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
};

// =============================================================================
// Per-sequence RMSNorm helper (called from block 0, sequences processed serially)
// =============================================================================

__device__ void batched_rmsnorm_seq(
    const float* __restrict__ input,     // [HIDDEN_SIZE] for one sequence
    float* __restrict__ residual_out,    // [HIDDEN_SIZE]
    float* __restrict__ normed_out,      // [HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ weight
) {
    __shared__ float smem_sq[LDG_NUM_WARPS];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
        float v = input[i];
        residual_out[i] = v;
        local_sq += v * v;
    }
    local_sq = warp_reduce_sum(local_sq);
    if (lane_id == 0) smem_sq[warp_id] = local_sq;
    __syncthreads();

    if (warp_id == 0) {
        float s = (lane_id < LDG_NUM_WARPS) ? smem_sq[lane_id] : 0.0f;
        s = warp_reduce_sum(s);
        if (lane_id == 0) smem_sq[0] = rsqrtf(s / float(HIDDEN_SIZE) + BATCHED_RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_sq[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
        normed_out[i] = residual_out[i] * rstd * __bfloat162float(__ldg(weight + i));
}

// =============================================================================
// QKV projection  —  weights loaded ONCE per row, reused across batch
// =============================================================================

__device__ void batched_matvec_qkv(
    cg::grid_group& grid,
    int batch_size,
    const __nv_bfloat16* __restrict__ input,        // [batch, HIDDEN_SIZE]  bf16
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ g_normalized,               // [batch, HIDDEN_SIZE]
    float* __restrict__ g_residual,                 // [batch, HIDDEN_SIZE]
    float* __restrict__ q_out,                      // [batch, Q_SIZE]
    float* __restrict__ k_out,                      // [batch, KV_SIZE]
    float* __restrict__ v_out                       // [batch, KV_SIZE]
) {
    int block_id  = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id   = threadIdx.x / WARP_SIZE;
    int lane_id   = threadIdx.x % WARP_SIZE;

    // ── Block 0: RMSNorm for every sequence (serial loop, cheap) ──────────────
    if (block_id == 0) {
        for (int b = 0; b < batch_size; b++) {
            const __nv_bfloat16* in_b  = input         + b * HIDDEN_SIZE;
            float*               res_b = g_residual     + b * HIDDEN_SIZE;
            float*               nor_b = g_normalized   + b * HIDDEN_SIZE;

            // Load bf16 input into g_residual as float
            __shared__ float smem_sq[LDG_NUM_WARPS];
            int warp_id2 = threadIdx.x / WARP_SIZE;
            int lane_id2 = threadIdx.x % WARP_SIZE;

            float local_sq = 0.0f;
            for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
                float v = __bfloat162float(__ldg(in_b + i));
                res_b[i] = v;
                local_sq += v * v;
            }
            local_sq = warp_reduce_sum(local_sq);
            if (lane_id2 == 0) smem_sq[warp_id2] = local_sq;
            __syncthreads();

            if (warp_id2 == 0) {
                float s = (lane_id2 < LDG_NUM_WARPS) ? smem_sq[lane_id2] : 0.0f;
                s = warp_reduce_sum(s);
                if (lane_id2 == 0) smem_sq[0] = rsqrtf(s / float(HIDDEN_SIZE) + BATCHED_RMS_EPS);
            }
            __syncthreads();

            float rstd = smem_sq[0];
            for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
                nor_b[i] = res_b[i] * rstd * __bfloat162float(__ldg(norm_weight + i));

            __syncthreads(); // between sequences
        }
    }

    grid.sync();

    // ── All blocks: QKV projection — key optimisation ─────────────────────────
    // Each warp owns one output row.
    // Weight row loaded once via __ldg into registers → reused for all B seqs.
    // Bandwidth cost: weight_bytes / batch_size per token.

    constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
    int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end   = min(row_start + rows_per_block, TOTAL_ROWS);

    // Registers to cache one weight row (HIDDEN_SIZE / WARP_SIZE = 32 values)
    constexpr int REG_PER_THREAD = HIDDEN_SIZE / WARP_SIZE;
    float w_reg[REG_PER_THREAD];

    for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= row_end) break;

        // Determine which projection this row belongs to
        const __nv_bfloat16* weight_row;
        int out_offset; // which output vector (q/k/v) and which row within it
        int out_stride; // stride between sequences in the output

        if (m < Q_SIZE) {
            weight_row = q_weight + m * HIDDEN_SIZE;
            out_offset = m;
            out_stride = Q_SIZE;
        } else if (m < Q_SIZE + KV_SIZE) {
            weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
            out_offset = m - Q_SIZE;
            out_stride = KV_SIZE;
        } else {
            weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
            out_offset = m - Q_SIZE - KV_SIZE;
            out_stride = KV_SIZE;
        }

        // ── Load weight row into registers (ONE load for all B sequences) ─────
        #pragma unroll
        for (int k = lane_id, j = 0; k < HIDDEN_SIZE; k += WARP_SIZE, j++)
            w_reg[j] = __bfloat162float(__ldg(weight_row + k));

        // ── Dot product reused B times ─────────────────────────────────────────
        for (int b = 0; b < batch_size; b++) {
            const float* norm_b = g_normalized + b * HIDDEN_SIZE;
            float sum = 0.0f;
            #pragma unroll
            for (int k = lane_id, j = 0; k < HIDDEN_SIZE; k += WARP_SIZE, j++)
                sum += w_reg[j] * norm_b[k];
            sum = warp_reduce_sum(sum);

            if (lane_id == 0) {
                float* out_ptr;
                if (m < Q_SIZE)
                    out_ptr = q_out + b * Q_SIZE + out_offset;
                else if (m < Q_SIZE + KV_SIZE)
                    out_ptr = k_out + b * KV_SIZE + out_offset;
                else
                    out_ptr = v_out + b * KV_SIZE + out_offset;
                *out_ptr = sum;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// QK Norm + RoPE + KV Cache  (batched)
// =============================================================================

__device__ void batched_qk_norm_rope_cache(
    cg::grid_group& grid,
    int batch_size,
    float* __restrict__ q,                      // [batch, Q_SIZE]
    float* __restrict__ k,                      // [batch, KV_SIZE]
    const float* __restrict__ v,                // [batch, KV_SIZE]
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,        // [batch, NUM_KV_HEADS, max_seq_len, HEAD_DIM]
    __nv_bfloat16* __restrict__ v_cache,
    int position,
    int max_seq_len
) {
    int block_id  = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id   = threadIdx.x / WARP_SIZE;
    int lane_id   = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

    // KV cache stride per sequence: NUM_KV_HEADS * max_seq_len * HEAD_DIM
    int kv_seq_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    // ── Q heads (norm + RoPE) ─────────────────────────────────────────────────
    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end   = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int b = 0; b < batch_size; b++) {
        float* q_b = q + b * Q_SIZE;

        for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
            float* q_head = q_b + h * HEAD_DIM;

            // RMSNorm
            float sum_sq = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
                sum_sq += q_head[i] * q_head[i];
            sum_sq = warp_reduce_sum(sum_sq);
            float scale = rsqrtf(sum_sq / float(HEAD_DIM) + BATCHED_RMS_EPS);
            scale = __shfl_sync(0xffffffff, scale, 0);

            float q_local[HEAD_DIM / WARP_SIZE];
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
                q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));

            // RoPE
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                float cos_v = __bfloat162float(__ldg(cos_pos + i));
                float sin_v = __bfloat162float(__ldg(sin_pos + i));
                int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
                int pair_idx = i + pair_offset;
                float pair_v = __shfl_sync(0xffffffff, q_local[pair_idx / WARP_SIZE],
                                           pair_idx % WARP_SIZE);
                q_head[i] = (i < HEAD_DIM/2)
                    ? q_local[j] * cos_v - pair_v * sin_v
                    : pair_v * sin_v + q_local[j] * cos_v;
            }
        }
    }

    // ── K heads (norm + RoPE + cache write) ───────────────────────────────────
    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end   = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int b = 0; b < batch_size; b++) {
        float*               k_b        = k + b * KV_SIZE;
        const float*         v_b        = v + b * KV_SIZE;
        __nv_bfloat16*       k_cache_b  = k_cache + b * kv_seq_stride;
        __nv_bfloat16*       v_cache_b  = v_cache + b * kv_seq_stride;

        for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
            float*         k_head       = k_b + h * HEAD_DIM;
            const float*   v_head       = v_b + h * HEAD_DIM;
            __nv_bfloat16* k_cache_head = k_cache_b + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
            __nv_bfloat16* v_cache_head = v_cache_b + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

            float sum_sq = 0.0f;
            for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
                sum_sq += k_head[i] * k_head[i];
            sum_sq = warp_reduce_sum(sum_sq);
            float scale = rsqrtf(sum_sq / float(HEAD_DIM) + BATCHED_RMS_EPS);
            scale = __shfl_sync(0xffffffff, scale, 0);

            float k_local[HEAD_DIM / WARP_SIZE];
            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
                k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));

            #pragma unroll
            for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
                float cos_v = __bfloat162float(__ldg(cos_pos + i));
                float sin_v = __bfloat162float(__ldg(sin_pos + i));
                int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
                int pair_idx = i + pair_offset;
                float pair_v = __shfl_sync(0xffffffff, k_local[pair_idx / WARP_SIZE],
                                           pair_idx % WARP_SIZE);
                float k_final = (i < HEAD_DIM/2)
                    ? k_local[j] * cos_v - pair_v * sin_v
                    : pair_v * sin_v + k_local[j] * cos_v;
                k_head[i]       = k_final;
                k_cache_head[i] = __float2bfloat16(k_final);
                v_cache_head[i] = __float2bfloat16(v_head[i]);
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Attention  (batched — each sequence attends to its own KV cache slice)
// =============================================================================

__device__ void batched_attention(
    cg::grid_group& grid,
    int batch_size,
    const float* __restrict__ q,           // [batch, Q_SIZE]
    const __nv_bfloat16* __restrict__ k_cache,  // [batch, NUM_KV_HEADS, max_seq_len, HEAD_DIM]
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,          // [batch, Q_SIZE]
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    int block_id  = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id   = threadIdx.x / WARP_SIZE;
    int lane_id   = threadIdx.x % WARP_SIZE;

    __shared__ float s_max_score[LDG_NUM_WARPS];
    __shared__ float s_sum_exp[LDG_NUM_WARPS];
    __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

    int kv_seq_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    int heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int head_start = block_id * heads_per_block;
    int head_end   = min(head_start + heads_per_block, NUM_Q_HEADS);

    // Each block processes its assigned Q heads for ALL sequences
    for (int b = 0; b < batch_size; b++) {
        const float*         q_b       = q        + b * Q_SIZE;
        float*               out_b     = attn_out + b * Q_SIZE;
        const __nv_bfloat16* k_cache_b = k_cache  + b * kv_seq_stride;
        const __nv_bfloat16* v_cache_b = v_cache  + b * kv_seq_stride;

        for (int qh = head_start; qh < head_end; qh++) {
            int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
            const float*         q_head  = q_b   + qh * HEAD_DIM;
            float*               out_head = out_b + qh * HEAD_DIM;

            float max_score = -INFINITY;
            float sum_exp   = 0.0f;
            float out_acc[4] = {0.f, 0.f, 0.f, 0.f};

            for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
                const __nv_bfloat16* k_pos = k_cache_b + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
                const __nv_bfloat16* v_pos = v_cache_b + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

                float score = 0.0f;
                for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE)
                    score += q_head[d] * __bfloat162float(__ldg(k_pos + d));
                score = warp_reduce_sum(score) * attn_scale;
                score = __shfl_sync(0xffffffff, score, 0);

                float old_max  = max_score;
                max_score      = fmaxf(max_score, score);
                float exp_diff = expf(old_max - max_score);
                sum_exp        = sum_exp * exp_diff + expf(score - max_score);
                float weight   = expf(score - max_score);

                #pragma unroll
                for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                    out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(__ldg(v_pos + d));
            }

            if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                s_out_acc[warp_id][d] = out_acc[j];
            __syncthreads();

            if (warp_id == 0) {
                float global_max = s_max_score[0];
                for (int w = 1; w < LDG_NUM_WARPS; w++)
                    if (s_max_score[w] > -INFINITY)
                        global_max = fmaxf(global_max, s_max_score[w]);

                float total_sum = 0.0f;
                float final_out[4] = {0.f, 0.f, 0.f, 0.f};

                for (int w = 0; w < LDG_NUM_WARPS; w++) {
                    if (s_max_score[w] > -INFINITY) {
                        float sc = expf(s_max_score[w] - global_max);
                        total_sum += s_sum_exp[w] * sc;
                        #pragma unroll
                        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                            final_out[j] += s_out_acc[w][d] * sc;
                    }
                }
                #pragma unroll
                for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++)
                    out_head[d] = final_out[j] / total_sum;
            }
            __syncthreads();
        }
    }

    grid.sync();
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP  (batched)
// =============================================================================

__device__ void batched_o_proj_postnorm_mlp(
    cg::grid_group& grid,
    int batch_size,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight,
    const float* __restrict__ attn_out,           // [batch, Q_SIZE]
    float* __restrict__ g_residual,               // [batch, HIDDEN_SIZE]
    float* __restrict__ g_activations,            // [batch, HIDDEN_SIZE]
    float* __restrict__ g_mlp_intermediate,       // [batch, INTERMEDIATE_SIZE]
    __nv_bfloat16* __restrict__ hidden_out        // [batch, HIDDEN_SIZE]
) {
    int block_id   = blockIdx.x;
    int num_blocks  = gridDim.x;
    int warp_id    = threadIdx.x / WARP_SIZE;
    int lane_id    = threadIdx.x % WARP_SIZE;

    constexpr int REG_PER_THREAD_H = Q_SIZE / WARP_SIZE;            // 64
    constexpr int REG_PER_THREAD_I = INTERMEDIATE_SIZE / WARP_SIZE; // 96

    // ── O Projection + Residual  (weights loaded once, applied to all seqs) ──
    int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
    int hid_start = block_id * hid_per_block;
    int hid_end   = min(hid_start + hid_per_block, HIDDEN_SIZE);

    float w_reg_o[REG_PER_THREAD_H];

    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= hid_end) break;

        const __nv_bfloat16* o_row = o_weight + m * Q_SIZE;

        // Load O weight row once
        #pragma unroll
        for (int k = lane_id, j = 0; k < Q_SIZE; k += WARP_SIZE, j++)
            w_reg_o[j] = __bfloat162float(__ldg(o_row + k));

        // Apply to all sequences
        for (int b = 0; b < batch_size; b++) {
            const float* attn_b = attn_out + b * Q_SIZE;
            float sum = 0.0f;
            #pragma unroll
            for (int k = lane_id, j = 0; k < Q_SIZE; k += WARP_SIZE, j++)
                sum += w_reg_o[j] * attn_b[k];
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                g_activations[b * HIDDEN_SIZE + m] = sum + g_residual[b * HIDDEN_SIZE + m];
        }
    }

    grid.sync();

    // ── Post-attention RMSNorm  (block 0, serial over batch) ─────────────────
    if (block_id == 0) {
        for (int b = 0; b < batch_size; b++) {
            float* act_b = g_activations + b * HIDDEN_SIZE;
            float* res_b = g_residual    + b * HIDDEN_SIZE;

            __shared__ float smem_sq[LDG_NUM_WARPS];
            int warp_id2 = threadIdx.x / WARP_SIZE;
            int lane_id2 = threadIdx.x % WARP_SIZE;

            float local_sq = 0.0f;
            for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
                float v = act_b[i];
                res_b[i] = v;
                local_sq += v * v;
            }
            local_sq = warp_reduce_sum(local_sq);
            if (lane_id2 == 0) smem_sq[warp_id2] = local_sq;
            __syncthreads();

            if (warp_id2 == 0) {
                float s = (lane_id2 < LDG_NUM_WARPS) ? smem_sq[lane_id2] : 0.0f;
                s = warp_reduce_sum(s);
                if (lane_id2 == 0) smem_sq[0] = rsqrtf(s / float(HIDDEN_SIZE) + BATCHED_RMS_EPS);
            }
            __syncthreads();

            float rstd = smem_sq[0];
            for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
                act_b[i] = res_b[i] * rstd * __bfloat162float(__ldg(post_norm_weight + i));

            __syncthreads();
        }
    }

    grid.sync();

    // ── Gate + Up projections  (weights loaded once, reused across batch) ─────
    int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
    int int_start = block_id * int_per_block;
    int int_end   = min(int_start + int_per_block, INTERMEDIATE_SIZE);

    float w_reg_gate[REG_PER_THREAD_H]; // HIDDEN_SIZE/WARP_SIZE = 32
    float w_reg_up[REG_PER_THREAD_H];

    for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= int_end) break;

        const __nv_bfloat16* gate_row = gate_weight + m * HIDDEN_SIZE;
        const __nv_bfloat16* up_row   = up_weight   + m * HIDDEN_SIZE;

        // Load gate + up rows once
        #pragma unroll
        for (int k = lane_id, j = 0; k < HIDDEN_SIZE; k += WARP_SIZE, j++) {
            w_reg_gate[j] = __bfloat162float(__ldg(gate_row + k));
            w_reg_up[j]   = __bfloat162float(__ldg(up_row   + k));
        }

        for (int b = 0; b < batch_size; b++) {
            const float* act_b = g_activations + b * HIDDEN_SIZE;
            float gate_sum = 0.0f, up_sum = 0.0f;
            #pragma unroll
            for (int k = lane_id, j = 0; k < HIDDEN_SIZE; k += WARP_SIZE, j++) {
                gate_sum += w_reg_gate[j] * act_b[k];
                up_sum   += w_reg_up[j]   * act_b[k];
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum   = warp_reduce_sum(up_sum);
            if (lane_id == 0)
                g_mlp_intermediate[b * INTERMEDIATE_SIZE + m] = ldg_silu(gate_sum) * up_sum;
        }
    }

    grid.sync();

    // ── Down projection + residual ────────────────────────────────────────────
    float w_reg_down[REG_PER_THREAD_I]; // INTERMEDIATE_SIZE/WARP_SIZE = 96

    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;
        if (m >= hid_end) break;

        const __nv_bfloat16* down_row = down_weight + m * INTERMEDIATE_SIZE;

        // Load down row once
        #pragma unroll
        for (int k = lane_id, j = 0; k < INTERMEDIATE_SIZE; k += WARP_SIZE, j++)
            w_reg_down[j] = __bfloat162float(__ldg(down_row + k));

        for (int b = 0; b < batch_size; b++) {
            const float* mlp_b = g_mlp_intermediate + b * INTERMEDIATE_SIZE;
            float sum = 0.0f;
            #pragma unroll
            for (int k = lane_id, j = 0; k < INTERMEDIATE_SIZE; k += WARP_SIZE, j++)
                sum += w_reg_down[j] * mlp_b[k];
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[b * HIDDEN_SIZE + m] =
                    __float2bfloat16(sum + g_residual[b * HIDDEN_SIZE + m]);
        }
    }

    grid.sync();
}

// =============================================================================
// Main batched kernel
// =============================================================================

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
batched_decode_kernel(
    const int* __restrict__ input_token_ids,   // [batch_size]
    const __nv_bfloat16* __restrict__ embed_weight,
    const BatchedLayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,       // [batch, NUM_KV_HEADS, max_seq_len, HEAD_DIM] per layer (layer-strided externally)
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden_buffer, // [batch, HIDDEN_SIZE]
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    int batch_size,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    cg::grid_group grid = cg::this_grid();
    int block_id  = blockIdx.x;
    int num_blocks = gridDim.x;

    // ── Embedding lookup for all sequences ────────────────────────────────────
    for (int b = 0; b < batch_size; b++) {
        int tok = input_token_ids[b];
        const __nv_bfloat16* embed_row = embed_weight + tok * HIDDEN_SIZE;
        __nv_bfloat16* hid_b = hidden_buffer + b * HIDDEN_SIZE;
        for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE;
             i += num_blocks * LDG_BLOCK_SIZE)
            hid_b[i] = __ldg(embed_row + i);
    }
    grid.sync();

    int kv_layer_stride = batch_size * NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const BatchedLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_layer_stride;

        batched_matvec_qkv(
            grid, batch_size,
            hidden_buffer, w.input_layernorm_weight,
            w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
            g_activations, g_residual, g_q, g_k, g_v
        );

        batched_qk_norm_rope_cache(
            grid, batch_size,
            g_q, g_k, g_v,
            w.q_norm_weight, w.k_norm_weight,
            cos_table, sin_table,
            layer_k_cache, layer_v_cache,
            position, max_seq_len
        );

        batched_attention(
            grid, batch_size,
            g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale
        );

        batched_o_proj_postnorm_mlp(
            grid, batch_size,
            w.o_proj_weight, w.post_attn_layernorm_weight,
            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
            hidden_buffer
        );
    }

    // ── Final RMSNorm (block 0, all sequences) ────────────────────────────────
    if (block_id == 0) {
        for (int b = 0; b < batch_size; b++) {
            __nv_bfloat16* hid_b = hidden_buffer + b * HIDDEN_SIZE;
            float*         nor_b = g_normalized  + b * HIDDEN_SIZE;

            __shared__ float smem_sq[LDG_NUM_WARPS];
            int warp_id2 = threadIdx.x / WARP_SIZE;
            int lane_id2 = threadIdx.x % WARP_SIZE;

            float local_sq = 0.0f;
            for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
                float v = __bfloat162float(hid_b[i]);
                nor_b[i] = v;
                local_sq += v * v;
            }
            local_sq = warp_reduce_sum(local_sq);
            if (lane_id2 == 0) smem_sq[warp_id2] = local_sq;
            __syncthreads();

            if (warp_id2 == 0) {
                float s = (lane_id2 < LDG_NUM_WARPS) ? smem_sq[lane_id2] : 0.0f;
                s = warp_reduce_sum(s);
                if (lane_id2 == 0) smem_sq[0] = rsqrtf(s / float(HIDDEN_SIZE) + BATCHED_RMS_EPS);
            }
            __syncthreads();

            float rstd = smem_sq[0];
            for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE)
                nor_b[i] *= rstd * __bfloat162float(__ldg(final_norm_weight + i));

            __syncthreads();
        }
    }
}

// =============================================================================
// LM Head  (batched argmax — one token per sequence)
// =============================================================================

__global__ void batched_lm_head_phase1(
    int batch_size,
    const float* __restrict__ hidden,         // [batch, HIDDEN_SIZE]
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,        // [batch, LDG_LM_NUM_BLOCKS]
    int*   __restrict__ block_max_idxs
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end   = min(row_start + rows_per_block, VOCAB_SIZE);

    // Shared memory: one copy of hidden per sequence would be too large.
    // Instead, stream hidden from global — still coalesced within a warp.

    for (int b = 0; b < batch_size; b++) {
        const float* hid_b = hidden + b * HIDDEN_SIZE;

        float local_max = -INFINITY;
        int   local_idx = -1;

        for (int m = row_start + warp_id; m < row_end;
             m += LDG_LM_BLOCK_SIZE / WARP_SIZE)
        {
            const __nv_bfloat16* w_row = weight + m * HIDDEN_SIZE;
            float sum = 0.0f;
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = __ldg(reinterpret_cast<const uint2*>(w_row + k));
                __nv_bfloat16* wp = reinterpret_cast<__nv_bfloat16*>(&w_u2);
                sum += __bfloat162float(wp[0]) * hid_b[k]   +
                       __bfloat162float(wp[1]) * hid_b[k+1] +
                       __bfloat162float(wp[2]) * hid_b[k+2] +
                       __bfloat162float(wp[3]) * hid_b[k+3];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0 && sum > local_max) { local_max = sum; local_idx = m; }
        }

        // Block-level reduce
        __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
        __shared__ int   warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];
        local_max = __shfl_sync(0xffffffff, local_max, 0);
        local_idx = __shfl_sync(0xffffffff, local_idx, 0);
        if (lane_id == 0) { warp_max[warp_id] = local_max; warp_idx[warp_id] = local_idx; }
        __syncthreads();

        if (warp_id == 0) {
            float mv = (lane_id < LDG_LM_BLOCK_SIZE/WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
            int   mi = (lane_id < LDG_LM_BLOCK_SIZE/WARP_SIZE) ? warp_idx[lane_id] : -1;
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, mv, off);
                int   oi = __shfl_down_sync(0xffffffff, mi, off);
                if (ov > mv) { mv = ov; mi = oi; }
            }
            if (lane_id == 0) {
                block_max_vals[b * gridDim.x + blockIdx.x] = mv;
                block_max_idxs[b * gridDim.x + blockIdx.x] = mi;
            }
        }
        __syncthreads();
    }
}

__global__ void batched_lm_head_phase2(
    int batch_size,
    const float* __restrict__ block_max_vals,  // [batch, num_blocks]
    const int*   __restrict__ block_max_idxs,
    int* __restrict__ output_tokens,           // [batch_size]
    int num_lm_blocks
) {
    int b = blockIdx.x;  // one block per sequence
    if (b >= batch_size) return;

    const float* vals = block_max_vals + b * num_lm_blocks;
    const int*   idxs = block_max_idxs + b * num_lm_blocks;

    __shared__ float s_vals[1024];
    __shared__ int   s_idxs[1024];

    float lmax = -INFINITY;
    int   lidx = -1;
    for (int i = threadIdx.x; i < num_lm_blocks; i += blockDim.x) {
        if (vals[i] > lmax) { lmax = vals[i]; lidx = idxs[i]; }
    }
    s_vals[threadIdx.x] = lmax;
    s_idxs[threadIdx.x] = lidx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && s_vals[threadIdx.x + s] > s_vals[threadIdx.x]) {
            s_vals[threadIdx.x] = s_vals[threadIdx.x + s];
            s_idxs[threadIdx.x] = s_idxs[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output_tokens[b] = s_idxs[0];
}

// =============================================================================
// Launch
// =============================================================================

extern "C" void launch_batched_decode(
    const int* input_token_ids,    // [batch_size] device ptr
    int* output_token_ids,         // [batch_size] device ptr
    const void* embed_weight,
    const BatchedLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,          // [batch, LDG_LM_NUM_BLOCKS]
    void* block_max_idxs,
    int batch_size,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    void* kernel_args[] = {
        (void*)&input_token_ids,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&batch_size,
        (void*)&num_layers,
        (void*)&position,
        (void*)&cache_len,
        (void*)&max_seq_len,
        (void*)&attn_scale,
    };

    cudaLaunchCooperativeKernel(
        (void*)batched_decode_kernel,
        dim3(LDG_NUM_BLOCKS),
        dim3(LDG_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    batched_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        batch_size,
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );

    batched_lm_head_phase2<<<batch_size, 256, 0, stream>>>(
        batch_size,
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_ids,
        LDG_LM_NUM_BLOCKS
    );
}