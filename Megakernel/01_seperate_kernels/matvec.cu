#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum_matvec(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void matvec_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ input,
    int M, int K
) {
    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= M) return;

    const __nv_bfloat16* row_ptr = weight + row * K;
    float sum = 0.0f;

    int k = lane * 4;
    for (; k <= K - 128; k += 128) {
        uint2 packed = __ldg(reinterpret_cast<const uint2*>(row_ptr + k));
        __nv_bfloat16* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
        sum += __bfloat162float(vals[0]) * input[k];
        sum += __bfloat162float(vals[1]) * input[k + 1];
        sum += __bfloat162float(vals[2]) * input[k + 2];
        sum += __bfloat162float(vals[3]) * input[k + 3];
    }

    for (; k < K; k += 128) {
        if (k < K) sum += __bfloat162float(__ldg(row_ptr + k)) * input[k];
        if (k + 1 < K) sum += __bfloat162float(__ldg(row_ptr + k + 1)) * input[k + 1];
        if (k + 2 < K) sum += __bfloat162float(__ldg(row_ptr + k + 2)) * input[k + 2];
        if (k + 3 < K) sum += __bfloat162float(__ldg(row_ptr + k + 3)) * input[k + 3];
    }

    sum = warp_reduce_sum_matvec(sum);

    if (lane == 0)
        output[row] = sum;
}

extern "C" void launch_matvec(
    float* output,
    const __nv_bfloat16* weight,
    const float* input,
    int M, int K
) {
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    matvec_kernel<<<blocks, threads>>>(output, weight, input, M, K);
}