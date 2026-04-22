#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void rmsnorm_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int N, float eps
) {
    __shared__ float shared[32];   // for rtx3050, may change to 32

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = __bfloat162float(input[i]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    if (lane == 0)
        shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < blockDim.x / 32) ? shared[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
       
    __syncthreads();

    float rstd = rsqrtf(shared[0] / (float)N + eps);

    if (warp_id == 0 && lane == 0)
        shared[0] = rstd;
    __syncthreads();

    rstd = shared[0];

    for (int i = tid; i < N; i += blockDim.x) {
        float val = __bfloat162float(input[i]);
        float w = __bfloat162float(weight[i]);
        output[i] = __float2bfloat16(val * rstd * w);
    }
}

extern "C" void launch_rmsnorm(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int N, float eps
) {
    rmsnorm_kernel<<<1, 256>>>(output, input, weight, N, eps);

}