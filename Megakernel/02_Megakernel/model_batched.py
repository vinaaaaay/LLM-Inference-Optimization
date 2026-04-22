import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.cpp_extension import load_inline

MODEL_NAME       = "Qwen/Qwen3-0.6B"
NUM_LAYERS       = 28
HIDDEN_SIZE      = 1024
NUM_KV_HEADS     = 8
HEAD_DIM         = 128
Q_SIZE           = 16 * HEAD_DIM
KV_SIZE          = 8  * HEAD_DIM
INTERMEDIATE_SIZE = 3072
MAX_SEQ_LEN      = 512
LDG_LM_NUM_BLOCKS = 1184

_kernel         = None
_batched_kernel = None


def read_source(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, filename)) as f:
        return f.read()


# ── Single-sequence kernel (unchanged) ───────────────────────────────────────
def compile_kernel():
    global _kernel
    if _kernel is not None:
        return _kernel

    cuda_src = read_source("megakernel.cu")
    cpp_src  = _single_seq_cpp_src()

    here = os.path.dirname(os.path.abspath(__file__))
    cc   = torch.cuda.get_device_capability()
    arch = f"-arch=sm_{cc[0]}{cc[1]}"

    _kernel = load_inline(
        name="megakernel_decode",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-std=c++17",
            arch, "--expt-relaxed-constexpr",
            "-I" + here, "-ccbin=/usr/bin/g++-10",
        ],
        verbose=False,
    )
    return _kernel


# ── Batched kernel ────────────────────────────────────────────────────────────
def compile_batched_kernel():
    global _batched_kernel
    if _batched_kernel is not None:
        return _batched_kernel

    cuda_src = read_source("ldg_decode_batched.cu")
    cpp_src  = _batched_cpp_src()

    here = os.path.dirname(os.path.abspath(__file__))
    cc   = torch.cuda.get_device_capability()
    arch = f"-arch=sm_{cc[0]}{cc[1]}"

    _batched_kernel = load_inline(
        name="megakernel_batched_decode",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-std=c++17",
            arch, "--expt-relaxed-constexpr",
            "-I" + here, "-ccbin=/usr/bin/g++-10",
        ],
        verbose=False,
    )
    return _batched_kernel


def _single_seq_cpp_src():
    return r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

struct LDGLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_ldg_decode(
    int input_token_id, int* output_token_id,
    const void* embed_weight, const LDGLayerWeights* layer_weights,
    const void* final_norm_weight, const void* lm_head_weight,
    const void* cos_table, const void* sin_table,
    void* k_cache, void* v_cache, void* hidden_buffer,
    void* g_activations, void* g_residual,
    void* g_q, void* g_k, void* g_v, void* g_attn_out,
    void* g_mlp_intermediate, void* g_normalized,
    void* block_max_vals, void* block_max_idxs,
    int num_layers, int position, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream);

torch::Tensor build_layer_weights(
    std::vector<torch::Tensor> weights_flat, int num_layers)
{
    std::vector<LDGLayerWeights> lw(num_layers);
    for (int i = 0; i < num_layers; i++) {
        lw[i].input_layernorm_weight     = weights_flat[i*11+0].data_ptr();
        lw[i].q_proj_weight              = weights_flat[i*11+1].data_ptr();
        lw[i].k_proj_weight              = weights_flat[i*11+2].data_ptr();
        lw[i].v_proj_weight              = weights_flat[i*11+3].data_ptr();
        lw[i].q_norm_weight              = weights_flat[i*11+4].data_ptr();
        lw[i].k_norm_weight              = weights_flat[i*11+5].data_ptr();
        lw[i].o_proj_weight              = weights_flat[i*11+6].data_ptr();
        lw[i].post_attn_layernorm_weight = weights_flat[i*11+7].data_ptr();
        lw[i].gate_proj_weight           = weights_flat[i*11+8].data_ptr();
        lw[i].up_proj_weight             = weights_flat[i*11+9].data_ptr();
        lw[i].down_proj_weight           = weights_flat[i*11+10].data_ptr();
    }
    auto d = torch::empty({num_layers*(int)sizeof(LDGLayerWeights)},
                          torch::dtype(torch::kUInt8).device(torch::kCUDA));
    cudaMemcpy(d.data_ptr(), lw.data(),
               num_layers*sizeof(LDGLayerWeights), cudaMemcpyHostToDevice);
    return d;
}

int decode(
    int input_token_id, torch::Tensor output_token,
    torch::Tensor embed_weight, torch::Tensor d_layer_weights,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor g_activations,
    torch::Tensor g_residual, torch::Tensor g_q,
    torch::Tensor g_k, torch::Tensor g_v,
    torch::Tensor g_attn_out, torch::Tensor g_mlp_intermediate,
    torch::Tensor g_normalized, torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs,
    int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_ldg_decode(
        input_token_id, (int*)output_token.data_ptr(),
        embed_weight.data_ptr(), (const LDGLayerWeights*)d_layer_weights.data_ptr(),
        final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
        cos_table.data_ptr(), sin_table.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(), hidden_buffer.data_ptr(),
        g_activations.data_ptr(), g_residual.data_ptr(),
        g_q.data_ptr(), g_k.data_ptr(), g_v.data_ptr(),
        g_attn_out.data_ptr(), g_mlp_intermediate.data_ptr(),
        g_normalized.data_ptr(), block_max_vals.data_ptr(),
        block_max_idxs.data_ptr(),
        num_layers, position, cache_len, max_seq_len, attn_scale, stream);
    return output_token.item<int>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_layer_weights", &build_layer_weights);
    m.def("decode", &decode);
}
"""


def _batched_cpp_src():
    return r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Must match BatchedLayerWeights in ldg_decode_batched.cu
struct BatchedLayerWeights {
    const void* input_layernorm_weight;
    const void* q_proj_weight;
    const void* k_proj_weight;
    const void* v_proj_weight;
    const void* q_norm_weight;
    const void* k_norm_weight;
    const void* o_proj_weight;
    const void* post_attn_layernorm_weight;
    const void* gate_proj_weight;
    const void* up_proj_weight;
    const void* down_proj_weight;
};

extern "C" void launch_batched_decode(
    const int* input_token_ids,
    int* output_token_ids,
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
    void* block_max_vals,
    void* block_max_idxs,
    int batch_size,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream);

torch::Tensor build_batched_layer_weights(
    std::vector<torch::Tensor> weights_flat, int num_layers)
{
    std::vector<BatchedLayerWeights> lw(num_layers);
    for (int i = 0; i < num_layers; i++) {
        lw[i].input_layernorm_weight     = weights_flat[i*11+0].data_ptr();
        lw[i].q_proj_weight              = weights_flat[i*11+1].data_ptr();
        lw[i].k_proj_weight              = weights_flat[i*11+2].data_ptr();
        lw[i].v_proj_weight              = weights_flat[i*11+3].data_ptr();
        lw[i].q_norm_weight              = weights_flat[i*11+4].data_ptr();
        lw[i].k_norm_weight              = weights_flat[i*11+5].data_ptr();
        lw[i].o_proj_weight              = weights_flat[i*11+6].data_ptr();
        lw[i].post_attn_layernorm_weight = weights_flat[i*11+7].data_ptr();
        lw[i].gate_proj_weight           = weights_flat[i*11+8].data_ptr();
        lw[i].up_proj_weight             = weights_flat[i*11+9].data_ptr();
        lw[i].down_proj_weight           = weights_flat[i*11+10].data_ptr();
    }
    auto d = torch::empty({num_layers*(int)sizeof(BatchedLayerWeights)},
                          torch::dtype(torch::kUInt8).device(torch::kCUDA));
    cudaMemcpy(d.data_ptr(), lw.data(),
               num_layers*sizeof(BatchedLayerWeights), cudaMemcpyHostToDevice);
    return d;
}

torch::Tensor batched_decode(
    torch::Tensor input_tokens,        // [batch_size] int32 on CUDA
    torch::Tensor output_tokens,       // [batch_size] int32 on CUDA
    torch::Tensor embed_weight,
    torch::Tensor d_layer_weights,
    torch::Tensor final_norm_weight,
    torch::Tensor lm_head_weight,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor hidden_buffer,
    torch::Tensor g_activations,
    torch::Tensor g_residual,
    torch::Tensor g_q,
    torch::Tensor g_k,
    torch::Tensor g_v,
    torch::Tensor g_attn_out,
    torch::Tensor g_mlp_intermediate,
    torch::Tensor g_normalized,
    torch::Tensor block_max_vals,
    torch::Tensor block_max_idxs,
    int batch_size,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_batched_decode(
        (const int*)input_tokens.data_ptr(),
        (int*)output_tokens.data_ptr(),
        embed_weight.data_ptr(),
        (const BatchedLayerWeights*)d_layer_weights.data_ptr(),
        final_norm_weight.data_ptr(),
        lm_head_weight.data_ptr(),
        cos_table.data_ptr(),
        sin_table.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        hidden_buffer.data_ptr(),
        g_activations.data_ptr(),
        g_residual.data_ptr(),
        g_q.data_ptr(),
        g_k.data_ptr(),
        g_v.data_ptr(),
        g_attn_out.data_ptr(),
        g_mlp_intermediate.data_ptr(),
        g_normalized.data_ptr(),
        block_max_vals.data_ptr(),
        block_max_idxs.data_ptr(),
        batch_size, num_layers, position, cache_len, max_seq_len, attn_scale, stream);
    return output_tokens;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_batched_layer_weights", &build_batched_layer_weights);
    m.def("batched_decode", &batched_decode);
}
"""


# ── Shared helpers ────────────────────────────────────────────────────────────
def precompute_rope(head_dim, max_seq_len, theta=1000000.0, device="cuda"):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    return (
        torch.cat([cos, cos], dim=-1).contiguous(),
        torch.cat([sin, sin], dim=-1).contiguous(),
    )


def pack_layer_weights(state_dict, layer_idx):
    p = f"model.layers.{layer_idx}"
    return [
        state_dict[f"{p}.input_layernorm.weight"].contiguous(),
        state_dict[f"{p}.self_attn.q_proj.weight"].contiguous(),
        state_dict[f"{p}.self_attn.k_proj.weight"].contiguous(),
        state_dict[f"{p}.self_attn.v_proj.weight"].contiguous(),
        state_dict[f"{p}.self_attn.q_norm.weight"].contiguous(),
        state_dict[f"{p}.self_attn.k_norm.weight"].contiguous(),
        state_dict[f"{p}.self_attn.o_proj.weight"].contiguous(),
        state_dict[f"{p}.post_attention_layernorm.weight"].contiguous(),
        state_dict[f"{p}.mlp.gate_proj.weight"].contiguous(),
        state_dict[f"{p}.mlp.up_proj.weight"].contiguous(),
        state_dict[f"{p}.mlp.down_proj.weight"].contiguous(),
    ]


# ── Original single-sequence Decoder (unchanged) ─────────────────────────────
class Decoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        hf_model.eval()
        sd = hf_model.state_dict()

        self.embed_weight      = sd["model.embed_tokens.weight"].contiguous()
        self.final_norm_weight = sd["model.norm.weight"].contiguous()
        self.lm_head_weight    = self.embed_weight

        self.layer_weights_flat = []
        for i in range(NUM_LAYERS):
            self.layer_weights_flat.extend(pack_layer_weights(sd, i))

        del hf_model
        torch.cuda.empty_cache()

        self.kernel          = compile_kernel()
        self.d_layer_weights = self.kernel.build_layer_weights(
            self.layer_weights_flat, NUM_LAYERS)

        self.cos, self.sin = precompute_rope(HEAD_DIM, MAX_SEQ_LEN)

        self.k_cache   = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
                                     dtype=torch.bfloat16, device="cuda")
        self.v_cache   = torch.zeros_like(self.k_cache)

        self.hidden_buffer      = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        self.g_activations      = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self.g_residual         = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self.g_q                = torch.empty(Q_SIZE,      dtype=torch.float32, device="cuda")
        self.g_k                = torch.empty(KV_SIZE,     dtype=torch.float32, device="cuda")
        self.g_v                = torch.empty(KV_SIZE,     dtype=torch.float32, device="cuda")
        self.g_attn_out         = torch.empty(Q_SIZE,      dtype=torch.float32, device="cuda")
        self.g_mlp_intermediate = torch.empty(INTERMEDIATE_SIZE, dtype=torch.float32, device="cuda")
        self.g_normalized       = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        self.block_max_vals     = torch.empty(LDG_LM_NUM_BLOCKS, dtype=torch.float32, device="cuda")
        self.block_max_idxs     = torch.empty(LDG_LM_NUM_BLOCKS, dtype=torch.int32,   device="cuda")
        self.output_token       = torch.empty(1, dtype=torch.int32, device="cuda")

        self.position   = 0
        self.attn_scale = 1.0 / (HEAD_DIM ** 0.5)

    def step(self, token_id: int) -> int:
        cache_len = self.position + 1
        result = self.kernel.decode(
            token_id, self.output_token,
            self.embed_weight, self.d_layer_weights,
            self.final_norm_weight, self.lm_head_weight,
            self.cos, self.sin,
            self.k_cache, self.v_cache,
            self.hidden_buffer, self.g_activations, self.g_residual,
            self.g_q, self.g_k, self.g_v,
            self.g_attn_out, self.g_mlp_intermediate, self.g_normalized,
            self.block_max_vals, self.block_max_idxs,
            NUM_LAYERS, self.position, cache_len, MAX_SEQ_LEN, self.attn_scale,
        )
        self.position += 1
        return result

    def reset(self):
        self.position = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


# ── BatchedDecoder — true batched kernel ──────────────────────────────────────
class BatchedDecoder:
    """
    Runs `batch_size` sequences in a single cooperative kernel launch.

    Weight matrices are loaded ONCE per row and reused across all sequences,
    so effective bandwidth cost per token = model_bytes / batch_size.

    Buffer layout (all batch-major, contiguous):
      hidden_buffer      : [batch, HIDDEN_SIZE]           bf16
      g_activations      : [batch, HIDDEN_SIZE]           f32
      g_residual         : [batch, HIDDEN_SIZE]           f32
      g_normalized       : [batch, HIDDEN_SIZE]           f32
      g_q                : [batch, Q_SIZE]                f32
      g_k / g_v          : [batch, KV_SIZE]               f32
      g_attn_out         : [batch, Q_SIZE]                f32
      g_mlp_intermediate : [batch, INTERMEDIATE_SIZE]     f32
      k_cache / v_cache  : [NUM_LAYERS, batch, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]  bf16
    """

    def __init__(self, batch_size: int, shared_weights_from: "Decoder | None" = None):
        self.batch_size = batch_size
        self.kernel     = compile_batched_kernel()

        if shared_weights_from is not None:
            # Share weight tensors — no extra GPU memory for weights
            self.embed_weight      = shared_weights_from.embed_weight
            self.final_norm_weight = shared_weights_from.final_norm_weight
            self.lm_head_weight    = shared_weights_from.lm_head_weight
            layer_weights_flat     = shared_weights_from.layer_weights_flat
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
            )
            hf_model.eval()
            sd = hf_model.state_dict()
            self.embed_weight      = sd["model.embed_tokens.weight"].contiguous()
            self.final_norm_weight = sd["model.norm.weight"].contiguous()
            self.lm_head_weight    = self.embed_weight
            layer_weights_flat     = []
            for i in range(NUM_LAYERS):
                layer_weights_flat.extend(pack_layer_weights(sd, i))
            del hf_model
            torch.cuda.empty_cache()

        self.d_layer_weights = self.kernel.build_batched_layer_weights(
            layer_weights_flat, NUM_LAYERS)
        self.cos, self.sin   = precompute_rope(HEAD_DIM, MAX_SEQ_LEN)

        B = batch_size

        # KV cache: [NUM_LAYERS, B, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM]
        self.k_cache = torch.zeros(
            NUM_LAYERS, B, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda")
        self.v_cache = torch.zeros_like(self.k_cache)

        # Activation buffers scaled by batch size
        self.hidden_buffer      = torch.empty(B * HIDDEN_SIZE,       dtype=torch.bfloat16, device="cuda")
        self.g_activations      = torch.empty(B * HIDDEN_SIZE,       dtype=torch.float32,  device="cuda")
        self.g_residual         = torch.empty(B * HIDDEN_SIZE,       dtype=torch.float32,  device="cuda")
        self.g_normalized       = torch.empty(B * HIDDEN_SIZE,       dtype=torch.float32,  device="cuda")
        self.g_q                = torch.empty(B * Q_SIZE,            dtype=torch.float32,  device="cuda")
        self.g_k                = torch.empty(B * KV_SIZE,           dtype=torch.float32,  device="cuda")
        self.g_v                = torch.empty(B * KV_SIZE,           dtype=torch.float32,  device="cuda")
        self.g_attn_out         = torch.empty(B * Q_SIZE,            dtype=torch.float32,  device="cuda")
        self.g_mlp_intermediate = torch.empty(B * INTERMEDIATE_SIZE, dtype=torch.float32,  device="cuda")

        # LM head reduction buffers: [B, LDG_LM_NUM_BLOCKS]
        self.block_max_vals = torch.empty(B * LDG_LM_NUM_BLOCKS, dtype=torch.float32, device="cuda")
        self.block_max_idxs = torch.empty(B * LDG_LM_NUM_BLOCKS, dtype=torch.int32,   device="cuda")

        # I/O token tensors on device
        self.input_tokens  = torch.empty(B, dtype=torch.int32, device="cuda")
        self.output_tokens = torch.empty(B, dtype=torch.int32, device="cuda")

        self.position   = 0
        self.attn_scale = 1.0 / (HEAD_DIM ** 0.5)

    def step(self, token_ids: list[int]) -> list[int]:
        """
        token_ids : list of length batch_size
        returns   : list of next token ids, one per sequence
        """
        assert len(token_ids) == self.batch_size
        self.input_tokens.copy_(
            torch.tensor(token_ids, dtype=torch.int32, device="cuda"))

        cache_len = self.position + 1

        self.kernel.batched_decode(
            self.input_tokens,
            self.output_tokens,
            self.embed_weight,
            self.d_layer_weights,
            self.final_norm_weight,
            self.lm_head_weight,
            self.cos,
            self.sin,
            self.k_cache,
            self.v_cache,
            self.hidden_buffer,
            self.g_activations,
            self.g_residual,
            self.g_q,
            self.g_k,
            self.g_v,
            self.g_attn_out,
            self.g_mlp_intermediate,
            self.g_normalized,
            self.block_max_vals,
            self.block_max_idxs,
            self.batch_size,
            NUM_LAYERS,
            self.position,
            cache_len,
            MAX_SEQ_LEN,
            self.attn_scale,
        )

        self.position += 1
        return self.output_tokens.tolist()

    def reset(self):
        self.position = 0
        self.k_cache.zero_()
        self.v_cache.zero_()