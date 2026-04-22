import torch
import time
from torch.utils.cpp_extension import load_inline
from pathlib import Path

HERE = Path(__file__).parent

with open(HERE / "rmsnorm.cu") as f:
    rmsnorm_src = f.read()
with open(HERE / "matvec.cu") as f:
    matvec_src = f.read()

ext = load_inline(
    name="separate_kernels",
    cpp_sources=["""
#include <torch/extension.h>
#include <cuda_bf16.h>

extern "C" void launch_rmsnorm(__nv_bfloat16*, const __nv_bfloat16*,
                               const __nv_bfloat16*, int, float);
extern "C" void launch_matvec(float*, const __nv_bfloat16*,
                              const float*, int, int);

torch::Tensor rmsnorm(torch::Tensor input, torch::Tensor weight, float eps) {
    auto output = torch::empty_like(input);
    launch_rmsnorm(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        input.numel(), eps
    );
    return output;
}

torch::Tensor matvec(torch::Tensor weight, torch::Tensor input) {
    int M = weight.size(0), K = weight.size(1);
    auto output = torch::empty({M}, torch::dtype(torch::kFloat32).device(weight.device()));
    launch_matvec(
        output.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        input.data_ptr<float>(),
        M, K
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm);
    m.def("matvec", &matvec);
}
"""],
    cuda_sources=[rmsnorm_src, matvec_src],
    extra_cuda_cflags=["-O3", "-ccbin=/usr/bin/g++-10"],
    verbose=False,
)

HIDDEN = 1024
INTER = 2816
N_HEADS = 16
HEAD_DIM = HIDDEN // N_HEADS
QKV_DIM = HIDDEN * 3
N_LAYERS = 28

norm_w = torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda")
w_qkv = torch.randn(QKV_DIM, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
w_o = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
w_gate = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
w_up = torch.randn(INTER, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.02
w_down = torch.randn(HIDDEN, INTER, dtype=torch.bfloat16, device="cuda") * 0.02

x = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda")

def one_layer(x):
    h = ext.rmsnorm(x, norm_w, 1e-6)
    h_f = h.float()
    qkv = ext.matvec(w_qkv, h_f)
    attn_out = qkv[:HIDDEN]
    o = ext.matvec(w_o, attn_out)
    x_mid = x.float() + o
    x_mid_bf = x_mid.bfloat16()
    h2 = ext.rmsnorm(x_mid_bf, norm_w, 1e-6)
    h2_f = h2.float()
    gate = ext.matvec(w_gate, h2_f)
    up = ext.matvec(w_up, h2_f)
    hidden = torch.nn.functional.silu(gate) * up
    down = ext.matvec(w_down, hidden)
    return (x_mid + down).bfloat16()

for _ in range(5):
    out = x.clone()
    for _ in range(N_LAYERS):
        out = one_layer(out)
    torch.cuda.synchronize()

TRIALS = 50
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(TRIALS):
    out = x.clone()
    for _ in range(N_LAYERS):
        out = one_layer(out)
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / TRIALS

per_layer = elapsed / N_LAYERS
kernels_per_layer = 8
total_kernels = kernels_per_layer * N_LAYERS

weight_bytes = (
    w_qkv.nelement() + w_o.nelement() +
    w_gate.nelement() + w_up.nelement() + w_down.nelement()
) * 2
total_weight_bytes = weight_bytes * N_LAYERS

props = torch.cuda.get_device_properties(0)
peak_bw = props.total_memory  # rough proxy; real bandwidth from spec
gpu_name = props.name

print(f"GPU: {gpu_name}")
print(f"Layers: {N_LAYERS}, Kernels/layer: {kernels_per_layer}")
print(f"Total kernels launched: {total_kernels}")
print(f"Total time: {elapsed*1000:.2f} ms")
print(f"Per-layer time: {per_layer*1e6:.1f} us")
print(f"Per-kernel avg: {elapsed/total_kernels*1e6:.1f} us")
print(f"Weight memory per pass: {total_weight_bytes/1e6:.1f} MB")
print(f"Estimated launch overhead: {total_kernels * 5e-6 * 1000:.2f} ms "
      f"(assuming ~5us per launch)")
print(f"Launch overhead fraction: "
      f"{total_kernels * 5e-6 / elapsed * 100:.1f}%")