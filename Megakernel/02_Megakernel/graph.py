import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Labels
# -----------------------------
labels = ['HF (Eager)', 'HF (SDPA)', 'Megakernel', 'Theoretical']
x = np.arange(4)

# -----------------------------
# Compute (same for all)
# -----------------------------
compute = np.array([7.14, 7.14, 7.14, 7.14])

# -----------------------------
# HF (Eager)
# -----------------------------
dram_eager = 6.8
kernel_eager = 2.6
sync_eager = 0
other_eager = 1.45

# -----------------------------
# HF (SDPA)
# -----------------------------
dram_sdpa = 4.2
kernel_sdpa = 1.5
sync_sdpa = 0
other_sdpa = 0.34

# -----------------------------
# Megakernel (corrected)
# -----------------------------
dram_mega = 0
kernel_mega = 0
sync_mega = 0.65
other_mega = 0.58

# -----------------------------
# Theoretical
# -----------------------------
dram_theo = 0
kernel_theo = 0
sync_theo = 0
other_theo = 0

# -----------------------------
# Combine arrays
# -----------------------------
dram = np.array([dram_eager, dram_sdpa, dram_mega, dram_theo])
kernel = np.array([kernel_eager, kernel_sdpa, kernel_mega, kernel_theo])
sync = np.array([sync_eager, sync_sdpa, sync_mega, sync_theo])
other = np.array([other_eager, other_sdpa, other_mega, other_theo])

# -----------------------------
# Colors (match clean style)
# -----------------------------
c_compute = "#2E9B45"
c_sync    = "#F39C12"
c_kernel  = "#E74C3C"
c_dram    = "#8E44AD"
c_other   = "#A6ACAF"

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(9,6))

plt.bar(x, compute, color=c_compute, label='Weight streaming (compute)')
plt.bar(x, sync, bottom=compute, color=c_sync, label='Grid sync / coordination')
plt.bar(x, kernel, bottom=compute+sync, color=c_kernel, label='Kernel launch overhead')
plt.bar(x, dram, bottom=compute+sync+kernel, color=c_dram, label='DRAM round-trips')
plt.bar(x, other, bottom=compute+sync+kernel+dram, color=c_other, label='Other overhead')

# -----------------------------
# Totals (match your real values)
# -----------------------------
totals = compute + sync + kernel + dram + other
for i, v in enumerate(totals):
    plt.text(i, v + 0.3, f"{v:.2f} ms", ha='center', fontsize=11, fontweight='bold')

# -----------------------------
# Labels
# -----------------------------
plt.xticks(x, labels)
plt.ylabel("Time per token (ms)")
plt.title("Where Decode Time Goes (RTX 3050, Qwen3-0.6B)")

plt.legend(loc='upper right')
plt.tight_layout()

# Save (SSH safe)
plt.savefig("final_corrected_breakdown.png", dpi=300)
print("Saved as final_corrected_breakdown.png")