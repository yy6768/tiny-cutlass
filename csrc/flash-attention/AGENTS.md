# Flash Attention Working Notes

## Goal
This directory tracks the current flash-attention study path inside tiny-cutlass.

- Optimize for SM80, SM89, and SM90 using CUTLASS 2.x or CUTLASS 3.x style code where each fits best.
- Use the current kernel family as a learning loop: understand the operator, implement the kernel, measure it, explain the changes, and keep the lesson for the next variant.
- Match a trusted reference implementation first, usually PyTorch, cuDNN, or TensorRT, before trusting performance data.
- Stay within the required numerical tolerance before treating a kernel as correct. The usual target is MAE <= 1e-3 unless a specific experiment needs something different.
- Treat Nsight Compute and Nsight Systems as the primary performance workflow. Keep `.ncu-rep`, `.nsys-rep`, and CSV outputs as first-class artifacts for both human review and agent analysis.
- Record learning outcomes with Chinese technical notes suitable for Zhihu, X, GitHub, or similar publication channels.

## Workflow
1. Build the kernel family.
2. Verify against the chosen reference.
3. Benchmark only after verification passes.
4. Profile with Nsight when performance work needs evidence.

## Conventions
- Variants are numbered in order, such as `00`, `01`, `02`, and so on.
- `00` is usually the baseline for a family, but the directory is not locked to a fixed number of variants.
- Keep `blogs/` for notes only.
- Keep generated build and profiling artifacts under `build/`.
- Keep changes local to the current family unless the user explicitly broadens scope.

## Current variants
- `00-naive-attention` materializes the full `P` matrix and is the baseline.
- `01-online-softmax` only changes the softmax kernel; it still materializes `P`.
- `02-tiled-online-attention` is the first IO-aware tiled variant and keeps the attention tile local instead of writing the full `P` matrix to global memory.
