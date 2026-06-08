# Flash Attention Working Notes

## Goal
This directory tracks the current flash-attention study path inside tiny-cutlass.

- Optimize for SM80, SM89, and SM90 using CUTLASS 2.x or CUTLASS 3.x style code where each fits best.
- Use the current kernel family as a learning loop: understand the operator, implement the kernel, measure it, explain the changes, and keep the lesson for the next numbered step.
- Match a trusted reference implementation first, usually PyTorch, cuDNN, or TensorRT, before trusting performance data.
- Stay within the required numerical tolerance before treating a kernel as correct. The usual target is MAE <= 1e-3 unless a specific experiment needs something different.
- Treat Nsight Compute and Nsight Systems as the primary performance workflow. Keep `.ncu-rep`, `.nsys-rep`, and CSV outputs as first-class artifacts for both human review and agent analysis.
- Record learning outcomes with Chinese technical notes suitable for Zhihu, X, GitHub, or similar publication channels.
- Keep fallback implementations to a minimum. Prefer explicit CMake dependency checks and explicit unsupported paths over hidden fallback code in kernels or test harnesses.
- Focus current optimization work around SM80 FP16 and SM89 FP8 paths unless the user explicitly expands the architecture or dtype scope.

## Workflow
1. Build the kernel family.
2. Verify against the chosen reference.
3. Benchmark only after verification passes.
4. Profile with Nsight when performance work needs evidence.

## Conventions
- Kernel steps are numbered in order, such as `00`, `01`, `02`, and so on.
- `00` is usually the baseline for a family, but the directory is not locked to a fixed number of steps.
- Keep numbered kernel `.cu` files as launch entries. The shared executable under `csrc/tests/flash-attention` owns input generation, reference execution, MAE checking, and timing.
- Use cuDNN SDPA as the current reference backend for fixed-seqlen flash-attention tests. Check cuDNN dependencies at CMake configure time and fail early if they are missing; do not add `HAS_CUDNN`-style fallback branches in C++.
- Keep every numbered kernel variant registered through the shared `Kernel` interface in `flash_attention.h`; `flash_attention_test --kernel=all` must cover 00/01/02 and future migrated variants.
- Prefer CUTLASS example-style C++ structure: keep public learning interfaces in the simple global scope and use anonymous namespaces only for file-local helpers. Do not add multi-level project namespaces such as `tiny_cutlass::flash_attention` in this workspace.
- Keep `blogs/` for notes only.
- Keep generated build and profiling artifacts under `build/`.
- Keep changes local to the current family unless the user explicitly broadens scope.

## Current kernels
- `00-naive-attention` materializes the full `P` matrix and is the baseline.
- `01-online-softmax` only changes the softmax kernel; it still materializes `P`.
- `02-tiled-online-attention` is the first IO-aware tiled kernel and keeps the attention tile local instead of writing the full `P` matrix to global memory.

## Current test entry
- `flash_attention_test` is the shared test executable for registered kernels.
- Use `--kernel=list`, `--kernel=00-naive`, `--kernel=01-online-softmax`, `--kernel=02-tiled-online`, or `--kernel=all`.
- Compatibility executables `flash_attention_00_naive_attention_test`, `flash_attention_01_online_softmax_attention_test`, and `flash_attention_02_tiled_online_attention_test` point at the same shared host C++ test main with different default kernels.
