---
name: cutlass-kernel
description: Generic workflow and guardrails for developing, verifying, benchmarking, and profiling CUTLASS-based CUDA kernels across any operator family, including GEMM, convolution, attention, fused epilogues, and related experiments. Use when creating or modifying CUTLASS experiments, writing .bat-driven build/verify/bench flows, adding reference tests under csrc/tests, or capturing Nsight Compute CSV and .ncu-rep artifacts.
---

# CUTLASS Kernel Workflow

## Goal
1. Optimize and program CUDA kernels for NVIDIA SM80, SM89, and SM90 targets, using CUTLASS 2.x and CUTLASS 3.x styles where they fit the experiment.
2. Treat every kernel experiment as a learning loop around the current project needs: study the operator, implement it, measure it, explain the optimization choices, and preserve the lessons for future kernels.
3. For each kernel, first match a trusted reference implementation, usually PyTorch, cuDNN, TensorRT, or another explicitly chosen reference. Only pursue performance once the implementation stays within the required numerical tolerance, usually MAE <= 1e-3 unless the user gives a different tolerance.
4. Make real profiling artifacts the center of performance work. Use Nsight Compute, Nsight Systems, and available CUDA profiling APIs to produce human-readable reports such as `.ncu-rep` and `.nsys-rep`, plus `.csv` outputs intended for agent-side analysis.
5. Preserve new learning by collaborating with the human on Chinese technical blog material suitable for Zhihu, X, GitHub, or similar public engineering notes.

## Scope
- Keep this skill generic for CUTLASS kernel work.
- Treat experiments as kernel families with any number of numbered variants, normally `00`, `01`, `02`, and so on.
- Do not hardcode attention, a fixed `00`/`01` relationship, or any single demo as the default structure.
- Keep `blogs/` as notes and learning material only. Do not make blogs part of the build, verification, or benchmark path.

## Workflow
1. Create or maintain one `.bat` entrypoint per kernel family, for example `scripts\kernels\<family>\<family>.bat`.
2. Make the `.bat` execute the workflow in this order: `build`, then `verify`, then `bench`.
3. Stop after `verify` if correctness fails. Do not benchmark or profile kernels that do not match the reference within the required tolerance.
4. Put all build outputs under `build/`.
5. Put verification and benchmark harnesses under `csrc/tests/`.
6. Compare against the chosen reference implementation before trusting any performance data.
7. Capture Nsight Compute and Nsight Systems artifacts in both human-readable report formats and machine-readable CSV formats.
8. Keep variants numbered and comparable within the same kernel family. Each new variant should have a clear reason to exist, such as a tiling change, memory-layout change, pipeline change, epilogue fusion, or architecture-specific path.

## DO
- Keep build artifacts inside `build/`.
- Keep generated reports, build debris, and `__pycache__` out of the worktree.
- Treat reference parity as a required gate before benchmark conclusions.
- Prefer small diffs that make the optimization hypothesis easy to inspect.
- Emit both machine-readable and human-readable profiling artifacts.

## DO NOT
- Do not write build outputs outside `build/`.
- Do not leave performance reports or temporary build files in source directories.
- Do not treat numbers from an unverified kernel as valid performance analysis.
- Do not accept a performance report that lacks reference parity.
- Do not make a kernel-family workflow depend on attention-specific assumptions unless the current user request is explicitly about attention.
