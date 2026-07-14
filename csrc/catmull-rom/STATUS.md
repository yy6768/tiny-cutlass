# Catmull-Rom Reprojection — Status

> Implementation state and verification results for `csrc/catmull-rom/`.
> For the algorithm and math, see [`README.md`](README.md).

Last updated: 2026-07-14.

## Summary

| Component | State | Reference | Result |
|-----------|-------|-----------|--------|
| Forward (UE Catmull-Rom, A=-0.5) | ✅ done | independent fp32 host gather | max_abs ~2e-4 (half) |
| Backward (history + motion) | ✅ done | Slang autodiff + analytic host | see below |
| Slang autodiff cross-check | ✅ done | slangc `bwd_diff` of UE math | fwd 1.2e-7, grad 2e-7 |
| Motion dilation (standalone) | ✅ done | independent host argmax | exact (max_abs 0) |
| Dilated + Catmull (fused) | ✅ done | host dilation + gather | max_abs ~1e-4 |
| 9-fetch texture variant | ⬜ planned | — | — |
| Backward through dilation | ⬜ planned | — | — |
| H100 performance / bench | ⬜ planned | bilinear baseline | — |

## Layout

```
csrc/catmull-rom/
  kernel/
    forward.h                          forward: UE weights, catmull_rom_interp16 core, __global__ + factory
    backward.h                         backward: weight derivs, grad_history scatter + grad_motion, factory
    dilation.h                         standalone dilation + fused gather
  device/                              CUTLASS-style ops (Arguments/can_implement/initialize/run/operator())
    forward.h
    backward.h
    dilation.h
  slang/
    catmull_rom.slang                  [Differentiable] UE math source
    catmull_rom.cu                     checked-in generated CUDA (autodiff backward)
  README.md                            algorithm, derivation, layout, and workflow
  STATUS.md                            this file

csrc/tests/catmull-rom/
  forward.cu                           forward vs host gather (half)
  backward.cu                          grad_history vs analytic scatter; grad_motion vs finite-diff
  autodiff.cu                          forward + both grads vs generated autodiff
  dilation.cu                          standalone dilation and fused pipeline checks

scripts/kernels/catmull-rom/
  run.bat                             build -> verify -> bench gate
  verify.py                           run the 4 correctness executables
  bench.py                            require verify parity; emits a skipped benchmark stage
```

## Verification detail

Run everything with `scripts/kernels/catmull-rom/run.bat`. The entrypoint
runs build -> verify -> bench in that order and stops immediately if verification
fails. The bench stage currently reports an explicit skip because there is no
dedicated timing harness yet. Latest run — all 4 test executables pass:

- **catmull_forward** (forward, half output): 4 shape cases, `max_abs`
  ~1.2e-4 … 2.4e-4 (tol 0.02); `C=0` correctly rejected.
- **catmull_backward** (float): `grad_history` vs analytic scatter
  `gh_max` ~1e-7 (tol 1e-3); `grad_motion` vs central finite-difference on the
  fp32 forward `gm_max` ~4e-3 (tol 5e-2, O(ε²) truncation).
- **catmull_autodiff** (ground truth): 256 random problems vs the
  slangc-generated autodiff reference — `max_fwd` 1.2e-7, `max_grad_mv` 2.4e-7,
  `max_grad_tap` 1.2e-7. Both sides use the identical UE `A=-0.5` weights, so
  this is fp32-rounding-level agreement.
- **catmull_dilation**: standalone op vs host 5-tap argmax — exact match
  (`max_abs = 0`), since dilation only selects an existing MV. Edge cases `1×N`,
  `N×1` exercise out-of-bounds tap skipping; `H=0` rejected. The same
executable checks fused dilation→gather with `max_abs` ~1.2e-4 (tol 0.02).

## Maintenance notes

- `run.bat` regenerates `slang/catmull_rom.cu` only when
  `CATMULL_ROM_REGENERATE_SLANG=1`; otherwise the generated file is unchanged.
- The regeneration step writes ASCII after patching `SLANG_globalParams` so
  NVCC can read the generated CUDA source consistently.
- `bench.py` is a verify gate only: no timing harness or performance result is
  included in this initial submission.

## Key correction (2026-07-10)

The prior grid_sample cross-check was **removed**. PyTorch
`grid_sample(mode='bicubic')` uses the Keys cubic with `A = -0.75`, which is a
different filter from UE Catmull-Rom (`A = -0.5`); matching against it was
incorrect. The reference is now the Slang autodiff port of the exact UE math.
The kernels themselves were already `A = -0.5` (the weight code was correct); the
change is the reference, plus explicit UE-form weights and a shared
`catmull_rom_interp16` core so the Slang check exercises the same math.

## Regenerating the Slang reference

The generated `slang/catmull_rom.cu` is checked in, so a normal build
needs no slangc. After editing the `.slang`, set
`CATMULL_ROM_REGENERATE_SLANG=1` before running the `.bat`. The script uses
`%SLANGC%` when set, otherwise the default Vulkan SDK path, and patches the
`SLANG_globalParams` declaration from `extern "C" __constant__` to a definition,
since the test `#include`s the generated `.cu` directly instead of linking the
Slang runtime.
