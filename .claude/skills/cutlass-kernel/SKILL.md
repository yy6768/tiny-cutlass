---
name: cutlass-kernel
description: Router and hard rules for all CUTLASS kernel work in this repository (SM80/SM89, CUTLASS 2.x). Use as the entry point whenever creating or modifying a CUTLASS experiment, then hand off to the stage skill that matches the current gate. Covers the layer hierarchy index, naming rules, and the four-gate workflow. Triggers on CUTLASS, implicit GEMM, epilogue fusion, kernel family, DefaultXxx factory, threadblock mainloop, tile iterator.
---

# CUTLASS Kernel Work — Router

This skill is an index and a rule set, not a procedure. Kernel work in this
repository runs through four gates, each with its own skill. Load the one that
matches where you are; do not run a later gate before an earlier one passes.

## The four gates

| Gate | Skill | Passes when |
|---|---|---|
| 1. Design | `cutlass-design` | A design doc exists in `<family>/docs/` naming the layer decomposition, what CUTLASS already provides, the reference contract, and a step order |
| 2. Implement | `cutlass-implement` | Code compiles and follows the layer + naming discipline; unsupported configs fail explicitly |
| 3. Verify | `cutlass-verify` | A test under `csrc/tests/<family>/` matches a trusted reference within tolerance; the target is registered in CMake |
| 4. Bench | `cutlass-bench` | `bench.py` produces numbers on a *verified* kernel, with artifacts under `profile/` |

Deep performance analysis after gate 4 belongs to `ncu-report`.
Writing this up as a learning blog belongs to `cutlass-blog-workflow` — it has
its own three gates (overview approval, reference parity, measured NCU) and runs
alongside, not inside, this workflow.

## Shared reference material

Read these instead of re-deriving CUTLASS structure from the headers:

- [references/layer-map.md](references/layer-map.md) — the five-layer hierarchy
  (`device` → `kernel` → `threadblock` → `warp` → `thread`), the SM80/89
  multistage `cp.async` pipeline, and what `NumStages` buys.
- [references/cutlass-dirs.md](references/cutlass-dirs.md) — what actually lives
  in `include/cutlass/gemm/` and `include/cutlass/conv/`, per subdirectory, and
  what is out of scope (`collective/`, SM90 paths).
- [references/examples.md](references/examples.md) — the two upstream examples
  that anchor the SM80/89 operator pattern (GEMM: `examples/14`, implicit-GEMM
  conv: `examples/16`), including the exact host-side call sequence every
  `device`-layer operator here must expose.
- [references/naming-rules.md](references/naming-rules.md) — the template-factory
  naming discipline. dtype / arch / layout never enter a primary name.

## Hard rules (apply at every gate)

- **SM80 / SM89 only, CUTLASS 2.x only.** CUTLASS 4.5.2's
  `gemm/collective/collective_builder.hpp` has no Sm80/89 specialization, so 3.x
  on this hardware means hand-writing `CollectiveMma<MainloopSm80CpAsync, ...>`
  with no conv implicit-GEMM, no epilogue visitor, and no from-smem B2B available.
  Do not add a 3.x/CuTe variant unless the user asks for one explicitly.
- **Reference parity gates everything downstream.** Numbers from an unverified
  kernel are not performance data. Do not benchmark or profile past a failed
  verify.
- **No silent downgrades.** A TensorOp experiment never falls back to SIMT or raw
  CUDA. Unsupported arch / dtype / layout / shape must fail through CMake, an
  assert, `cutlass::Status::kErrorNotSupported`, or a CUDA error.
- **Build outputs only under `build/`.** Reports under `profile/`. Nothing
  generated in source directories, and no `__pycache__` in the worktree.
- **One script entry point per family**: `scripts/kernels/<family>/run.bat`,
  order fixed `build → verify → bench`. Verification is `verify.py`, benchmark is
  `bench.py` — no `compare_*.py` / `*_perf.py` side names.
- **`blogs/` and `docs/` are notes only.** Never on the build, verify, or bench
  path.
- **Variants are numbered** `00`, `01`, `02`, ... within a family, and each needs
  a stated reason to exist: a tiling, layout, pipeline, fusion, or architecture
  change. Broader dtype/shape support is a new variant, not a fallback inside an
  existing one.

## Workspace conventions

Each family owns a directory under `csrc/<workspace>/<family>/`:

```text
ops/          public API: raw device pointer + problem descriptor + cudaStream_t
device/       CUTLASS device operator + explicit can_implement
kernel/       DefaultXxx<ArchTag, Element, TBShape, WarpShape> factory
threadblock/  CTA-level composition
warp/         warp-level primitives
epilogue/     output iterators and visitors
```

Only create directories that hold real content. Core runtime entry points take
raw device pointers, a problem descriptor, and a `cudaStream_t` — no ATen/Torch
tensor ownership below the `ops/` boundary.

Workspace-local `AGENTS.md` files carry the family's verified constraints and
known-wrong-turn list. Read the one for the workspace you are touching before
editing; it can add constraints but never relax the ones above.
