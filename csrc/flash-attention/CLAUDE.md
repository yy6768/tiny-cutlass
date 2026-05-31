# Flash Attention Notes

This folder contains the migrated flash-attention implementation inside tiny-cutlass.

## Current structure
- `00-naive-attention` is the baseline implementation.
- `01-online-softmax` is the numbered derivative of `00`; the intended change is the softmax path.
- `blogs/` holds notes only.
- `02-tiled-online-attention` and the older helper tree are no longer part of the active code path.

## How to work here
- Build, then verify, then benchmark.
- Use the chosen reference implementation for correctness checks before performance conclusions.
- Keep profiling artifacts in `build/reports/flash-attention/...`.
- Keep variant numbering explicit and local to the family.
- Avoid attention-specific assumptions outside this directory unless the user asks for a broader refactor.
