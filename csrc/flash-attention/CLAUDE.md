# Flash Attention Notes

This folder contains the migrated flash-attention implementation inside tiny-cutlass.

## Current structure
- `00-naive-attention` is the baseline implementation and still materializes full `P`.
- `01-online-softmax` only changes the softmax path; it is not IO-aware tiling.
- `02-tiled-online-attention` is the current IO-aware tiled variant, adapted locally from the CUTLASS flash-attention example.
- `epilogue/`, `gemm/`, `iterators/`, and `transform/` are the support headers needed by `02`.
- `blogs/` holds notes only.

## How to work here
- Build, then verify, then benchmark.
- Use the chosen reference implementation for correctness checks before performance conclusions.
- Keep profiling artifacts under `build/`.
- Keep variant numbering explicit and local to the family.
- Avoid attention-specific assumptions outside this directory unless the user asks for a broader refactor.
