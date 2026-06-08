# Flash Attention Notes

This folder contains the migrated flash-attention implementation inside tiny-cutlass.

## Current structure
- `00-naive-attention` is the baseline implementation and still materializes full `P`.
- `01-online-softmax` only changes the softmax path; it is not IO-aware tiling.
- `02-tiled-online-attention` is the current IO-aware tiled kernel step, adapted locally from the CUTLASS flash-attention example.
- `epilogue/`, `gemm/`, `iterators/`, and `transform/` are the support headers needed by `02`.
- `blogs/` holds notes only.

## How to work here
- Build, then verify, then benchmark.
- Use the chosen reference implementation for correctness checks before performance conclusions.
- Keep shared test logic in `csrc/tests/flash-attention/flash_attention_test.cpp`; numbered kernel `.cu` files should expose launch entries through `flash_attention.h`.
- Keep 00/01/02 and future migrated variants registered in the shared `Kernel` interface so `flash_attention_test --kernel=all` exercises them together.
- Follow CUTLASS example style for C++ scope: avoid multi-level project namespaces in this learning workspace; use anonymous namespaces only for file-local helpers.
- The current reference contract is cuDNN SDPA. CMake must fail early if cuDNN headers/frontend/import library are missing; avoid `HAS_CUDNN`-style C++ fallback branches.
- Minimize fallback implementations. Unsupported architectures, dtypes, or dependencies should be rejected explicitly instead of selecting weaker hidden paths.
- Focus current optimization work around SM80 FP16 and SM89 FP8 paths unless the user explicitly expands scope.
- Keep profiling artifacts under `build/`.
- Keep step numbering explicit and local to the family.
- Avoid attention-specific assumptions outside this directory unless the user asks for a broader refactor.
