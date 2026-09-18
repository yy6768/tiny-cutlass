# Fix the ScatterD issue in predicated_tile_iterator

Upstream: https://github.com/NVIDIA/cutlass/pull/3278

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

## Summary

This PR fixes the remaining `ScatterD` pointer-advance issue in `PredicatedTileIterator`. This change:
- Guards `increment_group` and `increment_cluster` in `load_with_byte_offset()` with `!ScatterD`
- Updates `operator+=()` to follow the same pointer-advance rules as `operator++()`

Fixes #3101.

## Validation

I added a minimal local repro [bug_repro.zip](https://github.com/user-attachments/files/28296041/bug_repro.zip) using:
- CUTLASS GEMM 2.x API
- SIMT core
- gather A + GEMM + scatter D
- `int8 x int8 -> int32`

Before the fix:
```text
num_mismatch: 8
```
After the fix:
```text
num_mismatch: 0
```

## Discussion

### github-actions[bot] · 2026-06-26T07:26:43Z

https://github.com/NVIDIA/cutlass/pull/3278#issuecomment-4807294979

This PR has been labeled `inactive-30d` due to no recent activity in the past 30 days. Please close this PR if it is no longer required. Otherwise, please respond with a comment indicating any updates. This PR will be labeled `inactive-90d` if there is no activity in the next 60 days.

### tterava · 2026-07-02T05:10:24Z

https://github.com/NVIDIA/cutlass/pull/3278#issuecomment-4862404367

Independent confirmation of this bug and the fix.

I hit the same defect through a different path: a grouped SIMT f32 GEMM (`GemmShape<64,64,8>`, warp `<32,32,8>`) with GatherA + ScatterD, where the epilogue loads a source operand (`beta == 1`) through a scatter-indexed `iterator_C` (same indices as D). For that configuration the output thread map has `Iterations::kGroup == 2` / `Delta::kGroup == 8`, so the ungated `increment_group` advance in `load_with_byte_offset()` offsets the second group iteration's source loads by 8 rows past the row selected by `indices_` — `D = alpha * AB + beta * C` silently picks up `C` from unrelated rows for half of each tile's rows. The store path and `operator++` already gate these advances on `!ScatterD`; only the load path was missing the guards, which is why `beta == 0` scatter workloads never see it.

Cherry-picking this PR's commit onto v4.5.2-era main fixed the corruption in my application where the accumulate feeds back every iteration, so the wrong reads were catastrophic rather than subtle.

Hoping this can get review attention — the failure mode is silent wrong results for any `beta != 0` epilogue with ScatterD whenever `Iterations::kGroup > 1` or `kCluster > 1`.

### github-actions[bot] · 2026-08-01T06:55:31Z

https://github.com/NVIDIA/cutlass/pull/3278#issuecomment-5150282489

This PR has been labeled `inactive-30d` due to no recent activity in the past 30 days. Please close this PR if it is no longer required. Otherwise, please respond with a comment indicating any updates. This PR will be labeled `inactive-90d` if there is no activity in the next 60 days.

## Reviews

### jackkosaian · 2026-08-14T10:40:08Z

https://github.com/NVIDIA/cutlass/pull/3278#pullrequestreview-4936383430


