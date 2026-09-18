# fix cp.async L2 prefetch typo

Upstream: https://github.com/NVIDIA/cutlass/pull/1187

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

fix a typo for cp.async L2 prefetch.

And I also found that, the copy_traits will always forward the SM80_CP_ASYNC_CACHEGLOBAL to SM80_CP_ASYNC_CACHEGLOBAL_ZFILL(https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm80.hpp#L57, https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/copy_traits_sm80.hpp#L79).
Thus the SM80_CP_ASYNC_CACHEGLOBAL's implementation will never be called any more, should we delete it(CopyOperation, Copy_Traits)?

## Discussion

### hwu36 · 2023-11-14T20:17:40Z

https://github.com/NVIDIA/cutlass/pull/1187#issuecomment-1811180380

@thakkarV  for the question above

### thakkarV · 2023-11-14T20:19:45Z

https://github.com/NVIDIA/cutlass/pull/1187#issuecomment-1811183930

I don't think we should delete the unused ones, as someone else may need them at some point. but this is indeed a typo.

### reed-lau · 2023-11-27T06:55:33Z

https://github.com/NVIDIA/cutlass/pull/1187#issuecomment-1827225128

Hi, @hwu36 could you help merge it? or do I need more modification?

## Reviews

### hwu36 · 2023-11-28T21:57:58Z

https://github.com/NVIDIA/cutlass/pull/1187#pullrequestreview-1754028017


