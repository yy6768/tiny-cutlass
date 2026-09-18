# cutlass-fna: 64b strides

Upstream: https://github.com/SHI-Labs/NATTEN/pull/337

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

token layout strides should be encoded in int64 for safety; left-most stride can get pretty big, especially if users go beyond the typical 128 head dim, and have high token counts on the right.

Some small regressions (up to ~ 15%) are expected.

Fixes #335.
