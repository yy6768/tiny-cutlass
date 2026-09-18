# Fix/cutlass fna dv nan issue

Upstream: https://github.com/SHI-Labs/NATTEN/pull/341

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

Fixes longstanding issue in CUTLASS FNA backward's mask that evaded nearly all unit tests, and
only broke one application (that we know of), resulting in NaNs in dV. We were avoiding a boundary
check that seems to be required when we have partial tiles, but only on SM80 and later.
