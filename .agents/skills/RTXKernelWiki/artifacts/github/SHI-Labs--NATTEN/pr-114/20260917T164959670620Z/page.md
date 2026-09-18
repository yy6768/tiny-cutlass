# Disable 64x128x128 GEMM config for SM86 and 89

Upstream: https://github.com/SHI-Labs/NATTEN/pull/114

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

Noticed that FNA unit tests fail on SM86 because it can't handle any configs with a 64x128x128 GEMM tile size. Obviously due to shared memory limits and how SM80 and 86 share all their kernels but there's a huge disparity between A100/H100 (SM80 and SM90) and the rest of the GPUs in their respective generations (SM86 and SM89).

While I can't test SM89, I'm disabling the overly-large GEMM config for both just to be safe until we can test SM89, but I'm relatively confident that SM89 won't be able to handle the same config.
