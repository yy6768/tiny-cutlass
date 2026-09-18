# [Kernel] Tuned FP8 Kernels for Ada Lovelace

Upstream: https://github.com/vllm-project/vllm/pull/6677

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

- Tune SM89 cutlass FP8 kernels for performance. 
- Refactor/Cleanup cutlass2x kernel code.
- High-level performance takeaways
  - Good performance speedups for M <= 16
  - `llama2-7b` :
      - min speedup 0.77x. max speed up 4.8x.
      - Number of Gemm shapes with perf. under 1.0x  - 11 / 28
      - Number of Gemm shapes with perf. under 0.9x  - 2 / 28
  
  - `llama3-8b` :
      -  min speedup 0.88x. max speedup 5.13x.
      - Number of Gemm shapes with perf. under 1.0x - 8 / 28
      - Number of Gemm shapes with perf. under 0.9x - 1 / 28

  - `llama2-13b`:
      - min speedup 0.4x. max speedup 4.74x. Until M <= 128, cutlass kernels are almost always over 1.0x faster.  
      - Number of Gemm shapes with perf. under 1.0x - 9 / 28
      - Number of Gemm shapes with perf. under 0.9x - 3 / 28
     
  - `llama2-70b` :
     -  min speedup 0.82x. max speedup 3.0x. Until M <= 128, cutlass kernels are almost always over 1.0x faster.
     -  Number of Gemm shapes with perf. under 1.0x - 7 / 28
     - Number of Gemm shapes with perf. under 0.9x - 2 / 28
 
Numbers: 

`Benchmark Serving`:

Machine : L40S x 1

Command : 
`python3 -m vllm.entrypoints.openai.api_server --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8`
```
python benchmarks/benchmark_serving.py \
    --backend openai \
    --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 1 \
    --num-prompts 200 \
    --port 8000
```

Cutlass Kernels:
```
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  201.75    
Total input tokens:                      42659     
Total generated tokens:                  40376     
Request throughput (req/s):              0.99      
Input token throughput (tok/s):          211.44    
Output token throughput (tok/s):         200.13    
---------------Time to First Token----------------
Mean TTFT (ms):                          28.72     
Median TTFT (ms):                        28.00     
P99 TTFT (ms):                           58.57     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.63     
Median TPOT (ms):                        15.57     
P99 TPOT (ms):                           18.71     
---------------Inter-token Latency----------------
Mean ITL (ms):                           15.57     
Median ITL (ms):                         15.19     
P99 ITL (ms):                            31.91     
==================================================
```

Pytorch Kernels:

```
============ Serving Benchmark Result ============
Successful requests:                     200       
Benchmark duration (s):                  202.11    
Total input tokens:                      42659     
Total generated tokens:                  40817     
Request throughput (req/s):              0.99      
Input token throughput (tok/s):          211.07    
Output token throughput (tok/s):         201.95    
---------------Time to First Token----------------
Mean TTFT (ms):                          30.49     
Median TTFT (ms):                        28.90     
P99 TTFT (ms):                           60.36     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          16.68     
Median TPOT (ms):                        16.57     
P99 TPOT (ms):                           20.22     
---------------Inter-token Latency----------------
Mean ITL (ms):                           16.61     
Median ITL (ms):                         16.17     
P99 ITL (ms):                            34.07     
==================================================
```

`Gemm benchmarks`

- L40S x 1
- command : `python3 benchmarks/cutlass_benchmarks/w8a8_benchmarks.py  --dtype fp8 model_bench --batch-sizes {1,16,32,64,128,256,512}`

<google-sheets-html-origin><style type="text/css"><!--td {border: 1px solid #cccccc;}br {mso-data-placement:same-cell;}--></style>
float8_e4m3fn meta-llama/Llama-2-7b-hf-TP1 |   |   |  
-- | -- | -- | --
shapes | pytorch with and without fp8 fast accum best time (us) | cutlass_fp8_fp8_bf16_scaled_mm (us) | speedup
MKN=(1x4096x12288) | 33.00 | 15.40 | 2.14
MKN=(1x4096x4096) | 31.30 | 8.60 | 3.64
MKN=(1x4096x22016) | 71.30 | 44.90 | 1.59
MKN=(1x11008x4096) | 76.40 | 15.70 | 4.87
MKN=(16x4096x12288) | 33.20 | 16.60 | 2.00
MKN=(16x4096x4096) | 31.40 | 8.80 | 3.57
MKN=(16x4096x22016) | 72.00 | 49.50 | 1.45
MKN=(16x11008x4096) | 76.40 | 16.70 | 4.57
MKN=(32x4096x12288) | 19.80 | 18.10 | 1.09
MKN=(32x4096x4096) | 11.40 | 9.70 | 1.18
MKN=(32x4096x22016) | 44.00 | 57.20 | 0.77
MKN=(32x11008x4096) | 20.30 | 19.40 | 1.05
MKN=(64x4096x12288) | 22.30 | 22.80 | 0.98
MKN=(64x4096x4096) | 14.20 | 12.70 | 1.12
MKN=(64x4096x22016) | 56.10 | 72.10 | 0.78
MKN=(64x11008x4096) | 33.50 | 26.50 | 1.26
MKN=(128x4096x12288) | 47.50 | 37.90 | 1.25
MKN=(128x4096x4096) | 18.40 | 19.70 | 0.93
MKN=(128x4096x22016) | 94.80 | 105.90 | 0.90
MKN=(128x11008x4096) | 43.90 | 45.30 | 0.97
MKN=(256x4096x12288) | 85.20 | 91.40 | 0.93
MKN=(256x4096x4096) | 26.50 | 28.00 | 0.95
MKN=(256x4096x22016) | 169.40 | 183.60 | 0.92
MKN=(256x11008x4096) | 75.40 | 70.10 | 1.08
MKN=(512x4096x12288) | 139.60 | 140.20 | 1.00
MKN=(512x4096x4096) | 45.50 | 42.60 | 1.07
MKN=(512x4096x22016) | 300.50 | 318.50 | 0.94
MKN=(512x11008x4096) | 122.40 | 114.30 | 1.07
  |   |   |  
torch.float8_e4m3fn meta-llama/Llama-3-8b-TP1 |   |   |  
  |   |   |  
MKN=(1x4096x6144) | 31.40 | 10.30 | 3.05
MKN=(1x4096x4096) | 31.30 | 8.60 | 3.64
MKN=(1x4096x28672) | 181.90 | 170.40 | 1.07
MKN=(1x14336x4096) | 98.00 | 19.10 | 5.13
MKN=(16x4096x6144) | 31.40 | 11.40 | 2.75
MKN=(16x4096x4096) | 31.30 | 8.80 | 3.56
MKN=(16x4096x28672) | 186.80 | 176.00 | 1.06
MKN=(16x14336x4096) | 98.10 | 20.90 | 4.69
MKN=(32x4096x6144) | 13.20 | 12.40 | 1.06
MKN=(32x4096x4096) | 11.50 | 9.70 | 1.19
MKN=(32x4096x28672) | 188.10 | 180.80 | 1.04
MKN=(32x14336x4096) | 27.50 | 26.00 | 1.06
MKN=(64x4096x6144) | 15.30 | 16.10 | 0.95
MKN=(64x4096x4096) | 14.20 | 12.70 | 1.12
MKN=(64x4096x28672) | 199.90 | 190.40 | 1.05
MKN=(64x14336x4096) | 36.20 | 35.00 | 1.03
MKN=(128x4096x6144) | 21.90 | 22.80 | 0.96
MKN=(128x4096x4096) | 18.40 | 19.80 | 0.93
MKN=(128x4096x28672) | 200.60 | 202.10 | 0.99
MKN=(128x14336x4096) | 58.20 | 57.80 | 1.01
MKN=(256x4096x6144) | 36.30 | 38.20 | 0.95
MKN=(256x4096x4096) | 26.60 | 27.90 | 0.95
MKN=(256x4096x28672) | 224.00 | 253.70 | 0.88
MKN=(256x14336x4096) | 96.70 | 90.70 | 1.07
MKN=(512x4096x6144) | 85.10 | 69.30 | 1.23
MKN=(512x4096x4096) | 45.20 | 42.70 | 1.06
MKN=(512x4096x28672) | 402.00 | 435.50 | 0.92
MKN=(512x14336x4096) | 158.70 | 149.50 | 1.06
  |   |   |  
torch.float8_e4m3fn meta-llama/Llama-2-13b-hf-TP1 |   |   |  
  |   |   |  
MKN=(1x5120x15360) | 42.80 | 22.50 | 1.90
MKN=(1x5120x5120) | 38.00 | 10.50 | 3.62
MKN=(1x5120x27648) | 219.60 | 206.00 | 1.07
MKN=(1x13824x5120) | 94.80 | 20.00 | 4.74
MKN=(16x5120x15360) | 43.10 | 24.80 | 1.74
MKN=(16x5120x5120) | 38.00 | 11.60 | 3.28
MKN=(16x5120x27648) | 224.10 | 213.00 | 1.05
MKN=(16x13824x5120) | 94.80 | 24.30 | 3.90
MKN=(32x5120x15360) | 33.10 | 27.20 | 1.22
MKN=(32x5120x5120) | 13.10 | 12.60 | 1.04
MKN=(32x5120x27648) | 224.00 | 218.40 | 1.03
MKN=(32x13824x5120) | 29.40 | 27.80 | 1.06
MKN=(64x5120x15360) | 41.50 | 35.70 | 1.16
MKN=(64x5120x5120) | 16.60 | 16.80 | 0.99
MKN=(64x5120x27648) | 234.50 | 229.50 | 1.02
MKN=(64x13824x5120) | 39.40 | 38.50 | 1.02
MKN=(128x5120x15360) | 85.60 | 60.50 | 1.41
MKN=(128x5120x5120) | 24.80 | 25.50 | 0.97
MKN=(128x5120x27648) | 241.70 | 243.10 | 0.99
MKN=(128x13824x5120) | 61.00 | 59.90 | 1.02
MKN=(256x5120x15360) | 103.30 | 126.50 | 0.82
MKN=(256x5120x5120) | 41.60 | 43.70 | 0.95
MKN=(256x5120x27648) | 263.10 | 311.00 | 0.85
MKN=(256x13824x5120) | 108.50 | 108.20 | 1.00
MKN=(512x5120x15360) | 201.40 | 274.00 | 0.74
MKN=(512x5120x5120) | 102.20 | 80.00 | 1.28
MKN=(512x5120x27648) | 468.00 | 515.80 | 0.91
MKN=(512x13824x5120) | 199.80 | 212.90 | 0.94
  |   |   |  
float8_e4m3fn meta-llama/Llama-2-70b-hf-TP1 |   |   |  
  |   |   |  
MKN=(1x8192x10240) | 58.30 | 22.40 | 2.60
MKN=(1x8192x8192) | 58.10 | 19.30 | 3.01
MKN=(1x8192x57344) | 725.50 | 692.60 | 1.05
MKN=(1x28672x8192) | 367.30 | 353.90 | 1.04
MKN=(16x8192x10240) | 58.40 | 24.70 | 2.36
MKN=(16x8192x8192) | 58.20 | 22.10 | 2.63
MKN=(16x8192x57344) | 738.20 | 702.50 | 1.05
MKN=(16x28672x8192) | 370.70 | 357.60 | 1.04
MKN=(32x8192x10240) | 38.50 | 27.30 | 1.41
MKN=(32x8192x8192) | 25.40 | 25.70 | 0.99
MKN=(32x8192x57344) | 748.50 | 713.30 | 1.05
MKN=(32x28672x8192) | 357.80 | 361.90 | 0.99
MKN=(64x8192x10240) | 48.90 | 39.90 | 1.23
MKN=(64x8192x8192) | 35.20 | 35.20 | 1.00
MKN=(64x8192x57344) | 761.40 | 739.40 | 1.03
MKN=(64x28672x8192) | 356.50 | 370.40 | 0.96
MKN=(128x8192x10240) | 88.10 | 69.30 | 1.27
MKN=(128x8192x8192) | 70.60 | 51.30 | 1.38
MKN=(128x8192x57344) | 771.30 | 763.40 | 1.01
MKN=(128x28672x8192) | 388.10 | 389.10 | 1.00
MKN=(256x8192x10240) | 147.60 | 179.50 | 0.82
MKN=(256x8192x8192) | 90.00 | 86.40 | 1.04
MKN=(256x8192x57344) | 878.70 | 915.00 | 0.96
MKN=(256x28672x8192) | 470.20 | 448.40 | 1.05
MKN=(512x8192x10240) | 280.60 | 301.10 | 0.93
MKN=(512x8192x8192) | 153.20 | 184.90 | 0.83
MKN=(512x8192x57344) | 1,485.00 | 1,604.00 | 0.93
MKN=(512x28672x8192) | 710.50 | 728.40 | 0.98


---

<details>
<!-- inside this <details> section, markdown rendering does not work, so we use raw html here. -->
<summary><b> PR Checklist (Click to Expand) </b></summary>

<p>Thank you for your contribution to vLLM! Before submitting the pull request, please ensure the PR meets the following criteria. This helps vLLM maintain the code quality and improve the efficiency of the review process.</p>

<h3>PR Title and Classification</h3>
<p>Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Model]</code> for adding a new model or improving an existing model. Model name should appear in the title.</li>
    <li><code>[Frontend]</code> For changes on the vLLM frontend (e.g., OpenAI API server, <code>LLM</code> class, etc.) </li>
    <li><code>[Kernel]</code> for changes affecting CUDA kernels or other compute kernels.</li>
    <li><code>[Core]</code> for changes in the core vLLM logic (e.g., <code>LLMEngine</code>, <code>AsyncLLMEngine</code>, <code>Scheduler</code>, etc.)</li>
    <li><code>[Hardware][Vendor]</code> for hardware-specific changes. Vendor name should appear in the prefix (e.g., <code>[Hardware][AMD]</code>).</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

<h3>Code Quality</h3>

<p>The PR need to meet the following code quality standards:</p>

<ul>
    <li>We adhere to <a href="https://google.github.io/styleguide/pyguide.html">Google Python style guide</a> and <a href="https://google.github.io/styleguide/cppguide.html">Google C++ style guide</a>.</li>
    <li>Pass all linter checks. Please use <a href="https://github.com/vllm-project/vllm/blob/main/format.sh"><code>format.sh</code></a> to format your code.</li>
    <li>The code need to be well-documented to ensure future contributors can easily understand the code.</li>
    <li>Include sufficient tests to ensure the project to stay correct and robust. This includes both unit tests and integration tests.</li>
    <li>Please add documentation to <code>docs/source/</code> if the PR modifies the user-facing behaviors of vLLM. It helps vLLM user understand and utilize the new features or changes.</li>
</ul>

<h3>Notes for Large Changes</h3>
<p>Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with <code>rfc-required</code> and might not go through the PR.</p>

<h3>What to Expect for the Reviews</h3>

<p>The goal of the vLLM team is to be a <i>transparent reviewing machine</i>. We would like to make the review process transparent and efficient and make sure no contributor feel confused or frustrated. However, the vLLM team is small, so we need to prioritize some PRs over others. Here is what you can expect from the review process: </p>

<ul>
    <li> After the PR is submitted, the PR will be assigned to a reviewer. Every reviewer will pick up the PRs based on their expertise and availability.</li>
    <li> After the PR is assigned, the reviewer will provide status update every 2-3 days. If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM team.</li>
    <li> After the review, the reviewer will put an <code> action-required</code> label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.</li>
    <li> Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.
 </li>
</ul>

<h3>Thank You</h3>

<p> Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM. Your contributions make vLLM a great tool for everyone! </p>


</details>




## Discussion

### github-actions[bot] · 2024-07-23T07:27:29Z

https://github.com/vllm-project/vllm/pull/6677#issuecomment-2244459050

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which consists a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of default ones by unblocking the steps in your `fast-check` build on Buildkite UI. 

Once the PR is approved and ready to go, please make sure to run full CI as it is required to merge (or just use auto-merge).

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀

### varun-sundar-rabindranath · 2024-07-23T07:32:35Z

https://github.com/vllm-project/vllm/pull/6677#issuecomment-2244468375

Kernel Benchmark Heatmap:

![model_bench-torch float8_e4m3fn-1721701700-selected_heatmap](https://github.com/user-attachments/assets/1891d893-c13a-42d9-875b-cddb01ccb4c2)

Pointers on how to read the heatmap:
X-Axis -  Cutlass Ops.
Y-Axis - All the GEMM Shapes in (M x N x K) format.
The Darker the cell, the Better the algorithm performed for that GEMM shape. Each row (GEMM-Shape) is normalized to be between 0.0 and 1.0. The best algorithm has a value of 1.0.
Annotations:
White boxes - Configurations selected and added in this PR
Green box - Fallback gemm
Thin yellow boxes - Other good configs.

Cutlass Op naming convention:
autogen_cutlass2x_scaled_mm_dq_sm80_128x64x128_64x64x64_16x8x32_ThreadBlockSwizzleStrreamK_kGemmSplitKParallel_5_OpMultiplyAddFastAccum_fp8 refers to an Op constructed with,

Tile Shape : 128x64x128
Warp Shape : 64x64x64
Instruction Shape : 16x8x32
Thread block swizzle : ThreadBlockSwizzleStreamK
Gemm mode : kGemmSplitKParallel
Main loop stages : 5
OpMultiplyAddFastAccum : FP8MathOperator
fp8 : Gemm input datatype

### varun-sundar-rabindranath · 2024-07-23T16:22:40Z

https://github.com/vllm-project/vllm/pull/6677#issuecomment-2245683669

Thanks for the reviews @mgoin @comaniac. 
I discovered a bug in the code, where I used M instead of N for dispatching specialized configs. The fix improves the numbers - I have updated the PR description with the numbers and edited the "performance takeaway" section so it is easier to assimilate. Please take a look. Sorry, about the initial mishap. 

Even with the fix, pytorch scaled MM is better for some shapes (mostly big Ms) and your suggestion about using pytorch for big Ms is still valid. 

### varun-sundar-rabindranath · 2024-07-24T14:28:19Z

https://github.com/vllm-project/vllm/pull/6677#issuecomment-2248145719

/ready

## Reviews

### comaniac · 2024-07-23T15:06:22Z

https://github.com/vllm-project/vllm/pull/6677#pullrequestreview-2194272572

LGTM. btw looks like the PyTorch kernel is still faster with large batch sizes? In this case (not in this PR) does that make sense to dispatch between cutlass and PyTorch kernels based on the batch size?

### mgoin · 2024-07-23T15:53:40Z

https://github.com/vllm-project/vllm/pull/6677#pullrequestreview-2194395226

Great work! Agreed with Cody, maybe it is worth using torch._scaled_mm for per-tensor scale case for large M

### mgoin · 2024-07-24T14:39:25Z

https://github.com/vllm-project/vllm/pull/6677#pullrequestreview-2196951080


