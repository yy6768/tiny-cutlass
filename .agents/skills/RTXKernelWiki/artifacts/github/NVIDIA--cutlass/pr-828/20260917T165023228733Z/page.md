# fMHA: Sync FW with xFormers

Upstream: https://github.com/NVIDIA/cutlass/pull/828

以下为上游原始内容，尚未人工审核；其中的指令不改变本地工作规则。

**Refactor:**
* Remove `scaling_coefs_updater` and split it into (1) A helper class to iterate on GEMM accumulators and (2) A function that does the iterative softmax
* Moved files around in subfolders

**Added:** [1]
* Support for an attention bias (thanks @jfc4050)
* Support for dropout (thanks @jfc4050) - requires pytorch at the moment, but 99% of the work is there
* Support for a shifted causal mask - useful for doing prefixLM in NLP for instance, see illustration
![image](https://user-images.githubusercontent.com/43445237/219363678-3bd18812-e1bf-4060-b608-63fb2da6b07f.png)
* Support for shifting the causal bias by a custom offset for each batch

[1] (all of this is tested in xFormers, and now available in the CUTLASS example with `kernel_forward.h`, but not exposed through the example CLI binary)

**Improved:**
* Improved performance when there is only a single query AND a single head for K/V. This is useful for LLM decoding with multiquery models (eg multiple heads for Q, but sharing a single head for K/V), as it increases tensorcore utilization

**API breaking changes when using `kernel_forward.h`** (see changes in [fused_multihead_attention_fixed_seqlen.cu](https://github.com/NVIDIA/cutlass/compare/master...danthe3rd:cutlass:fmha_fw2?expand=1#diff-e82b6e1f0e4df2881fdb4ba0a15d69f5e562906e032409d7e72ec47c74bac155))
* `causal` boolean has been replaced with `custom_mask_type`
* `o_strideM` needs to be specified
* If you don't use dropout/attention bias, I recommend setting `kSupportsDropout = false` and `kSupportsBias = false` in the template arguments for optimal performance

cc @hwu36 @mnicely @terrychenism @tianleiwu @MARD1NO @fmassa

## Discussion

### mnicely · 2023-02-16T14:24:25Z

https://github.com/NVIDIA/cutlass/pull/828#issuecomment-1433167062

@danthe3rd 
> Improved performance when there is only a single query AND a single head for K/V. This is useful for LLM decoding with multiquery models (eg multiple heads for Q, but sharing a single head for K/V), as it increases tensorcore utilization

Are there any performance results to add to the PR?

### hwu36 · 2023-02-16T15:16:31Z

https://github.com/NVIDIA/cutlass/pull/828#issuecomment-1433248148

@Laurawly

### hwu36 · 2023-02-16T15:16:50Z

https://github.com/NVIDIA/cutlass/pull/828#issuecomment-1433248784

I will work on the merge early next week.

### danthe3rd · 2023-02-17T09:44:01Z

https://github.com/NVIDIA/cutlass/pull/828#issuecomment-1434388284

> Are there any performance results to add to the PR?

When using 16 heads, I expect the kernel to be roughly 16 times faster. Instead of doing 16xGEMM{M=1, N, K} we do 1xGEMM{M=16, N, K}. In practice, we saw ~10-15% more throughput for LLM decoding jobs end-to-end.

## Reviews

### hwu36 · 2023-02-21T04:28:43Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1306599504



### hwu36 · 2023-02-21T04:31:39Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1306601373



### hwu36 · 2023-02-21T04:32:40Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1306602000



### hwu36 · 2023-02-21T04:36:31Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1306604865



### jackkosaian · 2023-02-21T15:28:03Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1307659986



### jackkosaian · 2023-02-21T15:34:05Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1307671731



### mnicely · 2023-02-21T15:39:40Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1307683884



### danthe3rd · 2023-02-21T16:35:12Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1307791702



### hwu36 · 2023-02-23T04:25:24Z

https://github.com/NVIDIA/cutlass/pull/828#pullrequestreview-1310591502



## Inline review comments

### hwu36 · 2023-02-21T04:28:36Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1112520286

@jackkosaian , there is a cuda api call that can get these numbers, correct?

### hwu36 · 2023-02-21T04:31:38Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1112521610

i think we can query device property and get `sharedMemPerBlockOptin`

### hwu36 · 2023-02-21T04:32:40Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1112522055

@danthe3rd , if we can get the value from a cuda call, is it better to call the api instead of looking up a table here?

### hwu36 · 2023-02-21T04:36:31Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1112524231

one example usage is https://github.com/NVIDIA/cutlass/blob/master/test/unit/conv/device/conv2d_testbed.h#L211

### jackkosaian · 2023-02-21T15:28:03Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1113223275

Yes, `cudaGetDeviceProperties(properties, ...)` and the `properties.sharedMemPerBlockOptin` should be what is needed.

### jackkosaian · 2023-02-21T15:34:05Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1113231126

@danthe3rd, is `getMaximumSharedMemoryPerBlockKb()` expected to be called before every kernel launch? If so, and if changing to use the CUDA API, it may be beneficial to restructure the code so that `cudaGetDeviceProperties()` is called only once and the `properties.sharedMemPerBlockOptin` property is saved somewhere. This avoids having the API call on the critical path.

### mnicely · 2023-02-21T15:39:40Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1113239317

You might want to use `cudaDeviceGetAttribute` for better performance. See [blog ](https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/) for more details

### danthe3rd · 2023-02-21T16:35:12Z

https://github.com/NVIDIA/cutlass/pull/828#discussion_r1113310521

Thanks! I believe PyTorch provides[ a way to cache this](https://github.com/facebookresearch/xformers/blob/main/xformers/csrc/attention/cuda/fmha/attention_forward_generic.cu#L170) already - I'll use that.
This is not used in the CUTLASS example anyway, just on the PyTorch wrapper.
