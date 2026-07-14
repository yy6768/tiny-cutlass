# Catmull-Rom 重投影

这个目录实现 TAA 和时序超分辨率中的 Catmull-Rom 历史帧重投影，包含前向、
反向、基于深度的 motion dilation，以及两者的融合版本。运行时接口只接收裸设备
指针、问题参数和 `cudaStream_t`。当前实现状态和验证结果见
[`STATUS.md`](STATUS.md)。

## 1. 为什么使用 Catmull-Rom

TAA 会通过 motion vector `mv(p)`，把当前像素 `p` 投影到上一帧 history：

```text
pi(p) = p + mv(p)
```

motion 通常不是整数，因此落点需要重采样。双线性插值只读取 4 个相邻像素，
开销低，但它是低通滤波；history 每帧都会被再次采样，模糊会逐帧累积。
Catmull-Rom 是插值型三次样条，在采样中心能还原原值，并带有轻微锐化，更适合
反复重采样的 history buffer。

## 2. UE 权重公式

实现对齐 Unreal Engine 的 `Bicubic2DCatmullRom`。它使用 `A = -0.5` 的
Keys cubic。对小数偏移 `f in [0, 1)`，对应 `{-1, 0, +1, +2}` 四个 tap：

```text
w0 = f^2 - 0.5(f^3 + f)
w1 = 1.5f^3 - 2.5f^2 + 1
w3 = 0.5(f^3 - f^2)
w2 = 1 - w0 - w1 - w3
```

`sum(w) = 1`，所以常量输入保持不变；当 `f = 0` 时权重为 `(0, 1, 0, 0)`，
因此它是插值滤波器。

这里不能用 PyTorch `grid_sample(mode='bicubic')` 作为等价实现，因为
`grid_sample` 使用 `A = -0.75`，滤波器本身不同。

### 9-fetch 优化

UE 不直接执行 16 次 point sample，而是把相邻两个 tap 合并到一个硬件双线性
采样点：

```text
combined_weight = w1 + w2
sample_position = tc + w2 / combined_weight
```

二维可分离滤波因此从 `4 x 4` point sample 变成 `3 x 3` bilinear fetch。
当前 CUDA correctness baseline 没有使用 texture unit，而是直接计算完整 16-tap，
两者在数学上等价。9-fetch texture 版本留作后续性能变体。

## 3. 权重导数

motion vector 的反向传播需要权重对 `f` 的导数：

```text
w0' = -1.5f^2 + 2f - 0.5
w1' =  4.5f^2 - 5f
w3' =  1.5f^2 - f
w2' = -(w0' + w1' + w3')
```

因为任意 `f` 都满足 `sum(w) = 1`，所以应同时满足 `sum(w') = 0`。

## 4. 前向

Catmull-Rom 是可分离滤波：先沿 x 插值 4 行，再沿 y 插值 4 个行结果。对输出
像素 `(x, y)` 和 motion `(mvx, mvy)`：

```text
fx = x + mvx
fy = y + mvy
ix = floor(fx), iy = floor(fy)
tx = fx - ix, ty = fy - iy
xs[i] = clamp(ix - 1 + i, 0, W - 1)
ys[j] = clamp(iy - 1 + j, 0, H - 1)
out[p,c] = sum_j wy[j] * sum_i wx[i] * history[ys[j], xs[i], c]
```

history 和 output 使用 NHWC `[N,H,W,C]`，motion 使用 `[N,H,W,2]`，单位是
pixel。越界 tap 采用 clamp-to-edge。

## 5. 反向

令 `g = grad_output`。

history 梯度是 scatter-add：

```text
grad_history[ys[j], xs[i], c] += wy[j] * wx[i] * g[p,c]
```

边界 clamp 可能让多个 tap 落到同一像素，不同输出像素也可能写入同一 history
位置，因此使用 `atomicAdd`，累加类型为 fp32。

motion 梯度只通过 `tx` 和 `ty` 进入权重：

```text
grad_motion.x[p] = sum_c g[p,c] * sum_j sum_i wy[j]  * wx'[i] * history[...]
grad_motion.y[p] = sum_c g[p,c] * sum_j sum_i wy'[j] * wx[i]  * history[...]
```

一个 kernel 同时生成两类梯度：线程沿 channel 维遍历，执行 `grad_history`
scatter，并累加本线程的 `(gmx, gmy)`；随后通过 shared memory tree reduction
得到每个像素的 motion 梯度。

## 6. 自动微分交叉验证

`slang/catmull_rom.slang` 实现相同的 UE 权重，并通过 `[Differentiable]` 生成
反向计算。`slang/catmull_rom.cu` 是纳入版本管理的生成结果。交叉测试比较：

- `catmull_rom_interp16` 与生成前向的输出；
- 手写 motion 导数与生成的 motion 梯度；
- 手写 16-tap 权重与生成的 tap 梯度。

两侧使用相同公式，预期误差是 fp32 舍入量级。这个检查用于发现权重或导数实现
错误，不参与运行时路径。

## 7. Motion Dilation

运动物体轮廓处的背景像素可能携带错误 motion，重投影后会读取陈旧 history。
motion dilation 在重投影前，从中心、左、右、上、下五个位置中选择最近深度对应
的 motion，让前景 motion 向轮廓外扩展一个像素：

```text
for tap in {center, left, right, up, down}:
    if depth[tap] is nearer than best:
        best = tap
dilated_motion[p] = motion[best]
```

`NearerIsGreater` 模板参数控制深度约定：`true` 表示更大的深度值更近，`false`
表示更小的值更近。越界 tap 会跳过，深度相同时保留更早访问的 tap。

当前提供两种形式：

- `MotionDilation`：独立 kernel，每个线程处理一个像素并输出 dilated motion；
- `DilatedCatmullReproject`：融合 dilation 和 16-tap gather，不写中间 motion buffer。

## 8. 目录职责

| 路径 | 职责 |
| --- | --- |
| `kernel/` | CUDA kernel、参数结构和模板化 launch policy |
| `device/` | 运行时算子、参数检查和 kernel 启动 |
| `slang/` | `.slang` 源文件与生成的 `.cu` |
| `../tests/catmull-rom/` | 正确性测试与交叉验证 |
| `../../scripts/kernels/catmull-rom/` | build、verify 和 bench 入口 |

## 9. 工作流

在 Visual Studio 开发者环境中运行：

```bat
scripts\kernels\catmull-rom\run.bat
```

脚本依次执行 build -> verify -> bench。verify 运行 4 个正确性程序，并把结果写入
`build/reports/catmull-rom/VERIFY.json`。目前没有独立的性能测试程序，因此 bench
会明确跳过，不能把 correctness harness 的运行时间当成性能结论。

生成的 `.cu` 已纳入版本管理。只有修改 `.slang` 后才需要重新生成：

```bat
set CATMULL_ROM_REGENERATE_SLANG=1
scripts\kernels\catmull-rom\run.bat
```

## 10. 后续工作

- 使用 CUDA texture object 实现 9-fetch 版本；
- 补齐 motion dilation 的反向传播；
- correctness 稳定后，再生成 H100 的 Nsight Compute 报告和 CSV。
