# 02 split-KV 实现与验证计划

当前实现保留官方FA1的外层KV、内层Q结构和CUTLASS 2.x的
device / kernel / threadblock / warp / epilogue分工。Q/K/V使用TensorOp shared swizzle，
P由score fragment直接进入PV，不经过shared memory。

当前循环参考：官方flash-attention v1.0.9，提交
`6d48e14a6c2f551db96f0badc658a6279a929df3`，`fmha_fprop_kernel_1xN.h`。
早期LeetCUDA参考只保留warp分工层面的意义。本实现使用CUTLASS 4.5.2头文件中的
2.x stock iterators和warp MMA，不移植PTX宏或Torch ownership。安装机器为SM89，
SM80仅为policy目标。

## 契约

- FP16 输入/输出，QK、PV 累加和 online softmax 统计为 FP32。
- 非因果、无 dropout/bias，连续 `[B,S,H,D]`，允许不同 Sq/Sk 与 sequence tail。
- 显式实例 D=Dv 为 32/64/96/128；其他 dtype/layout/head shape 明确拒绝。
- device 按 M=48KiB/sizeof(Element) 计算 Bc=ceil(M/(4D))、Br=min(Bc,D)。
  D=32/64/96/128 对应 Br/Bc=32/192、64/96、64/64、48/48。
- D=32使用两个warp，其余形状使用四个copy warp；每个计算warp负责16个Q行、
  全部KV列和全部输出通道。D=128的Br=48，只需三个计算warp，第四个warp仍参与
  Q/K/V搬运和CTA同步，防止raked ThreadMap的迭代次数截断漏掉shared末尾行。
  D=128的物理KV宽度补齐到64，逻辑Bc仍是48。
- 不分配全局S/P或跨CTA split workspace，P留在FP32 score fragment中，
  通过`MmaTensorOpFragmentIterator`转换为FP16的PV寄存器操作数。
  中间归一化O、m、l使用FP32 workspace，形状分别为[B,H,padded_Sq,Dv]及两个
  [B,H,padded_Sq]；padded_Sq=ceil(Sq/Br)*Br。容量由device查询，调用方分配。
- global→shared使用CUTLASS `PredicatedTileAccessIterator`、
  `RegularTileAccessIterator`和`arch::cp_async_zfill`。Q/K使用crosswise、
  V使用congruous TensorOp shared layout；物理尾部零填充，Q/K/V各占一块shared。
  外层每轮KV搬一次K/V，内层扫描不同Q tiles并重新搬入Q；不再有Q/P union。
- softmax 和输出更新按 FA1 Algorithm 1 第10～12行：先用当前 tile 最大值计算
  tilde P/tilde l，再合并 m/l，每轮 O=(alpha*l_old*O_old+beta*tilde P V)/l_new。
  不对 P 提前除以分母；最终 epilogue 只转换并写回，不接收 denominator。
- 一CTA负责一个batch/head，按官方num_splits=1组织外层KV、内层Q。
  首轮KV清零O/l并设m=-inf，不读取旧workspace；以后每轮恢复该Q tile的FP32状态。
  中间轮写回FP32归一化O，最后一轮写FP16输出；每轮写回m/l。
  Q/KV完整块在前、尾块在后；输入copy独立做真实尾块predication。
  尚未迁入官方LSE编码、Q方向CTA分组或流水线；不把这些后续选择混入当前基线。
- cuDNN SDPA parity：MAE<=1e-3，max abs<=1e-2，所有输出 finite。

## 具体实现

1. DefaultFlashAttn 模板工厂组装 layout、stock global/shared iterators、两个 warp MMA、
   mainloop 和 epilogue；保留简单全局类型，具体形状只在 launcher 实例化。
2. warp 类解释 CUTLASS C fragment 的坐标，调用 IteratorA/B、transform 与 MmaTensorOp。
3. threadblock持有KV→Q完整双层循环、copy与同步；内部QK、softmax、PV和归一化公式保持。
4. 标准MMA IteratorC读取/写回FP32中间O；自定义RowIterator处理m/l的行映射。
   epilogue只负责中间store或最终类型转换/store；device执行workspace等参数检查并launch。
5. 接到现有共享 Kernel 注册，launch TU 移至 device/flash_attn.cu，更新 CMake。

## 验证与证据

构建目录 `build/split-kv-study/cmake`；脚本入口
`scripts/kernels/attention/02-split-kv-attention.bat`，验证/计时驱动在
`csrc/tests/flash-attention/verify.py`、`bench.py`。
验证 D、B/H stride、Sq/Sk tail、不同 seed/输入幅度、非默认 stream、非法配置拒绝；
共享 --kernel=all 对00/01/02做回归。build→verify→bench，任何验证失败都停止。

## 先读懂官方循环，再对应本地

先看[官方device_1xN_loop](https://github.com/Dao-AILab/flash-attention/blob/6d48e14a6c2f551db96f0badc658a6279a929df3/csrc/flash_attn/src/fmha_fprop_kernel_1xN.h#L688-L700)：
`loop_step_idx`是KV块编号，首块/中间块/末块通过模板布尔参数区分。
进入[device_1xN_](https://github.com/Dao-AILab/flash-attention/blob/6d48e14a6c2f551db96f0badc658a6279a929df3/csrc/flash_attn/src/fmha_fprop_kernel_1xN.h#L382-L407)，
循环变量`l`才是Q tile索引，不是softmax分母ℓ。它每次推进Q，同时恢复对应行的旧输出。
K/V在这层Q循环外加载，因此不同Q tiles共享当前KV块。

接着看[官方归一化和输出](https://github.com/Dao-AILab/flash-attention/blob/6d48e14a6c2f551db96f0badc658a6279a929df3/csrc/flash_attn/src/fmha_fprop_kernel_1xN.h#L625-L645)：
`inv_sum`在每轮计算，并非只在`is_final_write`时计算。这个分支选择的是写入中间
FP32输出还是最终输出，不是选择是否归一化。

本地入口是[kernel/flash_attn.h](kernel/flash_attn.h)，CTA只选batch/head。
[threadblock/flash_attn_mma.h](threadblock/flash_attn_mma.h)的`operator()`用两层循环
直接展开上述调用关系，避免再包一层自定义Loader：

```text
for j in KV tiles:
    copy_key_value(K_j, V_j)
    reset Q / output / state iterators
    for q in Q tiles:
        first KV: O=0, m=-inf, l=0; otherwise: load FP32 state
        copy_query(Q_q)
        QK -> tilde P / row statistics -> PV -> normalize O
        intermediate KV: store FP32 O; final KV: store FP16 O
        store m/l -> synchronize -> advance Q/state/output
    advance K/V
```

这里复制的是官方循环与中间结果生命周期，m/l仍按论文显式保存，不改成官方LSE。
也暂不做Q方向CTA分组。`B*H`很小时并行度有限，是这份教学基线的边界；
不能把结构对齐称为性能优化，也不能沿用固定Q版本的计时/NCU结论。

在通过 cuDNN 的相同 shape 上用 NCU 采 full/source（若可用），保留 binary/source hash、
parity log、native report、CSV/HTML。当前已有的02-tiled报告属于废弃实现，不能复用。
博客沿现有 Overview 更新源码阅读顺序和实測 Profile，保留用户前言和人工后记位置。

## Shared swizzle / register P 验证（2026-09-27）

- SM89 Release构建通过；[42组reference parity与2组拒绝检查](../../../build/split-kv-study/validation/swizzle-fragment-final/verification.json)通过，
  包含D=128的KV尾块、零Q/K与NaN workspace复用。
- [5×50次普通计时](../../../build/split-kv-study/validation/swizzle-fragment-final/benchmark.csv)：
  四组02分别为0.055501、0.750592、2.168810、3.382170 ms。
  布局修改明显快于上一版02，但少CTA的中等长度仍慢于00/01；不把swizzle等同于充分并行。
- [NCU定点复测](../../../build/split-kv-study/profile/20260927-swizzle-register-p/REPORT.md)：
  `(1,4,1024,64/128)`两组shared-load、shared-store bank conflict计数均为0；
  原生`.ncu-rep`和raw CSV保留在该目录。不能据此推断未测形状也为0。

## 上一版循环验证（2026-09-26）

- Release共享/独立入口构建通过，RTX4070 Laptop SM89。
- 42组reference命令+2组非法shape拒绝通过；02专项38组最大MAE=2.33128e-5，
  max_abs=9.76562e-4。每组还覆盖非默认stream重复调用，以及NaN workspace重用。
- 缺失、容量不足、未对齐workspace均明确返回cudaErrorInvalidValue。
- 四个D的memcheck通过；D128多head、Q/KV尾块的racecheck、synccheck、initcheck通过。
- 同一二进制完成3轮、每轮20次迭代的普通benchmark；未重新采NCU。
  这是算法结构基线，少量B/H时CTA数量少，小批量计时变慢，不宣称是性能优化。
- 新证据：[验证说明](../../../build/split-kv-study/validation/20260926-fa1-kv-outer/README.md)。

## 历史：2026-09-23 FA1逐轮归一化（固定Q版本）

- Mainloop 显式计算 tilde m/tilde P/tilde l、alpha/beta 和归一化 O；
  tilde P V 使用独立 FP32 fragment，再与 alpha*l_old*O_old 合并并除以 l_new。
- 最终 epilogue 删除 denominator 参数，仅使用标准 fragment/output iterator转换写回。
  03此前继承此epilogue，其原有最终归一化已放回03自己的epilogue，计算语义不变。
- RTX4070 Laptop SM89：02验证42组命令+2组拒绝检查，03回归66组命令+2组拒绝检查，
  全部通过。02专项最大MAE=2.35897e-5、max_abs=9.76562e-4；03专项最大MAE=4.01022e-7。
- 原有4组benchmark shape已在同一验证二进制上完成5轮、每轮50次的计时。
  新证据在 `build/split-kv-study/validation/20260923-fa1-normalized/`，
  `verification.json`、`split-q-regression/verification.json`、`benchmark.json`
  均记录二进制SHA256及完整执行命令；本次未重新采集NCU。
- 博客为 `blogs/flash-attn/02-split-kv.md`，已同步更新公式、Mainloop、Epilogue，
  取消旧后置归一化示意图引用，并区分新计时与旧NCU证据；保留原有前言、后记。

复现命令（仓库根目录，PowerShell；每一步成功后才运行下一步）：

```powershell
cmake --build build/split-kv-study/cmake --config Release --target flash_attention_test split_kv_attention --parallel 4
$attentionExe = 'build/split-kv-study/cmake/csrc/flash-attention/Release/flash_attention_test.exe'
$attentionEvidence = 'build/split-kv-study/validation/20260923-fa1-normalized'
python -B csrc/tests/flash-attention/verify.py --exe $attentionExe --kernel 02-split-kv --output-dir $attentionEvidence
python -B csrc/tests/flash-attention/verify.py --exe $attentionExe --kernel 03-split-q --output-dir "$attentionEvidence/split-q-regression"
python -B csrc/tests/flash-attention/bench.py --exe $attentionExe --kernel 02-split-kv --output-dir $attentionEvidence
```

## 历史进展（2026-09-09，非当前版本验证）

- 五层 CUTLASS 实现和共享注册已接通，独立与共享 executable 均已构建。
- 22组正向命令（30次cuDNN对照）和2组非法配置拒绝已通过。
- 测试输出使用同一个非默认stream上的NaN预填，防止漏写或跨stream初始化竞态。
- 最终二进制已重新验证、benchmark 和采集 NCU full/source；verification、NCU 与
  当前 executable 的 SHA256 一致，报告内嵌的执行源码也与当前文件一致。
- memcheck、racecheck、synccheck 在 D=96、多head与双sequence tail用例上均通过。
- 博客已补齐五层源码阅读与实测 Profile，后记保留给用户；具体数值和报告链接以博客为准。
