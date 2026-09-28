# 03-split-q：任务草案与执行计划

## 任务契约

- 在当前工作区的 `02-split-kv` 上建立对照，不回退它的未提交改动。
- 实现 `03-split-q`，参考 FlashAttention-2 §3.1–3.3、Algorithm 1/2，以及指定 LeetCUDA split-Q 源码。
- CUTLASS 2.x 标准 ThreadMap、global/shared iterator、warp MMA、epilogue 组件；职责按 example 13 的 `B2bMmaBase` / `B2bMmaMultistage` 划分。
- 输入保持 `[B,S,H,D]` FP16，D=Dv∈{32,64,96,128}，任意正 Sq/Sk，非 causal，无 dropout；FP32 累加。
- 严格验收：相同输入、相同 cuDNN SDPA reference，MAE **< 1e-6**；四个现有 benchmark shape 分别要求 `t03/t02 < 0.95`，不通过平均值掩盖退化。
- backward 按原始完整契约实现；既有 02 和 LeetCUDA 接口仅 forward，因此 backward 使用独立梯度 reference，性能不虚构 02 backward 对照。

## 已读现状

- `02-split-kv/device/flash_attn.cu` 按 48 KiB 容量选择 `(Br,Bc)`：D=32→(32,192)、64→(64,96)、96→(64,64)、128→(48,48)。
- 当前源码已经是 grid=(Q tile,H,B)，循环末尾一次归一化。03 不将这些既有行为计作新增优化。
- 当前 02 的 QK warps 切列、PV warps 切输出通道；softmax 最大值/和经 shared memory 跨 warp 汇总，Q/P 共用空间，逐 KV tile 重载 Q。
- 共享 executable：`flash_attention_test`；验证：`csrc/tests/flash-attention/verify.py`；计时：同目录 `bench.py`。
- 当前默认门限 1e-3 不满足任务契约，03 验证必须显式使用严格门限。
- 现有二进制在 (B,H,Sq,Sk,D)=(1,1,256,256,64) 下实测 02 对 cuDNN MAE=5.20706e-6。此为已有二进制的诊断结果，尚未重新构建固定源码，不能当作 03 的结果。

## 实现草案

CTA 固定 Q tile，4 个 warp 沿 Q 行分工，每 warp 处理 16 行及完整 KV tile/输出通道。Q 在循环前加载，K/V 每轮由 Mma 内部搬运。QK 与 PV 继续调用标准 `DefaultMmaTensorOp`，使用同一 warp 行分工；softmax 是 attention 自定义步骤，4-lane shuffle 完成行归约，无跨 warp 统计表。保持 FP32 未归一化 O、m、l，epilogue 最后写回。

example 13 独立审查确认 `MmaTensorOpFragmentIterator` 可将 QK accumulator 的 column-major instruction ordering 转成 PV 的 A fragment（`default_b2b_mma.h:291`）。据此在修改代码前收敛第一候选：Q 独立驻留 shared，P 经该官方 fragment iterator 直接传入 PV，不增加 P shared buffer。若 FP16 概率舍入无法达到精度契约，记录失败并单独设计下一候选，禁止放宽门限、缩小输入范围或更换 reference 来制造通过。

| 层 | 内容 | 官方对应 |
|---|---|---|
| device | 模板 policy 分派、can_implement/initialize/run、launch | example 13 device operator |
| kernel | CTA 定位/早退、KV 循环次数、初始 iterator、Mma+epilogue | example 13 kernel composition |
| threadblock | storage、完整 KV 主循环、搬运、iterator 推进、同步、softmax | `b2b_mma_base.h`、`b2b_mma_multistage.h`；softmax 自定义 |
| warp | 标准 TensorOp iterator/transform/MMA | `DefaultMmaTensorOp` 与当前 02 的已核对映射 |
| epilogue | 最终行归一化、带边界写回 | `FragmentIteratorTensorOp`、`TileIteratorTensorOpCanonical` |

## 可执行计划（代码修改前固定）

1. [x] 读取仓库规则、当前 02、测试、论文及指定源码；并行 agent 核对 baseline 和 example 13。
2. [x] 固定当前 baseline 源码/二进制 hash，重新构建，保存严格精度诊断和既有门限验证；只对通过其 reference 门限的基线计时。
3. [x] 初始化博客前言与 Overview：算法改动、CTA 并行、warp 分工、CUTLASS 调用链、精度与性能证据；已发出 Overview 确认请求，正文待答复。
4. [x] 候选 C01（parent=B00）：新增独立分层 split-Q 实现，注册共享 Kernel 接口和 CMake；更新 verify/bench 和 family 脚本。
5. [x] 对 C01 build→verify；覆盖 02 的全部 shape、边界、输入 scale、复用，以及 03 的 tile 边界；严格 MAE<1e-6。C01失败已记录，未运行该候选 bench/profile。
6. [x] 只对验证通过候选做四个既有 shape 的交错计时；保存每次样本、median、t03/t02 和二进制 hash。C02性能失败，C03四项通过。
7. [x] C02/C03 先写父候选、单一变更假设、预期、结果，再进入下一次 build/verify；同时只保留一个活跃候选。
8. [x] 对最终正确候选做 NCU，保存 .ncu-rep+CSV；Profile 已写入最终采集和明确标注的早期对照。
9. [ ] Forward源码职责审查、scoped diff检查已完成；博客的前言/Overview/五节正文/Profile已完成。后记保留用户填写标记。独立backward已实现，仍在严格数值验证，不能将部分通过当作完成。

## 证据位置与候选记录

- 构建、源码快照、参考文件、验证/计时/NCU：`build/split-q-study/`。
- 可维护候选记录：`docs/03-split-q-results.md`，链接证据、命令、父子关系、失败原因。
- 博客：`blogs/flash-attn/03-split-q.md`。
- 对照 shape：(1,1,256,256,64)、(1,4,1024,1024,64)、(1,4,1024,1024,128)、(16,12,1024,1024,64)。

## 当前状态

历史C03 forward已通过任务门限：58项03严格检查最大MAE=4.01022e-7；当时完整脚本四项 t03/t02=0.587411/0.468053/0.214654/0.434950。66项总验证与2项拒绝检查通过，其NCU中Tensor elapsed activity=21.310961%，MIO throttle=1.179796 cycles/issued instruction。当前B08的forward重新通过66项+2拒绝检查、严格最大MAE仍为4.01022e-7，但没有重新计时；历史C03耗时不能作为B08的性能验收。既有02的9个文件与任务开始快照仍完全一致。

Backward已实现，尚未通过独立训练状态的完整验收：B08正式gate在第12项失败，全输入诊断共9项超限。B07共同O/LSE的58项梯度诊断达到门限，只用于误差归因，不代替正式gate。博客本轮已按用户明确要求从 FlashAttention-2 原文展开五节正文，并把历史 Profile 留在其后；后记仍由用户填写。尚待完成的是端到端梯度验收及当前最终候选的计时/profile，目标不能在此状态标记完成。

## Backward 可执行计划（B01，parent=C03；代码修改前固定）

验收解释：原始契约明确要求backward并行与MAE<1e-6，但02没有backward输入/输出基线，也没有规定O/LSE来自同一forward还是分别生成。因此目前实现采用更严格的独立状态端到端检查；这属于测试设计选择，不能说成用户已经确认的额外要求。相同状态的算子检查始终单独标为诊断。在用户回答验收口径前，不把诊断切换为通过门禁，也不放宽任何误差阈值。

原始契约已经要求 backward 的 seqlen 并行，继续完成实现，不把 forward 达标当作完整任务完成。参考 FA2 Algorithm 2；仅增加该算子的训练状态和梯度路径，不加入 causal、dropout 或其他 dtype。

1. [x] `Tensors` 增加可选 FP32 `logsumexp` 输出（连续 `[B,H,Sq]`），现有 forward 默认空指针。独立 `BackwardTensors`/`BackwardKernel` raw-pointer API；核心不引入 Torch ownership。
2. [x] B01：每 CTA 固定 64 行 KV，沿 Q 每 64 行扫描。四个 warp 各拥有 16 行 KV，计算 `K Q^T`、`V dO^T`、`P^T dO`、`dS^T Q`；dK/dV 用 FP32 寄存器跨 Q tile 累加。
3. [x] `P=exp(scale*S-L)`；`D_i=sum(O_i*dO_i)` 用 FP32 预处理；`dS=scale*P*(dP-D)` 在转 HALF 之前施加 scale，与 cuDNN 官方 reference 保持一致。P 的 FP32 值保留到 dS 完成。
4. [x] 第五个 TensorOp 计算 `dQ=dS K`：dS 存入标准 RowMajor TensorOp congruous shared layout，以对应 ColumnMajor layout 读取转置；dQ 四 warp 分为 2 个 Q 行组 × 2 个输出通道组，每 warp 32 行，归约维不跨 warp。D32 的物理输出宽度补至64，D96补至128；逻辑维度保持不变。
5. [x] dQ 使用线性 FP32 workspace 跨 KV CTA 原子累加，再转换 HALF；每次运行在同一 stream 清零。dK/dV 每 CTA 独占输出。所有五个矩阵乘法仍为 CUTLASS TensorOp；D/LSE、概率重建及梯度逐点计算明确属于 attention 自定义逻辑。
6. [x] kernel 只组装 CTA 坐标、初始 iterators、循环次数、Mma/epilogue；threadblock 拥有全部搬运、iterator 推进、shared memory 生命周期及同步。标准 ThreadMap、cp.async、MmaCore/warp iterator、accumulator fragment iterator 沿用 example 13 的组合方式。设备层检查 opt-in shared memory 容量并明确拒绝不支持的配置。
7. [ ] 测试生成相同 Q/K/V，独立 dO 用 seed+4、同样的 `[-2,2]` 分布。候选和 cuDNN 分别产生自己的 O/LSE，再做 end-to-end backward；逐项要求 dQ/dK/dV MAE<1e-6，不放宽门限。cuDNN 不支持 Sq=Sk=1，该输入单独保留严格解析恒等式检查 `dQ=dK=0,dV=dO`。
8. [ ] 覆盖继承的 shape/seed/scale、边界和 D32/64/96/128，NaN 初始化梯度、poison workspace、dO=0 的同内存/非默认 stream 复用以及非法 workspace 拒绝。严格验证失败即停止该候选 benchmark，记录父子关系和证据，再设计下一候选。
9. [ ] 完成 build→verify 后重新跑 forward 正式脚本，确保新增可选 LSE 不破坏已有58项严格 forward parity 和四项性能门限。仅在 backward parity 通过后记录 backward 自身延迟。

共享内存转置审查：canonical ColumnMajor A 的 half2 连续加载不适合此转置；使用 stock TensorOp congruous layout。其 A iterator 的 warp M16 偏移不能当作已支持，dQ 必须用 M32 和 2×2 的 warp 分工。此约束来自 `mma_tensor_op_tile_iterator.h` 的 tile-offset 逻辑与 layout 地址核对。

### B02（parent=B01）：标准数学函数生成 forward 训练状态

B01共同状态诊断表明：10个失败用例中，9个在仅用于诊断的共同O/LSE下达到梯度门限，首个失败不受状态替换影响。正式路径仍必须生成自己的O/LSE。下一候选只将 forward 的两个 `__expf` 和保存LSE的 `__logf` 替换为 CUTLASS `fast_exp` / `fast_log`（对应 `expf` / `logf`）；同一份公共O继续用于backward，不增加替代O，不改变输入、reference、递推、P的HALF转换或tile顺序。

依据：`cutlass/fast_math.h:877,904`、example41 `kernel_forward.h:1139–1144` 的LSE保存，以及 `epilogue_thread_apply_logsumexp.h:65` 的标准expf。假设是 intrinsic 近似在HALF舍入边界附近造成状态差异，需用测量检验，不能先宣称修复。

步骤：保存B01失败源码和二进制（已完成）→仅修改上述forward数学函数→build→严格forward/backward verify。失败则停止该候选计时，允许只做各输入的数值诊断；若9项仍失败，记录结果后再决定下一候选。B01首败的两个dS临界舍入项另行诊断，不同时修改Delta以免混合归因。

### B03（parent=B01）：Delta 的分组 FP32 归约

B02的标准exp/log仅修复旧失败中的1项，scale4误差完全不变，另有2个边界用例误差增大；该假设不足，保存其源码/二进制后回到B01 forward算术。B03只改变Delta预处理，不改五次TensorOp及其舍入位置。

首败的全部梯度差异定位到head1的两个dS临界点 `(Q23,K11)`、`(Q57,K8)`。残差与Q/K的符号共同要求这两项dS更正，即Delta更小。CPU逐步FP32模拟显示：当前串行Delta为-1.4178470373和4.0488119125；8个worker各累加连续8个元素、再按xor4/2/1归约，分别为-1.4178471565和4.0488109589，方向符合证据，仍须GPU验证。

采用本地FA2 `flash_bwd_preprocess_kernel.h:36–46`、`kernel_traits.h:292–315`、`utils.h:111–130` 的分工依据：每行8worker，每次每worker用CUTLASS `Array`/`arch::global_load` 读8个HALF，FP32局部和，descending shuffle归约。D32的无效向量置零，D96/128继续下一组64通道；CTA统一行块循环，尾行仍参加shuffle但不读写越界。device网格按每CTA16行计算。这里不声称cuDNN内部使用该归约树。

步骤：落实Delta单项变更→build→strict backward gate（不放宽门限）→记录首败是否修复及下一失败。所有输入通过后才重新完整forward gate与性能验收；任何剩余训练状态误差作为下一候选，禁止在B03里顺便混入forward tile调整。

### B04（parent=B03）：forward 用 base-2 online softmax

B03已修复第6项，正式backward gate向前通过11项，在 `(1,2,129,193,64)` 的训练状态误差上停止。下一候选保留B03 Delta树，只将forward递推按example41 `kernel_forward.h:1186–1191,1262` 改为base-2：score乘 `scale*log2(e)`，最大值保留相同单位，alpha/P均用exp2，最后按 `maximum/log2(e)+log(l)` 保存自然对数LSE（同文件1139–1144）。输出O仍为同一标准epilogue归一化结果，不建立第二份O。

CPU诊断依据：放大输入D64用例中，原始64宽tile的自然指数近似模型对cuDNN O有8个HALF差异，base-2模型有5个；该模型不能精确复现GPU QK累加，不是正确性证明。128宽tile反而有112个差异，因此不改变tile宽度。B04需重新build、forward严格gate、backward严格gate；失败不计时，记录每个旧失败用例变化，不能用CPU模型替代GPU验收。

### B05（parent=B04）：按 FA2 源码组织 online softmax 的缩放与归约

B04的forward gate通过，backward通过前22项，但单KV tile的D96及放大输入仍有训练状态差异。重新核对本地FA2 `softmax.h:65–94,135–183` 后，发现其FP32操作顺序与B04/example41形式仍有区别。下一候选以同一个FA2 softmax主循环为依据整体对齐以下相互关联的步骤：

- 正常正scale时最大值保留在raw QK单位，`exp2(fma(score,scale*log2(e),-max*(scale*log2(e))))` 生成P；alpha使用 `(old_max-new_max)*(scale*log2(e))`。
- 每lane保存行和的局部部分，每次先乘alpha，然后按accumulator列次序逐项加P；KV循环中不做行和shuffle。只在循环结束时按FA2 `Allreduce<4>` 的xor2→xor1归约一次，再归一化O。
- 自然对数LSE使用 `max*scale+__logf(l)`。保留API已有有限scale范围：负scale用反号raw score与绝对scale；零scale使用零score与单位exp比例，避免首轮 `-inf*0`。

这是attention自定义softmax算术，Mma/iterator/epilogue组成和五个backward TensorOp保持不变。除数值对齐外，延迟行和归约直接减少每个KV tile的非matmul shuffle，对应任务第一项目标。不得声称cuDNN内部正好执行这些指令；严格GPU gate决定是否接受。先保存B04源码/二进制，再只修改forward Mma，build→forward/backward verify；失败不计时。

### B06（parent=B05）：使用 FA2 的 unfused 指数乘法分支

B05在forward的D96、input-scale4用例MAE=1.13603e-6，未达到契约。FA2同一 `softmax.h:84–92` 提供 `UNFUSE_FMA` 分支，明确用 `__fmul_rn(score,scale_log2)-max_scaled` 避免指数参数的FMA融合。B06只落实该分支，保留B05 lane局部行和与最终一次归约、B03 Delta；不更换reference或放宽门限。

假设：FMA残差使最大score的指数参数不再精确为0，HALF P与FP32分母在放大输入下产生不同舍入。该分支有官方代码依据，但是否满足本任务仍需build→forward/backward严格验证；失败不计时。

### B07（parent=B06）：最终归一化的标准近似倒数

B06通过forward gate，但端到端backward仍有8/58项超限。PTX证据已确认当前最终归一化为 `rcp.rn` 后乘法；候选只在03的epilogue使用标准 `cutlass::reciprocal_approximate<float>`，保持accumulator/output iterator、HALF转换及全部主循环不变。CUTLASS定义在 `functional.h:313–323`，生成 `rcp.approx.f32`，不声称cuDNN必定采用该指令。

目的仅是检验最终倒数的浮点舍入是否解释少量O的HALF临界差异。有效row的分母为正且至少包含一个P=1，任务范围内不会遇到零或subnormal倒数。保存B06源码/二进制后再改03 epilogue，禁止修改02；build→forward/backward严格验证，失败不计时。原来的可复用标准epilogue仍作为03的基础类型；只覆盖attention归一化操作。

### B08（parent=B03）：保留自然指数，只延迟行和归约

B07的近似倒数未修复第12项，dQ仍为1.26442922525e-6；该候选不保留。下一候选从已修复Delta的B03分支继续，保留C03/B03的自然单位max、`score*=scale`、`__expf(score-max)` 与 `m+__logf(l)`，只引入FA2的lane局部行和组织：先乘alpha、逐P累加、KV结束后xor2→xor1归约一次。继续直接继承原标准归一化epilogue，不修改02。

此候选把“减少重复shuffle”和“改变指数单位/融合”分开，以验证前者本身对精度与操作数的影响。旧base-2候选在放大输入上的端到端梯度误差反而增大，不能因为论文实现采用base-2就假设一定更接近当前cuDNN。先保存B07源码/二进制，落实B03→B08单项组织变化，再build→forward/backward严格verify；不通过不计时。

## 参考

- [FlashAttention-2 原文](https://arxiv.org/pdf/2307.08691)：本地 `csrc/flash-attention/papers/flashattention-2-2307.08691.pdf`。
- [LeetCUDA split-Q](https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/flash-attn/mma/basic/flash_attn_mma_split_q.cu)：只参考算法/warp 分工，不能照搬其 FP16 累加或整除限制。
- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_base.h`
- `3rdparty/cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage.h`

## 2026-09-26 论文与当前源码复核

最初“已读现状”中的两条 02 判断有误，保留原文字作为候选计划的历史记录，在此更正：当前 02 kernel 的 grid=(batch,head)，一个 CTA 覆盖该 head；threadblock 的外层循环扫描 KV，内层遍历所有 Q tile；02 的 warp 已按 Q 行分工，并且每个 warp 覆盖完整 KV 列及输出通道，无跨 warp 的部分 O 合并。02 每次 KV 更新后归一化 O，并把中间 O/m/l 写入 FP32 workspace。03 才改为每个 Q tile 一个 CTA、外层 Q/内层 KV，且在最后归一化。依据是当前 02 kernel/default policy/threadblock 与当前 03 kernel/threadblock 的实际源码，已逐项写入博客正文。

用户本轮明确要求先按 FlashAttention-2 原文写博客，而不是从 Profile 开始。博客现已按原 Overview 顺序补全五节正文，论文 Algorithm 1/2、Figure 2/3 与项目源码分别核对；历史 C03 的 Profile 留在正文之后，B08 的梯度失败仍如实标注。之前等待 Overview 确认的状态已由本轮明确写作要求取代，不再将其作为正文编辑的障碍。
