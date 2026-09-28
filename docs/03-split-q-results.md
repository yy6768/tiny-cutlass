# 03-split-q 候选记录

任务与验收规则见 [draft.md](draft.md)。每个候选只在严格 reference parity 后进入 benchmark/profile。

| ID | Parent | 变化 | 状态 | 证据 |
|---|---|---|---|---|
| B00 | 当前工作区 02 | 固定对照实现 | 重新构建，诊断 MAE=5.20706e-6 | `build/split-q-study/baseline/` |
| C01 | B00 | 4 warp 沿 Q 行分工，保留 Q，官方 fragment iterator 传 P | build 通过；严格 verify 第 3 项失败，未 bench/profile | `build/split-q-study/c01/` |
| C02 | C01 | 只改变 KV 遍历：完整 tile 在前，尾块在后 | 严格 verify 通过，性能 1/4 通过 | `build/split-q-study/c02/` |
| C03 | C02 | V 使用 DefaultMmaCore 的 swizzled shared layout 与 vector cp.async | 严格 verify 通过，性能 4/4 通过 | `build/split-q-study/c03/` |
| B01 | C03 | 可选 forward LSE、FA2 Algorithm 2 backward；CTA 沿 KV 分块，五次 TensorOp、FP32 dQ 原子累加 | build通过；严格backward第6项dK失败；未计时 | `build/split-q-study/b01/` |
| B02 | B01 | forward 的 exp/log 改用 CUTLASS 标准数学函数，检验训练状态的舍入误差 | forward通过，backward仍失败；假设不足，未计时 | `build/split-q-study/b02/` |
| B03 | B01 | Delta改为8worker×8元素的FP32分组归约 | 修复旧第6项，前11项通过，第12项失败；未计时 | `build/split-q-study/b03/` |
| B04 | B03 | forward递推改为example41的base-2形式 | forward通过，backward前22项通过，第23项失败；未计时 | `build/split-q-study/b04/` |
| B05 | B04 | FA2 raw最大值/FMA指数、lane局部行和、循环结束一次归约 | forward第36项失败；未运行正式backward/计时 | `build/split-q-study/b05/` |
| B06 | B05 | 按FA2的UNFUSE_FMA分支分开指数乘法与减法 | forward通过，端到端backward仍有8项超限；未计时 | `build/split-q-study/b06/` |
| B07 | B06 | 03 epilogue使用标准近似倒数 | forward通过，backward仍第12项失败；未计时 | `build/split-q-study/b07/` |
| B08 | B03 | 自然指数不变，仅延迟lane行和归约 | forward通过，端到端backward仍有9项超限；未计时 | `build/split-q-study/b08/` |

现有旧二进制的先行诊断：shape=(1,1,256,256,64)，02 对 cuDNN MAE=5.20706e-6，超过本任务 1e-6。仅作为数值风险提示，不能据此宣称新候选可达标。

## C01

源码快照 `build/split-q-study/c01/source/`；构建日志 `build-fixed.log`；精度 `verification.json` / `verify.log`。首两项精度为 0；第三项 (2,3,63,65,32)、seed=3082、input-scale=1，MAE=2.59884e-6，max_abs=0.000488281。该项跨越 Bc64，CUTLASS 原始 iterator 先处理 residue1 再处理完整64；FP16 P 的舍入可能随分块次序变化。

C02 假设：让相同 KV 坐标按 [0,64)、[64,65) 顺序处理能减少与 cuDNN 的差异。使用完整逻辑 extent 构造标准 iterator，tail 的合法行数由 Mma 显式 predication；不改变输入、reference 或误差门限。假设须由验证决定。

## C02

66 项 verify + 2 项 unsupported rejection 均通过。其中 58 项 03 严格检查的最大 MAE=4.01022e-7（shape=(1,2,129,63,96)，seed=3102）；复用检查亦通过。binary SHA256=`fd2007e6f96110d9517bce801978836b3d3fc7cddd4634c4112b3a4fc104d473`。

同 shape、seed=3080、input-scale=1、每样本50次、5次交错采样，CUDA event median：

| (B,H,Sq,Sk,D) | 02 ms | 03 ms | t03/t02 |
|---|---:|---:|---:|
| (1,1,256,256,64) | 0.027546 | 0.026643 | 0.967218 |
| (1,4,1024,1024,64) | 0.145940 | 0.179773 | 1.231828 |
| (1,4,1024,1024,128) | 0.703428 | 0.413286 | 0.587531 |
| (16,12,1024,1024,64) | 5.131350 | 6.038040 | 1.176696 |

性能契约未通过，不能宣称优化完成。正在采集 C02 NCU；下一候选须由瓶颈证据支持。精度结果证明本次输入集合上的 parity，不证明 cuDNN backend 使用某个固定 tile 或舍入次序。

NCU 在 (1,4,1024,1024,64) 上实测：03 的 L1/TEX=90.62%（02=78.35%），MIO throttle 约5.8 cycles/issued instruction（02≈2.8），eligible warps≈0.15（02≈0.22）；03 registers=168，反而低于02的255。C03 针对 V 的逐元素 global load/shared 转置改用 `DefaultMmaCore` 的标准128-bit搬运及 TensorOp swizzle。D96 的物理 V 通道补齐128，逻辑输入/输出保持96且有边界 predication。其余数值递推、KV顺序不变。

## C03

66 项 verify + 2 项 unsupported rejection 通过；58 项03严格检查最大 MAE=4.01022e-7。对应 `c03/verification.json`、`c03/benchmark.json`、`c03/benchmark.csv` 和源码快照。

| (B,H,Sq,Sk,D) | 02 ms | 03 ms | t03/t02 |
|---|---:|---:|---:|
| (1,1,256,256,64) | 0.028549 | 0.016503 | 0.578057 |
| (1,4,1024,1024,64) | 0.145940 | 0.068605 | 0.470093 |
| (1,4,1024,1024,128) | 0.703611 | 0.150385 | 0.213733 |
| (16,12,1024,1024,64) | 5.158970 | 2.301950 | 0.446203 |

这是既有共享 harness 的 CUDA-event launch 序列计时。每次调用含 device 查询与初始化，因此 GPU 等待 host dispatch 的间隙也可能计入；不能直接等同于 NCU 的单 kernel duration。02/03使用相同调用路径、输入、seed、计时方法。此次四项均满足严格 `<0.95`。

最终检查对注释进行了澄清，并在 device `can_implement` 中补充 padded KV extent 的整数溢出拒绝；随后运行 `scripts/kernels/attention/attention.bat` 完整入口，证据写入 `build/split-q-study/validation/`。最终 NCU 与该最终二进制绑定。

正式入口 exit code=0，四项比值为0.587411/0.468053/0.214654/0.434950；最大严格MAE不变。最终二进制SHA256=`4a4d82e8e1730a1930d4113def814e8a781bc529f1676d9959958506e9f79480`。02全部9个源码文件与初始快照一致；验证、计时与源码manifest位于 `validation/`。

最终NCU在同一个D64 shape实测duration86.880us、Tensor elapsed activity21.310961%、MIO throttle1.179796；ordinary shared-store及cp.async/LDGSTS bank conflicts均为0，shared-load conflicts仍有2,293,760。完整[报告](../build/split-q-study/validation/profile/REPORT.md)区分早期02/C02和最终03采集，以及不同采集之间实际时钟的小幅差异。正式性能结论以同二进制的未插桩benchmark为准。

## B01

扩展原始任务要求的 backward。改代码前的设计与可执行步骤见 `draft.md` 的 Backward 计划。C03 正式验证二进制另存 `build/split-q-study/validation/c03-bin/flash_attention_test.exe`；上面的 C03 测量不代表新增 backward，也不自动覆盖后续修改。

验收保持每个梯度 MAE<1e-6。测试分别运行候选和 cuDNN 的 training forward，使用各自的 O/LSE 做 backward。cuDNN 本地 frontend 对 Sq=Sk=1 显式不支持，此形状必须单独标为解析恒等式检查。02 没有 backward 接口，因此本任务的 `<0.95×02` 只对可比较的 forward 测量；backward 达标后单独记录自身耗时。

可选LSE接入后，forward的66项验证与2项拒绝检查通过，58项03最大MAE仍为4.01022e-7。B01全路径构建通过，但backward第6项 `(1,2,129,17,64)`、seed=3085 的 dK MAE=1.68295467601e-6；dQ=1.25215959179e-7、dV=0。正式gate停止，未运行benchmark/profile。失败二进制与源码快照保存于 `b01/`。

随后只做故障诊断，逐项运行剩余输入，不计时、不放宽阈值：58项中48项达到门限，10项失败（正式gate的第6项加诊断sweep中的9项）。scale4在所有D上都存在超限梯度，普通scale1也有小序列/边界失败。具体每梯度指标见 `b01/diagnostic-sweep.json`；不能把部分通过写成B01通过。下一步用共同O/LSE的显式诊断模式归因，正式验收仍保留两边独立的training状态。

共同状态诊断结果：除首败用例外，其余9项在共同O/LSE下的各梯度均小于1e-6；其中放大输入的dV在共同LSE下误差为0，dQ/dK主要受O影响。首败的所有梯度误差不变，且其O差异发生在head0、梯度差异发生在head1，排除了O作为该例原因。详见 `b01/common-state.json` 与CPU dump分析；共同状态只用于归因，不是验收结果。

## B02

假设及修改前计划见 `draft.md`。只改forward标准exp/log调用，保留B01 backward算术与独立training-state验收；另包含已核对的device initialize/run职责整理和显式诊断CLI，这两项没有改变算子数学。

build和forward66项+2拒绝检查通过，forward严格最大MAE=3.38371e-7。backward仍在第6项失败，dK误差不变。旧10个失败中的9项仍未达到门限，scale4所有梯度误差完全不变，两个65×65边界用例还出现更大的dQ/dK误差。保存 `b02/source/`、二进制、正式验证和 `diagnostic.json`，不计时；下一候选从B01算术分支继续。

## B03

单项Delta归约候选的源码依据、两个临界点和精确FP32模拟结果见 `draft.md` 及 `b01/failing-case-delta-direction.json`。正式验收仍使用独立O/LSE。

B03构建通过，旧首败 `(1,2,129,17,64)` 已通过；正式gate共通过前11项。第12项 `(1,2,129,193,64)`、seed3091的dQ MAE=1.52248499352e-6、dK=1.01278193411e-6，仍未达标；dV=4.7502276811e-8。该例在B01共同状态诊断中已定位为主要受O/LSE差异影响。保存 `b03/source/`、二进制、日志及manifest，无benchmark/profile。

## B04

修改前计划及CPU模型的适用边界见 `draft.md`。保持64宽KV tile和B03 Delta，按CUTLASS example41显式base-2递推；正式验收仍是GPU端两边独立training状态。

B04构建及forward66项+2拒绝检查通过。backward第23项 `(1,2,129,63,96)` 的dK MAE=2.33743241224e-6，停止正式gate。旧10个失败中的5项现在达到门限，但scale4 D96/128误差有增大；仍未计时。完整旧失败集的诊断指标见 `b04/diagnostic.json`，源码与二进制已保存。

## B05

修改前计划见 `draft.md`；以本地FA2 softmax源码的完整操作顺序为一个候选，不拼接未经验证的cuDNN内部假设。

B05构建通过；forward在 `(1,2,65,129,96)`、input-scale4、seed941的MAE=1.13603e-6处失败（第36项）。正式backward、benchmark/profile均未运行。完整源码、两个可执行文件、manifest及日志已保存于 `b05/`。

## B06

仅改指数参数的FMA融合，依据和修改前计划见 `draft.md`；继续保留减少非matmul归约的主循环组织。

B06构建和forward66项+2拒绝检查通过。backward前11项通过，第12项dQ MAE=1.270320382e-6；其余输入的只读数值诊断共发现8项超限（含该项），已记录 `b06/diagnostic-sweep.json`。不计时，不把成功的单项结果视作完整parity。源码、二进制与manifest已保存。

## B07

最终倒数舍入的单项候选；实现前计划与CUTLASS原始组件位置见 `draft.md`。

B07构建与forward66项+2拒绝检查通过，backward仍在 `(1,2,129,193,64)`、seed3091失败：dQ MAE=1.26442922525e-6。与B06相比只改变极少数O，未解决主要误差；保存源码、二进制后不保留该epilogue覆盖。未计时。

随后将全部58项输入都运行在显式 `--diagnostic-state=reference-both` 模式，只用于隔离 backward 算术与上游保存状态。相同O/LSE下全部梯度达到数值门限，最大dQ MAE=2.56188930227e-7、dK=1.16880983114e-7、dV=0；证据见 `b07/common-input.json`。这个诊断不执行复用验收、不生成parity stamp、不计时，不能替代两边分别生成O/LSE的端到端gate。结果支持继续排查保存状态的舍入，而不是无证据修改五个backward TensorOp。

## B08

回到B03的自然指数算术，只采用FA2延迟行和归约，计划在修改前写入 `draft.md`。

构建、forward66项+2拒绝检查通过，58项03最大MAE=4.01022e-7。正式backward在第12项 `(1,2,129,193,64)`、seed3091停止：dQ MAE=1.2923833416e-6、dK=8.06522936864e-7、dV=3.02279319788e-7。其余输入的数值诊断共有9项失败，见 `b08/diagnostic-sweep.json`；保存源码、二进制与manifest，不计时。延迟行和归约减少循环内shuffle，但不能据此宣称端到端精度或最终性能验收通过。

额外有限scale smoke：负scale的singleton通过，包括同workspace/非默认stream的零dO复用；三个普通shape的零/负scale检查遇到非有限reference。零scale `(1,2,63,65,64)` 的dump确认候选dQ/dK/dV均有限，而cuDNN 9.21的全部reference梯度为NaN。该现象属于当前reference在此配置的限制，不能将额外检查写作通过，也不影响原始58项使用正attention scale的验收结论。见 `b08/scale-smoke.log` 和 `b08/zero-scale-*.bin`。

内存与同步补充检查：同一B08二进制在 `(1,3,65,257,96)`、seed3086、input-scale1上运行Compute Sanitizer 2025.2的memcheck、synccheck和racecheck，均exit0、0错误/0 hazard；该单项端到端梯度与零dO复用也通过。过滤 `kns=FlashAttn` 覆盖候选forward、Delta、主backward及dQ转换，输入包含Q/KV尾块、D96补至128、跨KV CTA的dQ累加和shared union复用。命令与二进制SHA见 `b08/sanitize.py`、`b08/sanitizer.json`，日志为 `memcheck.log`、`synccheck.log`、`racecheck.log`。此检查支持所测输入的访存/同步正确性，不替代58项严格数值gate；dQ的FP32原子加法仍允许不同CTA累加顺序造成末位变化。
