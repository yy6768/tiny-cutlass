# SM89 padded window attention：KDA 任务契约与后续计划

2026-09-15。当前交付阶段是 KDA 本地维护、实现规范、静态门禁与现状审查；
完整 FP8 kernel 仍未实现。源码依据及结构问题见 [draft](draft.md)。

## 固定边界

- 当前主线：SM89 CUTLASS 2.x，矩阵乘法复用库的 TensorOp/MMA/iterator；
  CUDA 用于 norm、softmax、索引和必要搬运。其他架构待用户指定，不扩展 CMake。
- 接口保持现有模板化 raw pointer + descriptor + stream；无 Torch ownership、
  SIMT/raw GEMM fallback、旧名 alias 或 dtype/arch 写入公共名字的入口。
- 沿用现有 window4/reflect/RMSNorm/QK mode/预展开 bias/crop 语义；
  complete attention 与 complete block 分开验收和计时。
- 用 [源码门禁配置](../../kernel-policy.json) 和 [实现规范](../../../../.agents/skills/kernel-design-agents/references/implementation-policy.md)
  执行候选前预检与采用前结构审查；登记底层例外不代表支持或性能已验证。

## FP8 实施前必填契约

下列项目本轮保持 `待确认`。它们是后续实现的输入，不妨碍完成本轮 KDA/门禁交付。
未齐备前继续源码调研、整理候选依据，不启用 `quant_fp8`，不臆造默认量化方案。

| 必填项 | 需要明确的内容 | 当前状态 |
|---|---|---|
| reference / 数据 | 用户量化参考的确切版本、代表性真实输入及权重 | 待确认；仓库仅有非量化 attention reference |
| 格式与位置 | QKV/QK/PV/proj 各阶段的输入、权重、输出类型，以及哪些张量量化 | 待确认；不默认全流程 FP8 |
| scale | 编码/解码方向，per-tensor/group/row 粒度，动态或静态、更新生命周期 | 待确认；grouped GEMM 公共 alpha 不代表 attention 契约 |
| 数值规则 | round、saturate、零 scale/异常输入、累加类型及中间舍入点 | 待确认 |
| PV | P/V 的最终类型及是否存在额外 quantizer | 待确认；不得因为 QK 用 FP8 就连带量化 PV |
| 误差和质量 | 相对量化 reference 的算子误差；相对原模型的质量要求与数据 | 待确认；两个层次分开记录 |
| 主 workload | attention 的 B/H/W/C/G/Dq/Dv、shared/separate、shift、出现频率 | 待确认；已有 720p 是 block 用例 |
| 性能目标 | baseline、稳态/冷启动、Graph/eager、量化/准备成本、目标指标及可接受回归 | 待确认；不预设加速倍数 |

## 后续阶段与验收条件

1. **补齐量化契约。** 依据用户 reference 固定上表，并列出支持矩阵及显式 unsupported 行为。
   为每个阶段选择可复用的固定版本 CUTLASS 组件，先记录库能力缺口。
2. **保持 FP16 对拍的结构整理。** 统一 shared-memory 大小/对齐/生命周期契约，
   拆清输入准备、MMA/operand、softmax、输出职责和 block 复用边界；一次候选回答一个问题。
   不更改 FP16 reference 或容差，以完整 attention 和 block parity 为准。
3. **接入 FP8 primitive。** 以阶段模板/policy 表达 operand、accumulator、MMA instruction、
   layout 和 scaling；stock grouped GEMM 只作为 primitive 证据，不等同单 kernel attention fusion。
   如库能力不足，先给出隔离 primitive 的 decision/验证方案，不在业务层插入临时 PTX。
4. **完整验证。** 根据独立量化 reference 验证所有契约内形状和逐阶段数值；覆盖边界反射、
   单像素轴、两轴 shift、shared/separate QK、不同 Dq/Dv、bias 顺序、量化边界和非有限值。
   回归 FP16、完整 block、CUDA Graph、非默认 stream、对齐及 unsupported；存储/同步变更再跑 sanitizer。
5. **目标设备实测优化。** parity 通过后测同输入 baseline/candidate，计入量化、scale 更新、
   转换及必要 workspace 的成本，单次准备与每次调用成本分别标明。再用 NCU 解释 register/shared
   memory、spill、访存与同步开销；普通无 profiler 测量决定收益，噪声内变化不宣称加速。

后续候选必须逐项通过源码门禁、结构审查和 reference。不能调整容差使候选通过，也不能用
局部 grouped GEMM 或小图成绩代替目标算子。必要例外在 candidate 阶段可为 `not-run`，
采用前必须有库缺口、接口、parity 和实测收益/维护成本证据。

## 当前命令与证据口径

从仓库根执行，三个正式入口均在 build 前检查源码，随后 build → verify → bench；
任一门禁或数值失败会退出，不进入后续 benchmark/profile。

```text
python -S -B csrc/tests/swin/verify.py --source-only
python -S -B -m unittest discover -s csrc/tests/swin -p test_source_policy.py -v
scripts\kernels\swin\swin.bat
scripts\kernels\swin\swin.bat block
scripts\kernels\swin\swin.bat grouped-gemm
```

当前 FP16 attention 判据 MAE≤1e-3、max_abs≤2e-2、非有限立即失败；这只是既有路径回归条件。
FP8 attention 的判据必须来自上表，不继承这个数字。直接 verify/bench 也会检查源码，
门禁不依赖 Torch/CUDA；原有数值验证和 benchmark 使用现有 GPU harness。

每次优化记录 candidate/parent、假设、源码 revision 和未提交 patch、policy/例外/审查结论、
设备/工具版本、输入标识、验证命令/误差、计时范围/样本/波动、采用或淘汰理由。
新实验输出放 `build/experiments/swin/<run-id>/`，不覆盖旧证据；文档保留结论及路径，
标明忽略目录中的原始产物未随 Git 交付。缺失数值记 `null`/`not-run`。

## 本轮实施记录

- 上游：NVlabs/KDA `7d9e94a636f9149b4ffe84d4ca0e721e77ee140e`，20 个普通文件，
  两个 gitlink 未纳入；manifest/来源信息位于仓库技能目录。
- 源码预检：当前 16 个本地源码文件通过；规则没有发现违规不代表完整结构审查合格。
- CPU 回归：10 项测试通过，覆盖规则、注释/字符串、第三方/产物排除、例外、独立入口及
  default/block/grouped-gemm 路由。Windows 路由测试使用真实 cmd 控制流程与 CPU 替身，
  在测试副本中给 CMake `.bat` 替身加 CALL，不冒充 GPU build/benchmark。
- 数值对拍：重新构建 `swin_window_attention` 后，在 RTX 4070 Laptop（SM89）、PyTorch
  2.7.1+cu126 上完成 66/66 用例及 Graph replay。逐用例 MAE 最大为 `1.00511e-5`，
  max_abs 最大为 `4.8828125e-4`，均通过既有容差。原始结果为
  `build/kda-local-maintenance/parity/reports/window_attention/verify.json`，完整日志为
  `build/kda-local-maintenance/attention-parity.log`。本轮未测 FP8 attention，也未做 benchmark/NCU。
- 完整性：20 个上游文件的 blob ID/SHA-256 与固定版本一致；技能结构校验通过。
  attention 目录 17 个文件（16 个源码及 README）的 SHA-256 与实施前一致。
- 本轮日志与源码校验保存在 `build/kda-local-maintenance/`，属于本地产物，不随 Git 交付。
