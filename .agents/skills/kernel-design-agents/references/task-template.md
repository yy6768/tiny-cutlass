# 端侧 Diffusion 算子任务模板

下面是供用户/agent 填写的任务模板，字段尚未填写表示未知，不是安装遗漏。
用户只给部分信息时，先从仓库补全能确认的项；影响实现正确性的缺项再询问。

```text
使用 $kernel-design-agents 完成以下任务。

模型与版本：
任务范围：规划 / 实现 / 验证 / benchmark / NCU 诊断
目标设备：SM89 / SM90 / SM120，具体 GPU 和运行环境
目标算子 / 子图：
输入来源与 shape、stride、layout：
精度、累加类型、mask/activation/epsilon 等语义：
可信 reference 与版本：
数值容差及模型质量要求：
性能目标、显存/功耗约束：
冷启动 / 稳态、CUDA Graph 模式：
现有源码 / harness / 数据路径：
实际 build / verify / bench 命令（从仓库确认后填写）：
采样器、步数、CFG、prompts/seeds（涉及模型验收时）：

先读取仓库约束与目标架构页，确认当前支持状态。
为实现任务形成 draft 和可执行 plan，再迭代候选。
严格 build → verify → bench；通过 parity 后再 profile。
记录候选关系、环境、数值结果、性能数据和采用/淘汰原因。
三架构逐一标注设计、编译、运行、parity 和性能状态。
没有目标设备时完成可做的设计/编译工作，并列出实机待验证项。
```

一个不依赖尚未提供模型数据的起步请求：

> 使用 $kernel-design-agents，检查当前 SM89 仓库中可复用的 attention、MLP、norm 和
> conv family，为后续端侧 Diffusion trace 制定输入采集与 reference 验证计划，
> 同时列出 SM90、SM120 的迁移检查点。

一个已有输入后的实现请求应补充：实际 family、shape 清单、数据路径、reference、
容差和目标设备。不要把模板里的架构枚举理解成每个任务都必须同时运行三张卡。
