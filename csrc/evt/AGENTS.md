# evt 工作区 Agent 指南

这个工作区是**纯学习/设计工作区**：读 EVT（Epilogue Visitor Tree）原论文和 CUTLASS
`include/` 里的实现，产出技术博客和融合设计。它当前**不含 kernel 源码、不含 target、
不进构建**。

## 目录

```text
papers/    原论文 PDF 与 pdftotext 抽取的纯文本（引用页码时以 PDF 为准）
blogs/     技术博客（走 cutlass-blog-workflow skill）
```

## 规则

- 全局规则见仓库根 `AGENTS.md`，这里不覆盖。
- 博客必须走 `cutlass-blog-workflow`：`前言 -> Overview -> 正文 -> Profile -> 后记`；
  `前言` 只按用户 prompt 写，`后记` 由用户本人写，`Profile` 只写通过 reference parity
  的 NCU 实测。
- 引用源码必须带路径和行号，且指向 `3rdparty/cutlass` 当前 checkout 的真实符号；不确定
  的地方写成公开问题，不靠推断补。
- 本工作区**不新增 kernel**。EVT 的可运行实例在
  `csrc/swin/kernel/evt_attention_scores.h`（target `swin_evt_attention_scores`），
  博客引用它，不在这里复制一份。
- 论文译文标注为译文，不与源码结论混写；论文描述的编译器（graph pass、ILP partitioner）
  属于上游 EVT_AE 项目，**不是** CUTLASS `include/` 里的内容，必须区分。
