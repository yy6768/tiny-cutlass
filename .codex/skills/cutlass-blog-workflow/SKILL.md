---
name: cutlass-blog-workflow
description: Run the human-led tiny-cutlass CUDA/CUTLASS learning-blog workflow. Use when creating or continuing a CUTLASS kernel blog, initializing an article overview with agents, turning an approved overview into source-backed learning sections, or profiling a verified kernel with Nsight Compute before the human writes the closing notes.
---

# CUTLASS Blog Workflow

Use `tiny-cutlass-blog-style` for voice and `cutlass-kernel` for kernel, verification, benchmark,
and profiling rules. Keep the human in control of the article's intent and conclusions.

## Fixed Article Shape

Use [assets/blog-template.md](assets/blog-template.md). Preserve this order:

```text
前言 -> Overview -> overview-derived sections -> Profile -> 后记
```

Do not replace `Overview` with `实现`. The overview is the agreed reading map; the body follows
that map.

## Ownership

- **Human owns `前言`**: write version one only from the user's prompt. Preserve uncertainty,
  motivation, and personal wording. Do not manufacture a polished motivation.
- **Agents initialize `Overview`**: inspect the named example, local implementation, and direct
  CUTLASS dependencies. Produce a compact source-reading map, not the complete article.
- **Human and Codex learn through the body**: derive body headings from the approved overview.
  Explain one heading at a time from real source symbols. Do not fill unknown sections by
  extrapolation.
- **Codex owns measured `Profile` evidence**: profile only the existing, verified source path and
  preserve NCU artifacts.
- **Human owns `后记`**: leave a placeholder unless the user supplies text. Codex may organize or
  lightly edit supplied text but must not invent the conclusion.

## Workflow

### 1. Draft The Preface

Turn the user's prompt into `## 前言`. Limit the first version to what the user actually said:

- why this kernel or CUTLASS path is being studied;
- what previous experiment motivates it;
- what is currently unclear;
- which source/example is the starting point.

Mark missing facts as open questions or omit them. Stop after the preface if that is all the user
requested.

### 2. Initialize The Overview With Agents

When the user asks to initialize the overview, use parallel agents only for independent read-only
source tracing. Give each agent one bounded path, for example:

- public/example configuration and launch path;
- `DefaultXxx` policy/factory expansion;
- kernel/threadblock main loop and data residency;
- existing verify/bench/profile entrypoints.

The primary agent must reconcile their evidence. Write an overview containing:

1. the operator dataflow;
2. the source call path;
3. the concepts that need separate body sections;
4. explicit boundaries for what this article will not explain yet.

Keep it short enough to review before drafting the body. Do not let agents write independent prose
directly into the article.

### 3. Expand One Overview Item At A Time

After the overview is accepted, create body headings in the same order. For each heading:

1. read the exact local source and its direct dependency;
2. state the question this section answers;
3. map math or dataflow to real types, iterators, fragments, and calls;
4. distinguish verified behavior from interpretation;
5. record remaining uncertainty instead of hiding it.

Prefer a small code excerpt plus explanation over a large source dump. Never write later sections
as if they are understood merely because their names appear in the overview. If the user is working
through the code interactively, stop at the current section and wait for the next request.

### 4. Gate Profiling On Correctness

Before NCU, follow the repository workflow:

```text
build -> verify -> bench -> profile
```

- Keep build and profile artifacts under `build/`.
- Use the family script under `scripts/kernels/<family>/` when available.
- Do not benchmark or profile when reference parity fails.
- Profile the existing verified executable and representative problem; do not create a substitute
  microkernel unless the user explicitly changes scope.
- Capture both `.ncu-rep` and CSV. Include the exact build, verify, benchmark, and NCU commands.

Write `## Profile` only from measured artifacts. At minimum record:

- GPU, architecture, dtype, layout, and problem shape;
- executable, kernel name, launch configuration, and NCU command;
- elapsed time or benchmark context;
- relevant memory, Tensor Core, occupancy, stall, and pipeline metrics;
- what the source predicts, what NCU confirms, and what remains a hypothesis.

Do not call a kernel optimized merely because NCU was collected.

### 5. Reserve The Closing Note

Keep this in the draft until the user writes it:

```markdown
## 后记

<!-- HUMAN: profiling 后补充自己的结论、踩坑和下一步。 -->
```

When the user supplies closing notes, preserve first-person judgment and technical uncertainty.
Only correct structure, clarity, and factual conflicts supported by source or profiling evidence.

## State Discipline

- A new article may contain only `前言`, an initialized `Overview`, empty body placeholders,
  an empty `Profile`, and the human-owned `后记` marker.
- Do not complete all stages in one pass unless the user explicitly asks for the entire workflow
  and the kernel is already verified and profileable.
- Treat overview approval, reference parity, and measured NCU artifacts as three separate gates.
- Never backfill measured claims from comments, expected behavior, or a different kernel variant.
