# Naming and template discipline

One rule underneath all of these: **dtype, architecture, and layout are template
parameters, not name fragments.** They are selected at the launcher, explicit
instantiation, test, or CMake layer — never baked into a primary name.

## DO

- Express kernel policies as template type factories, normally
  `DefaultXxx<ArchTag, Element..., ThreadblockShape..., WarpShape..., EpilogueOp...>`.
- Pass CUTLASS shape types (`cutlass::gemm::GemmShape<...>`) directly as template
  parameters when the shape choice is local and simple.
- Use CUTLASS-provided arch tags, storage/layout types, swizzles, and tensor
  reference helpers directly. Do not wrap them in local one-off functions or long
  concrete aliases.
- Keep concrete policy choices in launcher code, explicit instantiation units,
  CMake configuration, or local tests.
- Keep shape defaults in the `DefaultXxx` factory's default template arguments
  until a separate policy layer earns its keep.
- Introduce traits only when they remove real duplication across multiple
  architecture or dtype specializations.
- Report unsupported architectures or problem shapes through CMake checks,
  asserts, or an explicit `kErrorNotSupported` path in `can_implement`.

## DO NOT

- No primary class, struct, function, target, or public API name with the
  architecture written in: `SomethingSm80`, `SomethingSm89`, `SomethingSM90`.
- No name with the dtype or layout written in: `SomethingFp16`, `SomethingBF16`,
  `SomethingNHWC`, `SomethingRowMajor`.
- No non-template specialization alias:
  `using SomethingFp16 = DefaultSomething<cutlass::arch::Sm80, cutlass::half_t>`.
- No old-name compatibility shim: `using OldSomething = NewTemplatedSomething<...>`.
  Rename the call sites instead.
- No one-off trait that merely wraps a single `GemmShape`.
- No threadblock/warp shape file that exposes only one concrete architecture
  struct.
- No `Impl_`-style dual-entry template parameter used to keep an older
  implementation alive next to a new one. A superseded implementation gets
  deleted, or becomes its own numbered variant with its own test.
- No `_test` suffix on test files or CMake targets under `csrc/tests/<family>/` —
  the directory already says test.

## The one legitimate exception

A name may carry a dtype or arch fragment when it marks a **real, un-templatable
family boundary** — a genuinely different algorithm, not a different instantiation
of the same one. An FP8 path that needs its own scaling design and its own
verification is a family; an fp16 instantiation of an existing template is not.
When in doubt it is not an exception.
