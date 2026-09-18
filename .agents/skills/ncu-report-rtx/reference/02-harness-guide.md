# Reproducible profiling harness

Prefer an existing verified executable exposing the real launch. Framework/JIT
code does not automatically require isolation; dispatch, cache state or graph
dependencies may otherwise change.

## Build contract

Harness source belongs in `csrc/tests/<family>/`, all compiler output in
`build/<family>/`. Reuse the family build script. Configure supported
`sm_80`, `sm_86`, `sm_89` or `sm_120` targets, feature suffixes only
when required. Use optimized flags plus `-lineinfo`, not `-G`.

Example for an existing harness and detected SM89 device:

```powershell
$Target = 'sm_89'
New-Item -ItemType Directory -Force 'build/profile-harness' | Out-Null
nvcc -O3 -std=c++17 -lineinfo "-arch=$Target" csrc/tests/example/harness.cu -o build/profile-harness/harness.exe
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
```

On Windows use a compatible developer shell. Direct intermediates/`--keep`
to a build directory too. Architecture/dtype/layout/tile belong in policies,
not primary operator or public API names.

## Required behavior

- Include/call the actual launcher, validate sizes/strides/alignment and inputs.
- Print device properties, workload/seed, selected policy and launch geometry.
- Initialize input/workspace; preserve production work distribution.
- Check CUDA calls and immediate launch errors, then synchronize for async faults.
- Run `verify.py` against the trusted reference and agreed tolerance before
  `bench.py` or NCU. A successful launch is not parity.
- Warm up outside the capture; document matching launch skip or supported
  profiler/NVTX range. Include all required subkernels in operator timing.

The [template](../helpers/harness_template.cu) fails until the real allocation,
validation and launch are implemented. It is not a dummy benchmark.

## Workload and JIT paths

Use [list_flashinfer_workloads.py](../helpers/list_flashinfer_workloads.py)
with `--dataset`, `--definition`, `--filter`, `--uuid` or
`--unique-axes` to choose representative dispatch paths and length distributions.

The minimal [safetensors reader](../helpers/safetensors_loader.h) is host-side.
Check dtype, shape, byte count and destination capacity. Packed quantization
and scale conventions require a format-aware implementation; raw loading
does not implement decoding.

For JIT/framework builds, record generated code/cache identity and enable line
information through supported compiler options. Preserve SASS-level findings
when CUDA source correlation is unavailable; do not assume a PTX rebuild
reproduces the original dispatch.
