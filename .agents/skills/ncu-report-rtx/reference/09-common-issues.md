# NCU troubleshooting

| Symptom | Check/action |
|---|---|
| Unsupported target/device | Actual CC, NCU release support, nvcc --list-gpu-code; fail explicitly, no architecture fallback |
| ERR_NVGPUCTRPERM | Driver counter policy; Windows administrator controls NVIDIA Control Panel performance-counter access; Linux uses authorized administrator workflow |
| No matching launches | Device visibility, name mode/regex, process, skip count, real dispatch |
| Missing source | Optimized -lineinfo binary, matching revision and accessible source; import-source cannot create line tables |
| No PM data | Installed sections/names, driver/platform, duration/sample count and errors; keep overview and mark limitation |
| Missing metric | Report names versus device query; preserve N/A |
| Ambiguous report | --list-actions, explicit range/action or analyzer --all-actions |
| API/DLL failure | Matching NCU extras/python, Python ABI/bitness, NCU_PYTHON_PATH; use CLI CSV/details fallback |
| Slow full profile | Reduce sections/launches; replay modes have different costs, no fixed pass count |
| NCU-only hang/mismatch | Races, uninitialized state, host/device dependencies and replay compatibility; reverify |
| RTX timing jitter | Display contention, clocks, laptop power/thermals; repeat unprofiled trials |
| Section deployment warning | Directory permissions and documented section-folder option; do not repurpose HOME/USERPROFILE |

PM and context-switch trace have different support matrices. Current documentation
excludes vGPU for PM; WSL/MIG/multi-client MPS may limit context tracing without
disabling every PM metric. Check the installed version. Few samples cannot
resolve a short tail; different passes can be misaligned.

Use supported JIT line-information options and preserve generated artifacts.
A separately rebuilt Triton PTX harness need not reproduce its dispatch.
CUDA Graph behavior depends on supported graph/replay options.

NCU can serialize launches, flush caches and control clocks. Keep capture
policies consistent and retain separate unprofiled timing. Do not silently
change machine-wide permissions, persistent mode, clocks or driver settings
to make an example run.

Sources: [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
[release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html),
[counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
