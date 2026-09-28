#include <cmath>
#include <string>
#include "../kernel/default_flash_attn.h"
#include "flash_attn.h"

namespace {

bool can_run(Problem const& p, std::string& reason) {
  if (p.batch_size <= 0 || p.batch_size > 65535 || p.head_number <= 0 || p.head_number > 65535 ||
      p.seq_length <= 0 || p.seq_length_kv <= 0 || !std::isfinite(p.scale)) {
    reason = "Expected positive extents, finite scale and B/H <= 65535";
    return false;
  }
  if ((p.head_size != 32 && p.head_size != 64 && p.head_size != 96 && p.head_size != 128) ||
      p.head_size_v != p.head_size) {
    reason = "03-split-q requires D=Dv in {32,64,96,128}";
    return false;
  }
  int device = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    reason = "Cannot query current CUDA device";
    return false;
  }
  int cc = props.major * 10 + props.minor;
  if (cc != 80 && cc != 89) {
    reason = "03-split-q has explicit SM80/SM89 policies only";
    return false;
  }
  return true;
}

template <class Arch, int D>
cudaError_t launch(Problem const& p, Tensors const& t, Workspace workspace, cudaStream_t stream) {
  // Four warps own distinct 16-row slices of the Q/output tile.
  using Factory = DefaultFlashAttnSplitQ<Arch, Element,
      cutlass::gemm::GemmShape<64, 64, D>,
      cutlass::gemm::GemmShape<16, 64, 16>,
      cutlass::gemm::GemmShape<16, (D == 96 ? 128 : D), 16>>;
  FlashAttnSplitQ<typename Factory::Kernel> op;
  return op({p, t}, workspace, stream);
}

template <class Arch>
cudaError_t dispatch(Problem const& p, Tensors const& t, Workspace workspace, cudaStream_t stream) {
  switch (p.head_size) {
    case 32: return launch<Arch, 32>(p, t, workspace, stream);
    case 64: return launch<Arch, 64>(p, t, workspace, stream);
    case 96: return launch<Arch, 96>(p, t, workspace, stream);
    case 128: return launch<Arch, 128>(p, t, workspace, stream);
    default: return cudaErrorNotSupported;
  }
}

cudaError_t run(Problem const& p, Tensors const& t, Workspace workspace, cudaStream_t stream) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) return err;
  switch (props.major * 10 + props.minor) {
    case 80: return dispatch<cutlass::arch::Sm80>(p, t, workspace, stream);
    case 89: return dispatch<cutlass::arch::Sm89>(p, t, workspace, stream);
    default: return cudaErrorNotSupported;
  }
}

std::size_t workspace_bytes(Problem const&) { return 0; }

Kernel const kKernel = {"03-split-q",
    "FlashAttention (CUTLASS TensorOp, Q-tile CTA, split-Q warps, register P)",
    workspace_bytes, can_run, run};
} // namespace

Kernel const& kernel_03_split_q() { return kKernel; }
