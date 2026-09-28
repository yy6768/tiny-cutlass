#include <cmath>
#include <string>
#include "../kernel/default_flash_attn.h"
#include "flash_attn.h"

namespace {

cudaError_t to_cuda_error(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess: return cudaSuccess;
    case cutlass::Status::kErrorInvalidProblem:
    case cutlass::Status::kErrorMisalignedOperand:
    case cutlass::Status::kErrorWorkspaceNull: return cudaErrorInvalidValue;
    case cutlass::Status::kErrorArchMismatch:
    case cutlass::Status::kErrorNotSupported: return cudaErrorNotSupported;
    default: return cudaErrorUnknown;
  }
}

bool can_run(Problem const& p, std::string& reason) {
  if (p.batch_size <= 0 || p.head_number <= 0 ||
      p.seq_length <= 0 || p.seq_length_kv <= 0 || !std::isfinite(p.scale)) {
    reason = "Expected positive extents and finite scale";
    return false;
  }
  if ((p.head_size != 32 && p.head_size != 64 && p.head_size != 96 && p.head_size != 128) ||
      p.head_size_v != p.head_size) {
    reason = "02-split-kv requires D=Dv in {32,64,96,128}";
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
    reason = "02-split-kv has explicit SM80/SM89 policies only";
    return false;
  }
  if (p.batch_size > props.maxGridSize[0] || p.head_number > props.maxGridSize[1]) {
    reason = "Batch/head count exceeds the device grid limit";
    return false;
  }
  return true;
}

template <int D>
constexpr int kTileBc = (48 * 1024 / sizeof(Element) + 4 * D - 1) / (4 * D);

template <int D>
constexpr int kTileBr = kTileBc<D> < D ? kTileBc<D> : D;

template <class Arch, int D>
using SelectedKernel = typename DefaultFlashAttn<Arch, Element,
    cutlass::gemm::GemmShape<kTileBr<D>, kTileBc<D>, D>,
    cutlass::gemm::GemmShape<16, ((kTileBc<D> + 31) / 32) * 32, 16>,
    cutlass::gemm::GemmShape<16, D, 16>>::Kernel;

template <class Arch, int D>
cudaError_t launch(Problem const& p, Tensors const& t, Workspace workspace, cudaStream_t stream) {
  FlashAttn<SelectedKernel<Arch, D>> op;
  return to_cuda_error(op({p, t}, workspace, stream));
}

template <class Arch, int D>
std::size_t workspace_size_for(Problem const& p) {
  return FlashAttn<SelectedKernel<Arch, D>>::get_workspace_size({p, {}});
}

template <class Arch>
std::size_t workspace_size_dispatch(Problem const& p) {
  switch (p.head_size) {
    case 32: return workspace_size_for<Arch, 32>(p);
    case 64: return workspace_size_for<Arch, 64>(p);
    case 96: return workspace_size_for<Arch, 96>(p);
    case 128: return workspace_size_for<Arch, 128>(p);
    default: return 0;
  }
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

std::size_t workspace_bytes(Problem const& p) {
  int device = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&props, device) != cudaSuccess) return 0;
  switch (props.major * 10 + props.minor) {
    case 80: return workspace_size_dispatch<cutlass::arch::Sm80>(p);
    case 89: return workspace_size_dispatch<cutlass::arch::Sm89>(p);
    default: return 0;
  }
}

Kernel const kKernel = {"02-split-kv",
    "FlashAttention (CUTLASS TensorOp, FA1 KV-outer/Q-inner, Q-row warps)",
    workspace_bytes, can_run, run};
} // namespace

Kernel const& kernel_02_split_kv() { return kKernel; }
