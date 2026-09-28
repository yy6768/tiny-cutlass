#include <cmath>
#include <string>
#include "../kernel/default_flash_attn_backward.h"
#include "flash_attn_backward.h"

namespace {

template <class Arch, int D>
using BackwardFactory = DefaultFlashAttnBackward<Arch, Element,
    cutlass::gemm::GemmShape<64, 64, D>,
    cutlass::gemm::GemmShape<16, 64, 16>,
    cutlass::gemm::GemmShape<16, (D <= 64 ? 64 : 128), 16>,
    cutlass::gemm::GemmShape<32, (D <= 64 ? 32 : 64), 16>>;

template <class Arch, int D>
cudaError_t launch(Problem const& p, BackwardTensors const& t,
                   Workspace workspace, cudaStream_t stream) {
  using Factory = BackwardFactory<Arch, D>;
  using Device = FlashAttnBackward<typename Factory::Kernel>;
  return Device{}({p, t}, workspace, stream);
}

template <class Arch>
cudaError_t dispatch(Problem const& p, BackwardTensors const& t,
                     Workspace workspace, cudaStream_t stream) {
  switch (p.head_size) {
    case 32: return launch<Arch, 32>(p, t, workspace, stream);
    case 64: return launch<Arch, 64>(p, t, workspace, stream);
    case 96: return launch<Arch, 96>(p, t, workspace, stream);
    case 128: return launch<Arch, 128>(p, t, workspace, stream);
    default: return cudaErrorNotSupported;
  }
}

template <int D>
std::size_t workspace_for(Problem const& p) {
  using Factory = BackwardFactory<cutlass::arch::Sm80, D>;
  using Device = FlashAttnBackward<typename Factory::Kernel>;
  return Device::get_workspace_size(p);
}

std::size_t workspace_bytes(Problem const& p) {
  switch (p.head_size) {
    case 32: return workspace_for<32>(p);
    case 64: return workspace_for<64>(p);
    case 96: return workspace_for<96>(p);
    case 128: return workspace_for<128>(p);
    default: return 0;
  }
}

bool can_run(Problem const& p, std::string& reason) {
  if ((p.head_size != 32 && p.head_size != 64 && p.head_size != 96 && p.head_size != 128) ||
      p.head_size_v != p.head_size) {
    reason = "03-split-q backward requires D=Dv in {32,64,96,128}";
    return false;
  }
  if (!workspace_bytes(p)) {
    reason = "03-split-q backward requires finite scale, valid positive extents and representable workspace";
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
    reason = "03-split-q backward has explicit SM80/SM89 policies only";
    return false;
  }
  std::size_t bytes = 0;
  switch (p.head_size) {
    case 32: bytes = sizeof(BackwardFactory<cutlass::arch::Sm80, 32>::Kernel::SharedStorage); break;
    case 64: bytes = sizeof(BackwardFactory<cutlass::arch::Sm80, 64>::Kernel::SharedStorage); break;
    case 96: bytes = sizeof(BackwardFactory<cutlass::arch::Sm80, 96>::Kernel::SharedStorage); break;
    case 128: bytes = sizeof(BackwardFactory<cutlass::arch::Sm80, 128>::Kernel::SharedStorage); break;
  }
  if (bytes > std::size_t(props.sharedMemPerBlockOptin)) {
    reason = "03-split-q backward exceeds the device opt-in shared-memory limit";
    return false;
  }
  return true;
}

cudaError_t run(Problem const& p, BackwardTensors const& t,
                Workspace workspace, cudaStream_t stream) {
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

BackwardKernel const kBackward = {"03-split-q", workspace_bytes, can_run, run};
} // namespace

BackwardKernel const& backward_03_split_q() { return kBackward; }
