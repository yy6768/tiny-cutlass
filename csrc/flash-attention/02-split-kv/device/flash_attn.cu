#include <cmath>
#include <string>
#include "../kernel/default_flash_attn.h"
#include "flash_attn.h"

namespace {

std::size_t workspace_bytes(Problem const&) { return 0; }

bool can_run(Problem const& p, std::string& reason) {
  if (p.batch_size <= 0 || p.batch_size > 65535 || p.head_number <= 0 || p.head_number > 65535 ||
      p.seq_length <= 0 || p.seq_length_kv <= 0 || !std::isfinite(p.scale)) {
    reason = "Expected positive extents, finite scale and B/H <= 65535";
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
  return true;
}

template <class Arch, int D>
cudaError_t launch(Problem const& p, Tensors const& t, cudaStream_t stream) {
  // The device layer is the only place that selects a concrete learning policy.
  // QK's K extent describes its padded shared-memory iterator; the mainloop still
  // executes exactly D / 16 mma instruction groups. PV covers the padded 128
  // output channels with four N=32 warps and predicates the real D columns.
  static constexpr int kPitchQ = ((D + 63) / 64) * 64;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, D>;
  using WarpShapeQK = cutlass::gemm::GemmShape<32, 16, kPitchQ>;
  using WarpShapePV = cutlass::gemm::GemmShape<32, 32, 64>;
  using Factory = DefaultFlashAttn<
      Arch, Element, ThreadblockShape, WarpShapeQK, WarpShapePV, 2>;
  FlashAttn<typename Factory::Kernel> op;
  cudaError_t err = op.initialize({p, t});
  return err == cudaSuccess ? op.run(stream) : err;
}

template <class Arch>
cudaError_t dispatch(Problem const& p, Tensors const& t, cudaStream_t stream) {
  switch (p.head_size) {
    case 32: return launch<Arch, 32>(p, t, stream);
    case 64: return launch<Arch, 64>(p, t, stream);
    case 96: return launch<Arch, 96>(p, t, stream);
    case 128: return launch<Arch, 128>(p, t, stream);
    default: return cudaErrorNotSupported;
  }
}

cudaError_t run(Problem const& p, Tensors const& t, Workspace, cudaStream_t stream) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  cudaDeviceProp props{};
  err = cudaGetDeviceProperties(&props, device);
  if (err != cudaSuccess) return err;
  switch (props.major * 10 + props.minor) {
    case 80: return dispatch<cutlass::arch::Sm80>(p, t, stream);
    case 89: return dispatch<cutlass::arch::Sm89>(p, t, stream);
    default: return cudaErrorNotSupported;
  }
}

Kernel const kKernel = {"02-split-kv",
    "Split-KV attention (CUTLASS warp MMA, FP32 online softmax, 2-stage async K)",
    workspace_bytes, can_run, run};
} // namespace

Kernel const& kernel_02_split_kv() { return kKernel; }
