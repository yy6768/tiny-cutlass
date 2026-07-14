/*
  Cross-check against an independently generated autodiff implementation.

  csrc/catmull-rom/slang/catmull_rom.slang implements
  the exact UE4 Bicubic2DCatmullRom math (Keys cubic, A = -0.5) as a
  [Differentiable] function; slangc compiles it to CUDA with an automatically
  generated backward (bwd_diff). That backward is the analytical gradient we
  validate our hand-written backward kernel against.

  This test:
    1. Builds a batch of random per-pixel problems: a 4x4 tap neighborhood,
       a motion vector, and integer target coords.
    2. Runs the Slang kernel (forward + d/dmv + d/dtaps) -> reference.
    3. Runs our forward interp (kernel/forward.h:
       catmull_rom_interp16) and our backward weight math to reproduce the same
       forward value and gradients, and compares.

  Because both sides use the identical UE weight formulas, agreement is at
  fp32 rounding level; any divergence flags a real bug in our weight or
  derivative code.
*/

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "catmull-rom/kernel/forward.h"
#include "catmull-rom/kernel/backward.h"

// The generated kernel declares:
//   struct CatmullRomGrads_0 { float forward_val, grad_mvx, grad_mvy; FixedArray<float,16> grad_taps; };
//   struct GlobalParams_0 { RWStructuredBuffer<CatmullRomGrads_0> results; StructuredBuffer<float> inputs; int count; };
//   extern "C" __constant__ GlobalParams_0 SLANG_globalParams;
//   extern "C" __global__ void crReference();
#include "catmull-rom/slang/catmull_rom.cu"

namespace {

int const kInputStride = 20;  // mvx, mvy, taps[16], x, y

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    count_ = count;
    if (count_) cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count_);
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  bool from_host(std::vector<T> const& h) {
    return h.empty() ||
           cudaMemcpy(ptr_, h.data(), sizeof(T) * h.size(), cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool to_host(std::vector<T>& h) const {
    return h.empty() ||
           cudaMemcpy(h.data(), ptr_, sizeof(T) * h.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

float value_at(int index, float scale, float phase) {
  float a = std::sin(float(index + 1) * 0.173f + phase);
  float b = std::cos(float(index + 3) * 0.071f - phase);
  return scale * (0.7f * a + 0.3f * b);
}

// Run the Slang reference kernel over `count` packed problems.
bool run_autodiff(
    std::vector<float> const& inputs, std::vector<CatmullRomGrads_0>& out, int count) {
  DeviceBuffer<float> d_inputs(inputs.size());
  DeviceBuffer<CatmullRomGrads_0> d_results(count);
  if (!d_inputs.from_host(inputs)) return false;

  GlobalParams_0 params;
  params.inputs_0.data = d_inputs.get();
  params.inputs_0.count = inputs.size();
  params.results_0.data = d_results.get();
  params.results_0.count = count;
  params.count_0 = count;

  cudaError_t err = cudaMemcpyToSymbol(SLANG_globalParams, &params, sizeof(params));
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpyToSymbol failed: " << cudaGetErrorString(err) << "\n";
    return false;
  }

  int const block = 64;  // must match [numthreads(64,1,1)]
  int const grid = (count + block - 1) / block;
  crReference<<<grid, block>>>();
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "generated kernel failed: " << cudaGetErrorString(cudaGetLastError()) << "\n";
    return false;
  }

  out.resize(count);
  return d_results.to_host(out);
}

}  // namespace

int main() {
  namespace k = tiny_cutlass::catmull_rom::kernel;

  int const count = 256;
  std::vector<float> inputs(int64_t(count) * kInputStride);

  // Build random-ish problems. Keep the fractional offset away from integer
  // grid lines so floor() is stable (the Slang forward uses floor too, but
  // exact-integer f would make grad_mv ill-defined on both sides).
  for (int p = 0; p < count; ++p) {
    int base = p * kInputStride;
    inputs[base + 0] = value_at(p, 1.7f, 0.31f) + 0.37f;        // mvx
    inputs[base + 1] = value_at(p + 11, 1.7f, 0.67f) + 0.41f;   // mvy
    for (int t = 0; t < 16; ++t) {
      inputs[base + 2 + t] = value_at(p * 16 + t, 0.9f, 0.13f);  // taps
    }
    inputs[base + 18] = float(3 + (p % 5));  // x
    inputs[base + 19] = float(2 + (p % 7));  // y
  }

  std::vector<CatmullRomGrads_0> ref;
  if (!run_autodiff(inputs, ref, count)) {
    std::cerr << "autodiff cross-check failed\n";
    return 1;
  }

  // Reproduce forward + gradients on the host with our own weight/derivative
  // functions (the same math the CUDA kernels use).
  float max_fwd = 0.0f, max_gmv = 0.0f, max_gtap = 0.0f;

  for (int p = 0; p < count; ++p) {
    int base = p * kInputStride;
    float mvx = inputs[base + 0];
    float mvy = inputs[base + 1];
    float taps[16];
    for (int t = 0; t < 16; ++t) taps[t] = inputs[base + 2 + t];
    int x = int(inputs[base + 18]);
    int y = int(inputs[base + 19]);

    float fx = float(x) + mvx, fy = float(y) + mvy;
    int ix = int(std::floor(fx)), iy = int(std::floor(fy));
    float tx = fx - float(ix), ty = fy - float(iy);

    // Forward via our shared core.
    float fwd = k::catmull_rom_interp16(taps, tx, ty);

    // Backward (grad_output = 1): reproduce grad_mv and grad_taps.
    float wx[4], wy[4], dwx[4], dwy[4];
    k::catmull_rom_weights(tx, wx);
    k::catmull_rom_weights(ty, wy);
    k::catmull_rom_weight_derivs(tx, dwx);
    k::catmull_rom_weight_derivs(ty, dwy);

    float gmx = 0.0f, gmy = 0.0f;
    float gtap[16];
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        float h = taps[j * 4 + i];
        gmx += wy[j] * dwx[i] * h;
        gmy += dwy[j] * wx[i] * h;
        gtap[j * 4 + i] = wy[j] * wx[i];
      }
    }

    // CatmullRomGrads_0 is { float forward_val, grad_mvx, grad_mvy; float[16] };
    // Its FixedArray operator[] is __device__-only, so read the POD as raw
    // floats: [0]=fwd [1]=gmvx [2]=gmvy [3..18]=grad_taps.
    float const* r = reinterpret_cast<float const*>(&ref[p]);
    max_fwd = std::max(max_fwd, std::abs(fwd - r[0]));
    max_gmv = std::max(max_gmv, std::abs(gmx - r[1]));
    max_gmv = std::max(max_gmv, std::abs(gmy - r[2]));
    for (int t = 0; t < 16; ++t) {
      max_gtap = std::max(max_gtap, std::abs(gtap[t] - r[3 + t]));
    }
  }

  bool ok = max_fwd <= 1e-4f && max_gmv <= 1e-3f && max_gtap <= 1e-5f;
  std::cout << (ok ? "pass" : "FAIL") << " autodiff cross-check (UE Catmull-Rom, A=-0.5)"
            << " count=" << count << " max_fwd=" << max_fwd << " max_grad_mv=" << max_gmv
            << " max_grad_tap=" << max_gtap << "\n";
  return ok ? 0 : 1;
}

