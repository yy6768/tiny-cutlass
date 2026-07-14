/*
  Correctness test for depth-based motion dilation.

  Runs the CUTLASS-style MotionDilation operator against an independent host
  reference that performs the same 5-tap cross depth argmax + motion copy.
  The tie-break (>) must match between device and host, so the reference uses
  the identical comparison order (center first, then L, R, U, D).
*/

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "catmull-rom/device/dilation.h"
#include "cutlass/half.h"

namespace {

using Element = cutlass::half_t;
using ElementDepth = float;
using ElementMotion = float;
namespace cr_device = tiny_cutlass::catmull_rom::device;

struct Case {
  std::string name;
  int N;
  int H;
  int W;
  cutlass::Status expected_status = cutlass::Status::kSuccess;
};

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
  bool ok() const { return count_ == 0 || ptr_ != nullptr; }
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

// Independent host reference: NearerIsGreater = true (larger depth = nearer).
// Comparison order matches the device kernel exactly: center, L, R, U, D, with
// strict '>' so ties keep the earlier tap.
std::vector<ElementMotion> reference(
    Case const& c, std::vector<ElementDepth> const& depth,
    std::vector<ElementMotion> const& motion) {
  std::vector<ElementMotion> out(int64_t(c.N) * c.H * c.W * 2, 0.0f);
  int const dx_off[5] = {0, -1, 1, 0, 0};
  int const dy_off[5] = {0, 0, 0, -1, 1};

  for (int n = 0; n < c.N; ++n)
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x) {
        int64_t plane = int64_t(n) * c.H;
        int pixel = int((plane + y) * c.W + x);
        int best_pixel = pixel;
        float best_depth = depth[pixel];
        for (int k = 1; k < 5; ++k) {
          int nx = x + dx_off[k], ny = y + dy_off[k];
          if (nx < 0 || nx >= c.W || ny < 0 || ny >= c.H) continue;
          int npix = int((plane + ny) * c.W + nx);
          if (depth[npix] > best_depth) {
            best_depth = depth[npix];
            best_pixel = npix;
          }
        }
        out[pixel * 2 + 0] = motion[best_pixel * 2 + 0];
        out[pixel * 2 + 1] = motion[best_pixel * 2 + 1];
      }
  return out;
}

bool run_case(Case const& c) {
  int64_t pixels = int64_t(c.N) * c.H * c.W;
  std::vector<ElementDepth> depth(pixels);
  std::vector<ElementMotion> motion(pixels * 2);
  for (int64_t p = 0; p < pixels; ++p) {
    depth[p] = value_at(int(p), 1.0f, 0.19f);
    motion[p * 2 + 0] = value_at(int(p), 2.0f, 0.31f);
    motion[p * 2 + 1] = value_at(int(p) + 7, 2.0f, 0.67f);
  }

  DeviceBuffer<ElementDepth> d_depth(depth.size());
  DeviceBuffer<ElementMotion> d_motion(motion.size());
  DeviceBuffer<ElementMotion> d_output(motion.size());
  bool empty = (pixels == 0);
  if ((!empty && (!d_depth.ok() || !d_motion.ok() || !d_output.ok()))) return false;
  if (!d_depth.from_host(depth) || !d_motion.from_host(motion)) return false;

  using Op = cr_device::MotionDilation<cutlass::arch::Sm80, ElementDepth, ElementMotion, true, 256>;
  typename Op::Arguments args;
  args.depth = d_depth.get();
  args.motion = d_motion.get();
  args.output = d_output.get();
  args.N = c.N;
  args.H = c.H;
  args.W = c.W;

  Op op;
  cutlass::Status status = op(args);
  if (status != c.expected_status) {
    std::cerr << "case " << c.name << " returned " << cutlassGetStatusString(status)
              << ", expected " << cutlassGetStatusString(c.expected_status) << "\n";
    return false;
  }
  if (status != cutlass::Status::kSuccess) {
    std::cout << "pass " << c.name << " rejected with " << cutlassGetStatusString(status) << "\n";
    return true;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "case " << c.name << " sync failed: "
              << cudaGetErrorString(cudaGetLastError()) << "\n";
    return false;
  }

  std::vector<ElementMotion> output(motion.size());
  if (!d_output.to_host(output)) return false;

  std::vector<ElementMotion> expected = reference(c, depth, motion);
  float max_abs = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    float diff = std::abs(float(output[i]) - float(expected[i]));
    max_abs = std::max(max_abs, diff);
    if (diff > 1e-6f) {
      std::cerr << "case " << c.name << " mismatch at " << i << " actual=" << output[i]
                << " expected=" << expected[i] << "\n";
      return false;
    }
  }

  std::cout << "pass " << c.name << " shape=(" << c.N << "," << c.H << "," << c.W << ")"
            << " max_abs=" << max_abs << "\n";
  return true;
}

bool run_all() {
  std::vector<Case> cases = {
      {"small", 1, 8, 8},
      {"rect", 2, 6, 10},
      {"single_row", 1, 1, 16},
      {"single_col", 1, 16, 1},
      {"reject_zero_h", 1, 0, 8, cutlass::Status::kErrorInvalidProblem},
  };
  for (auto const& c : cases) {
    if (!run_case(c)) return false;
  }
  return true;
}


struct FusedCase {
  std::string name;
  int N;
  int H;
  int W;
  int C;
  float mv_scale;
};

int64_t nhwc_index(int n, int y, int x, int c, int H, int W, int C) {
  return ((int64_t(n) * H + y) * W + x) * C + c;
}

void cr_weights(float t, float A, float w[4]) {
  float const x0 = 1.0f + t, x1 = t, x2 = 1.0f - t, x3 = 2.0f - t;
  w[0] = ((A * x0 - 5.0f * A) * x0 + 8.0f * A) * x0 - 4.0f * A;
  w[1] = ((A + 2.0f) * x1 - (A + 3.0f)) * x1 * x1 + 1.0f;
  w[2] = ((A + 2.0f) * x2 - (A + 3.0f)) * x2 * x2 + 1.0f;
  w[3] = ((A * x3 - 5.0f * A) * x3 + 8.0f * A) * x3 - 4.0f * A;
}

int clamp_index_ref(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

template <typename Elem>
std::vector<float> fused_reference(
    FusedCase const& c, std::vector<Elem> const& history, std::vector<ElementDepth> const& depth,
    std::vector<ElementMotion> const& motion) {
  float const A = -0.5f;
  int const dx_off[5] = {0, -1, 1, 0, 0};
  int const dy_off[5] = {0, 0, 0, -1, 1};

  std::vector<float> out(int64_t(c.N) * c.H * c.W * c.C, 0.0f);

  for (int n = 0; n < c.N; ++n)
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x) {
        int64_t plane = int64_t(n) * c.H;
        int pixel = int((plane + y) * c.W + x);

        // Stage 1: depth dilation (NearerIsGreater = true).
        int best_pixel = pixel;
        float best_depth = depth[pixel];
        for (int k = 1; k < 5; ++k) {
          int nx = x + dx_off[k], ny = y + dy_off[k];
          if (nx < 0 || nx >= c.W || ny < 0 || ny >= c.H) continue;
          int npix = int((plane + ny) * c.W + nx);
          if (depth[npix] > best_depth) {
            best_depth = depth[npix];
            best_pixel = npix;
          }
        }
        float mvx = motion[best_pixel * 2 + 0];
        float mvy = motion[best_pixel * 2 + 1];

        // Stage 2: Catmull-Rom gather.
        float fx = float(x) + mvx, fy = float(y) + mvy;
        int ix = int(std::floor(fx)), iy = int(std::floor(fy));
        float wx[4], wy[4];
        cr_weights(fx - float(ix), A, wx);
        cr_weights(fy - float(iy), A, wy);
        for (int cc = 0; cc < c.C; ++cc) {
          float acc = 0.0f;
          for (int j = 0; j < 4; ++j) {
            int yy = clamp_index_ref(iy - 1 + j, c.H - 1);
            for (int i = 0; i < 4; ++i) {
              int xx = clamp_index_ref(ix - 1 + i, c.W - 1);
              acc += wy[j] * wx[i] * float(history[nhwc_index(n, yy, xx, cc, c.H, c.W, c.C)]);
            }
          }
          out[nhwc_index(n, y, x, cc, c.H, c.W, c.C)] = acc;
        }
      }
  return out;
}

bool run_fused_case(FusedCase const& c) {
  int64_t pixels = int64_t(c.N) * c.H * c.W;
  int64_t data_count = pixels * c.C;

  std::vector<Element> history(data_count), output(data_count);
  std::vector<ElementDepth> depth(pixels);
  std::vector<ElementMotion> motion(pixels * 2);

  for (int64_t i = 0; i < data_count; ++i) history[i] = Element(value_at(int(i), 0.5f, 0.11f));
  for (int64_t p = 0; p < pixels; ++p) {
    depth[p] = value_at(int(p), 1.0f, 0.19f);
    motion[p * 2 + 0] = c.mv_scale * value_at(int(p), 1.0f, 0.31f);
    motion[p * 2 + 1] = c.mv_scale * value_at(int(p) + 7, 1.0f, 0.67f);
  }

  DeviceBuffer<Element> d_history(history.size()), d_output(output.size());
  DeviceBuffer<ElementDepth> d_depth(depth.size());
  DeviceBuffer<ElementMotion> d_motion(motion.size());
  if (!d_history.ok() || !d_output.ok() || !d_depth.ok() || !d_motion.ok()) return false;
  if (!d_history.from_host(history) || !d_depth.from_host(depth) || !d_motion.from_host(motion))
    return false;

  using Op = cr_device::DilatedCatmullReproject<
      cutlass::arch::Sm80, Element, ElementDepth, ElementMotion, true, 128>;
  typename Op::Arguments args;
  args.history = d_history.get();
  args.depth = d_depth.get();
  args.motion = d_motion.get();
  args.output = d_output.get();
  args.N = c.N;
  args.H = c.H;
  args.W = c.W;
  args.C = c.C;

  Op op;
  cutlass::Status status = op(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "case " << c.name << " op failed: " << cutlassGetStatusString(status) << "\n";
    return false;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "case " << c.name << " sync failed: "
              << cudaGetErrorString(cudaGetLastError()) << "\n";
    return false;
  }

  if (!d_output.to_host(output)) return false;

  std::vector<float> expected = fused_reference<Element>(c, history, depth, motion);
  float max_abs = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    float diff = std::abs(float(output[i]) - expected[i]);
    max_abs = std::max(max_abs, diff);
    if (diff > 0.02f) {
      std::cerr << "case " << c.name << " mismatch at " << i << " actual=" << float(output[i])
                << " expected=" << expected[i] << " diff=" << diff << "\n";
      return false;
    }
  }

  std::cout << "pass " << c.name << " shape=(" << c.N << "," << c.H << "," << c.W << "," << c.C
            << ") mv_scale=" << c.mv_scale << " max_abs=" << max_abs << "\n";
  return true;
}

bool run_fused_all() {
  std::vector<FusedCase> cases = {
      {"small", 1, 8, 8, 4, 1.5f},
      {"rect", 2, 6, 10, 8, 2.5f},
      {"wide_c", 1, 12, 12, 32, 3.0f},
      {"big_motion", 1, 16, 16, 16, 5.0f},
  };
  for (auto const& c : cases) {
    if (!run_fused_case(c)) return false;
  }
  return true;
}

}  // namespace

int main() { return run_all() && run_fused_all() ? 0 : 1; }
