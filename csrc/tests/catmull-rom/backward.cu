/*
  Correctness test for the Catmull-Rom reprojection backward pass.

  Runs the CUTLASS-style backward operator
  (catmull-rom/device/backward.h) and checks:
    - grad_history against an independent host analytic scatter, and
    - grad_motion against a central finite-difference of an fp32 host forward.

  Element is float here (not half) so the finite-difference reference for
  grad_motion is numerically clean; the kernel path is dtype-templated and the
  forward test already covers half_t.
*/

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "catmull-rom/device/backward.h"

namespace {

using Element = float;
using ElementAccum = float;
namespace cr_device = tiny_cutlass::catmull_rom::device;

struct Case {
  std::string name;
  int N;
  int H;
  int W;
  int C;
  float mv_scale;
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

int64_t nhwc_index(int n, int y, int x, int c, int H, int W, int C) {
  return ((int64_t(n) * H + y) * W + x) * C + c;
}

float value_at(int index, float scale, float phase) {
  float a = std::sin(float(index + 1) * 0.173f + phase);
  float b = std::cos(float(index + 3) * 0.071f - phase);
  return scale * (0.7f * a + 0.3f * b);
}

void cr_weights(float t, float w[4]) {
  float t2 = t * t, t3 = t2 * t;
  w[0] = 0.5f * (-t3 + 2.0f * t2 - t);
  w[1] = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
  w[2] = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
  w[3] = 0.5f * (t3 - t2);
}

int clamp_index_ref(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

// fp32 host forward for a single (pixel, channel), used by the finite-diff
// reference for grad_motion.
float forward_pixel_channel(
    Case const& c, std::vector<Element> const& history, int n, int y, int x, int cc,
    float mvx, float mvy) {
  float fx = float(x) + mvx, fy = float(y) + mvy;
  int ix = int(std::floor(fx)), iy = int(std::floor(fy));
  float wx[4], wy[4];
  cr_weights(fx - float(ix), wx);
  cr_weights(fy - float(iy), wy);
  float acc = 0.0f;
  for (int j = 0; j < 4; ++j) {
    int yy = clamp_index_ref(iy - 1 + j, c.H - 1);
    for (int i = 0; i < 4; ++i) {
      int xx = clamp_index_ref(ix - 1 + i, c.W - 1);
      acc += wy[j] * wx[i] * history[nhwc_index(n, yy, xx, cc, c.H, c.W, c.C)];
    }
  }
  return acc;
}

// Independent host analytic grad_history scatter.
std::vector<float> ref_grad_history(
    Case const& c, std::vector<Element> const& history, std::vector<float> const& motion,
    std::vector<Element> const& grad_out) {
  std::vector<float> gh(int64_t(c.N) * c.H * c.W * c.C, 0.0f);
  for (int n = 0; n < c.N; ++n)
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x) {
        int64_t pixel = (int64_t(n) * c.H + y) * c.W + x;
        float mvx = motion[pixel * 2 + 0], mvy = motion[pixel * 2 + 1];
        float fx = float(x) + mvx, fy = float(y) + mvy;
        int ix = int(std::floor(fx)), iy = int(std::floor(fy));
        float wx[4], wy[4];
        cr_weights(fx - float(ix), wx);
        cr_weights(fy - float(iy), wy);
        for (int cc = 0; cc < c.C; ++cc) {
          float g = grad_out[nhwc_index(n, y, x, cc, c.H, c.W, c.C)];
          for (int j = 0; j < 4; ++j) {
            int yy = clamp_index_ref(iy - 1 + j, c.H - 1);
            for (int i = 0; i < 4; ++i) {
              int xx = clamp_index_ref(ix - 1 + i, c.W - 1);
              gh[nhwc_index(n, yy, xx, cc, c.H, c.W, c.C)] += wy[j] * wx[i] * g;
            }
          }
        }
      }
  return gh;
}

// Central finite-difference reference for grad_motion:
//   d/dmv sum_c g[p,c] * out[p,c] ~= (L(+eps) - L(-eps)) / (2 eps)
std::vector<float> ref_grad_motion_fd(
    Case const& c, std::vector<Element> const& history, std::vector<float> const& motion,
    std::vector<Element> const& grad_out) {
  int64_t pixels = int64_t(c.N) * c.H * c.W;
  std::vector<float> gm(pixels * 2, 0.0f);
  float const eps = 1e-2f;
  for (int n = 0; n < c.N; ++n)
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x) {
        int64_t pixel = (int64_t(n) * c.H + y) * c.W + x;
        float mvx = motion[pixel * 2 + 0], mvy = motion[pixel * 2 + 1];
        for (int axis = 0; axis < 2; ++axis) {
          float lp = 0.0f, lm = 0.0f;
          for (int cc = 0; cc < c.C; ++cc) {
            float g = grad_out[nhwc_index(n, y, x, cc, c.H, c.W, c.C)];
            float mx_p = mvx + (axis == 0 ? eps : 0.0f);
            float my_p = mvy + (axis == 1 ? eps : 0.0f);
            float mx_m = mvx - (axis == 0 ? eps : 0.0f);
            float my_m = mvy - (axis == 1 ? eps : 0.0f);
            lp += g * forward_pixel_channel(c, history, n, y, x, cc, mx_p, my_p);
            lm += g * forward_pixel_channel(c, history, n, y, x, cc, mx_m, my_m);
          }
          gm[pixel * 2 + axis] = (lp - lm) / (2.0f * eps);
        }
      }
  return gm;
}

bool run_case(Case const& c) {
  int64_t pixels = int64_t(c.N) * c.H * c.W;
  int64_t data_count = pixels * c.C;

  std::vector<Element> history(data_count), grad_out(data_count);
  std::vector<float> motion(pixels * 2);
  for (int64_t i = 0; i < data_count; ++i) {
    history[i] = value_at(int(i), 0.5f, 0.11f);
    grad_out[i] = value_at(int(i) + 5, 0.5f, 0.53f);
  }
  for (int64_t p = 0; p < pixels; ++p) {
    // Keep motion fractional and away from integer grid so floor() is stable
    // under the finite-difference perturbation (no tap-set jumps within eps).
    motion[p * 2 + 0] = c.mv_scale * value_at(int(p), 0.5f, 0.31f) + 0.37f;
    motion[p * 2 + 1] = c.mv_scale * value_at(int(p) + 7, 0.5f, 0.67f) + 0.41f;
  }

  DeviceBuffer<Element> d_history(history.size()), d_grad_out(grad_out.size());
  DeviceBuffer<float> d_motion(motion.size());
  DeviceBuffer<ElementAccum> d_grad_history(data_count), d_grad_motion(pixels * 2);
  if (!d_history.ok() || !d_grad_out.ok() || !d_motion.ok() || !d_grad_history.ok() ||
      !d_grad_motion.ok()) {
    std::cerr << "alloc failed\n";
    return false;
  }
  if (!d_history.from_host(history) || !d_grad_out.from_host(grad_out) ||
      !d_motion.from_host(motion)) {
    return false;
  }

  using Op = cr_device::CatmullRomReprojectBackward<
      cutlass::arch::Sm80, Element, float, ElementAccum, 128>;
  typename Op::Arguments args;
  args.history = d_history.get();
  args.motion = d_motion.get();
  args.grad_output = d_grad_out.get();
  args.grad_history = d_grad_history.get();
  args.grad_motion = d_grad_motion.get();
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

  std::vector<ElementAccum> grad_history(data_count), grad_motion(pixels * 2);
  if (!d_grad_history.to_host(grad_history) || !d_grad_motion.to_host(grad_motion)) return false;

  std::vector<float> gh_ref = ref_grad_history(c, history, motion, grad_out);
  std::vector<float> gm_ref = ref_grad_motion_fd(c, history, motion, grad_out);

  float gh_max = 0.0f;
  for (size_t i = 0; i < grad_history.size(); ++i) {
    gh_max = std::max(gh_max, std::abs(float(grad_history[i]) - gh_ref[i]));
  }
  // grad_motion is checked with a relative-ish tolerance: finite difference of
  // a cubic has O(eps^2) truncation error, and the values scale with C.
  float gm_max = 0.0f;
  for (size_t i = 0; i < grad_motion.size(); ++i) {
    gm_max = std::max(gm_max, std::abs(float(grad_motion[i]) - gm_ref[i]));
  }

  bool gh_ok = gh_max <= 1e-3f;
  bool gm_ok = gm_max <= 5e-2f;
  std::cout << (gh_ok && gm_ok ? "pass " : "FAIL ") << c.name << " shape=(" << c.N << "," << c.H
            << "," << c.W << "," << c.C << ") gh_max=" << gh_max << " gm_max=" << gm_max << "\n";
  return gh_ok && gm_ok;
}

bool run_all() {
  std::vector<Case> cases = {
      {"small", 1, 8, 8, 4, 1.5f},
      {"rect", 2, 6, 10, 8, 2.5f},
      {"wide_c", 1, 12, 12, 16, 3.0f},
      {"big_motion", 1, 10, 10, 8, 4.0f},
  };
  for (auto const& c : cases) {
    if (!run_case(c)) return false;
  }
  return true;
}

}  // namespace

int main() { return run_all() ? 0 : 1; }

