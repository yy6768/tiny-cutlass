/*
  Correctness test for Catmull-Rom reprojection.

  A small problem-size sweep run through the CUTLASS-style device operator
  (catmull-rom/device/forward.h), each checked against an
  independent CPU reference that computes the same 16-tap separable
  Catmull-Rom gather in fp32 with clamp-to-edge addressing. The reference
  shares no code with the device kernel path.
*/

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "catmull-rom/device/forward.h"
#include "cutlass/half.h"

namespace {

using DefaultElement = cutlass::half_t;
using DefaultMotion = float;
namespace cr_device = tiny_cutlass::catmull_rom::device;

struct Case {
  std::string name;
  int N;
  int H;
  int W;
  int C;
  float mv_scale;  // magnitude of the synthetic motion vectors (pixels)
  cutlass::Status expected_status = cutlass::Status::kSuccess;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) { reset(count); }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  bool reset(size_t count) {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = count;
    if (!count_) return true;
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count_);
    if (error != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
  }

  bool copy_from_host(std::vector<T> const& host) {
    cudaError_t error =
        cudaMemcpy(ptr_, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
  }

  bool copy_to_host(std::vector<T>& host) const {
    cudaError_t error =
        cudaMemcpy(host.data(), ptr_, sizeof(T) * host.size(), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
      std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
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

void catmull_rom_weights_ref(float t, float w[4]) {
  float t2 = t * t;
  float t3 = t2 * t;
  w[0] = 0.5f * (-t3 + 2.0f * t2 - t);
  w[1] = 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f);
  w[2] = 0.5f * (-3.0f * t3 + 4.0f * t2 + t);
  w[3] = 0.5f * (t3 - t2);
}

int clamp_index_ref(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

// Independent CPU reference: 16-tap separable Catmull-Rom gather in fp32 with
// clamp-to-edge addressing. Motion is (dx, dy) in pixels; source location is
// (x + dx, y + dy).
template <typename Element>
std::vector<float> reference(
    Case const& c,
    std::vector<Element> const& history,
    std::vector<float> const& motion) {
  std::vector<float> out(int64_t(c.N) * c.H * c.W * c.C, 0.0f);

  for (int n = 0; n < c.N; ++n) {
    for (int y = 0; y < c.H; ++y) {
      for (int x = 0; x < c.W; ++x) {
        int64_t pixel = (int64_t(n) * c.H + y) * c.W + x;
        float mvx = motion[pixel * 2 + 0];
        float mvy = motion[pixel * 2 + 1];
        float fx = float(x) + mvx;
        float fy = float(y) + mvy;
        int ix = int(std::floor(fx));
        int iy = int(std::floor(fy));
        float wx[4], wy[4];
        catmull_rom_weights_ref(fx - float(ix), wx);
        catmull_rom_weights_ref(fy - float(iy), wy);

        for (int cc = 0; cc < c.C; ++cc) {
          float acc = 0.0f;
          for (int j = 0; j < 4; ++j) {
            int yy = clamp_index_ref(iy - 1 + j, c.H - 1);
            float row_acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int xx = clamp_index_ref(ix - 1 + i, c.W - 1);
              row_acc += wx[i] * float(history[nhwc_index(n, yy, xx, cc, c.H, c.W, c.C)]);
            }
            acc += wy[j] * row_acc;
          }
          out[nhwc_index(n, y, x, cc, c.H, c.W, c.C)] = acc;
        }
      }
    }
  }
  return out;
}

template <typename Element>
bool run_case(Case const& c) {
  int64_t pixels = int64_t(c.N) * c.H * c.W;
  int64_t data_count = pixels * c.C;
  int64_t motion_count = pixels * 2;

  std::vector<Element> history(data_count);
  std::vector<float> motion(motion_count);
  std::vector<Element> output(data_count);

  for (int64_t i = 0; i < data_count; ++i) {
    history[i] = Element(value_at(int(i), 0.5f, 0.11f));
  }
  // Smoothly varying fractional motion so taps land off-grid.
  for (int64_t p = 0; p < pixels; ++p) {
    motion[p * 2 + 0] = c.mv_scale * value_at(int(p), 1.0f, 0.31f);
    motion[p * 2 + 1] = c.mv_scale * value_at(int(p) + 7, 1.0f, 0.67f);
  }

  DeviceBuffer<Element> d_history(history.size());
  DeviceBuffer<float> d_motion(motion.size());
  DeviceBuffer<Element> d_output(output.size());
  // A null pointer is only a failure when the buffer was meant to be non-empty
  // (a degenerate case like C == 0 has zero-sized buffers and must be rejected
  // by the op, not by this allocation guard).
  if ((data_count > 0 && (!d_history.get() || !d_output.get())) ||
      (motion_count > 0 && !d_motion.get())) {
    return false;
  }
  if (!history.empty() && !d_history.copy_from_host(history)) return false;
  if (!motion.empty() && !d_motion.copy_from_host(motion)) return false;

  using Op = cr_device::CatmullRomReproject<cutlass::arch::Sm80, Element, float, 128>;
  typename Op::Arguments args;
  args.history = d_history.get();
  args.motion = d_motion.get();
  args.output = d_output.get();
  args.N = c.N;
  args.H = c.H;
  args.W = c.W;
  args.C = c.C;

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

  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cerr << "case " << c.name << " failed synchronize: " << cudaGetErrorString(error) << "\n";
    return false;
  }

  if (!d_output.copy_to_host(output)) return false;

  std::vector<float> expected = reference<Element>(c, history, motion);
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

template <typename Element>
bool run_all() {
  std::vector<Case> cases = {
      {"small", 1, 8, 8, 4, 1.5f},
      {"rect", 2, 6, 10, 8, 2.5f},
      {"wide_c", 1, 12, 12, 32, 3.0f},
      {"big_motion", 1, 16, 16, 16, 5.0f},
      {"reject_zero_c", 1, 4, 4, 0, 1.0f, cutlass::Status::kErrorInvalidProblem},
  };
  for (auto const& c : cases) {
    if (!run_case<Element>(c)) return false;
  }
  return true;
}

}  // namespace

int main() { return run_all<DefaultElement>() ? 0 : 1; }

