#pragma once

// Shared test utilities for tiny-cutlass kernel families (conv-fused,
// flash-attention, ...). Intentionally buffer-agnostic: everything here
// operates on host std::vector<T> and raw device pointers, so it works with
// both the hand-rolled DeviceBuffer below and cutlass::DeviceAllocation.
//
// Nothing in here depends on a specific operator, so a family test only needs
// to: fill host tensors -> upload -> run its op -> download -> build a
// reference -> compare.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime_api.h>

namespace tiny_cutlass::testing {

// ---------------------------------------------------------------------------
// Device buffer: minimal owning RAII wrapper over cudaMalloc/cudaMemcpy.
// (Kept instead of cutlass::DeviceAllocation so conv-fused tests don't need to
// pull in the cutlass util headers; API is a strict subset of what they share.)
// ---------------------------------------------------------------------------
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) { reset(count); }

  ~DeviceBuffer() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  bool reset(size_t count) {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = count;
    if (!count_) {
      return true;
    }
    cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * count_);
    if (error != cudaSuccess) {
      std::cerr << "cudaMalloc failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
  }

  bool copy_from_host(std::vector<T> const& host) {
    cudaError_t error = cudaMemcpy(
        ptr_, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(error) << "\n";
      return false;
    }
    return true;
  }

  bool copy_to_host(std::vector<T>& host) const {
    cudaError_t error = cudaMemcpy(
        host.data(), ptr_, sizeof(T) * host.size(), cudaMemcpyDeviceToHost);
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

// ---------------------------------------------------------------------------
// NHWC / KRSC index helpers (row-major packed).
// ---------------------------------------------------------------------------
inline int64_t nhwc_index(
    int n, int h, int w, int c, int height, int width, int channels) {
  return ((int64_t(n) * height + h) * width + w) * channels + c;
}

// 1x1 filter in KRSC with R=S=1 collapses to (k, c).
inline int64_t krsc_index(int k, int c, int channels) {
  return int64_t(k) * channels + c;
}

// ---------------------------------------------------------------------------
// Tensor fills.
// ---------------------------------------------------------------------------
// Deterministic, reproducible fill (no RNG). Good for correctness cases where
// a byte-exact CPU reference is compared.
inline float deterministic_value(int index, float scale, float phase) {
  float x = std::sin(float(index + 1) * 0.173f + phase);
  float y = std::cos(float(index + 3) * 0.071f - phase);
  return scale * (0.7f * x + 0.3f * y);
}

template <typename Element>
void fill_deterministic(std::vector<Element>& tensor, float scale, float phase) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = Element(deterministic_value(int(i), scale, phase));
  }
}

template <typename Element>
void fill_bias(std::vector<Element>& tensor, bool enabled, float scale, float phase) {
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = enabled ? Element(deterministic_value(int(i), scale, phase))
                        : Element(0.0f);
  }
}

// Random uniform fill (mirrors flash-attention's fill_random_uniform).
template <typename Element>
void fill_random_uniform(
    std::vector<Element>& tensor, int seed, float lo = -2.0f, float hi = 2.0f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor[i] = Element(dist(gen));
  }
}

// ---------------------------------------------------------------------------
// Comparison. Reports BOTH mean-abs-error (MAE) and max-abs so a family test
// can pick whichever verdict it wants (conv uses max_abs threshold, FA uses
// MAE); non-finite output fails immediately.
// ---------------------------------------------------------------------------
struct CompareResult {
  double mae = 0.0;
  float max_abs = 0.0f;
  int64_t max_index = 0;
  bool finite = true;
};

// Compares two host arrays cast to float. Element and Ref may differ (e.g.
// half output vs float reference).
template <typename Element, typename Ref>
CompareResult compare_host(
    std::vector<Element> const& output, std::vector<Ref> const& reference) {
  CompareResult result;
  int64_t const n = int64_t(std::min(output.size(), reference.size()));
  long double abs_sum = 0.0L;

  for (int64_t i = 0; i < n; ++i) {
    float actual = float(output[i]);
    float expected = float(reference[i]);
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
      result.finite = false;
      result.max_index = i;
      result.max_abs = std::numeric_limits<float>::infinity();
      result.mae = std::numeric_limits<double>::infinity();
      return result;
    }
    float diff = std::fabs(actual - expected);
    abs_sum += diff;
    if (diff > result.max_abs) {
      result.max_abs = diff;
      result.max_index = i;
    }
  }

  result.mae = n > 0 ? double(abs_sum / (long double)n) : 0.0;
  return result;
}

}  // namespace tiny_cutlass::testing
