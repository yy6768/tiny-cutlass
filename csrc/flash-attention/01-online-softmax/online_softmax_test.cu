/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Online Softmax Test Framework.

    This file provides a test harness for online softmax kernels following
    the CUTLASS fused multi-head attention example conventions.

    The framework handles:
      - Command-line option parsing
      - Input tensor allocation and initialization
      - Reference (naive) softmax computation on host
      - Correctness verification with configurable tolerances
      - Profiling with CUDA events

    You only need to implement:
      1. Your online softmax kernel (device function)
      2. The launch logic in TestbedOnlineSoftmax::launch_kernel_()

    Usage:
      $ ./online_softmax_test
      $ ./online_softmax_test --rows=2048 --cols=4096 --iterations=50
*/

/////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "online_softmax.cu"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gbps;       // memory bandwidth in GB/s
  cudaError_t error;
  bool passed;

  Result(
    double runtime_ms = 0,
    double gbps = 0,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gbps(gbps), error(error), passed(true) { }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;

  int rows;           // number of rows (batch of independent softmax)
  int cols;           // number of columns (softmax dimension)
  int iterations;

  //
  // Methods
  //

  Options():
    help(false),
    error(false),
    reference_check(true),
    rows(1024),
    cols(1024),
    iterations(20)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("rows", rows, 1024);
    cmd.get_cmd_line_argument("cols", cols, 1024);
    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("reference-check", reference_check, true);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "online_softmax_test\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --rows=<int>                Number of rows / independent softmax instances (default: 1024)\n"
      << "  --cols=<int>                Number of columns / softmax dimension (default: 1024)\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (default: 20)\n"
      << "  --reference-check=<bool>    If true, performs reference check (default: true)\n";

    return out;
  }

  /// Compute memory bandwidth in GB/s
  /// Softmax reads input and writes output: 2 * rows * cols * sizeof(float)
  double bandwidth_gbps(double runtime_s) const {
    int64_t bytes = int64_t(2) * rows * cols * sizeof(float);
    return double(bytes) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// TODO: Implement your online softmax kernel here
//
// Example signature:
//
// __global__ void online_softmax_kernel(
//     float const* __restrict__ input,    // [rows, cols], row-major
//     float*       __restrict__ output,   // [rows, cols], row-major
//     int rows,
//     int cols
// ) {
//     // Your online softmax implementation
//     // Each row is an independent softmax:
//     //   output[r][c] = exp(input[r][c] - max_r) / sum_r(exp(input[r][c] - max_r))
//     //
//     // "Online" means you compute max and sum in a single pass (or fused passes)
//     // rather than requiring separate passes for max, subtract, exp, sum, divide.
// }

/////////////////////////////////////////////////////////////////////////////////////////////////

class TestbedOnlineSoftmax {
public:

  using Element = float;

private:

  Options & options;
  uint32_t seed;

  cutlass::DeviceAllocation<Element> block_input;
  cutlass::DeviceAllocation<Element> block_output;

  std::vector<Element> host_input;
  std::vector<Element> host_output;
  std::vector<Element> host_ref_output;

public:

  TestbedOnlineSoftmax(
    Options &options_,
    uint32_t seed_ = 3080
  ):
    options(options_), seed(seed_) { }

private:

  /// Initialize input tensor with random data
  void initialize_() {

    int64_t total_elements = int64_t(options.rows) * options.cols;

    block_input.reset(total_elements);
    block_output.reset(total_elements);

    // Fill input with uniform random values in [-8, 8]
    cutlass::reference::device::BlockFillRandomUniform(
      block_input.get(), total_elements, seed, Element(8), Element(-8), 0);

    // Zero output
    cudaMemset(block_output.get(), 0, total_elements * sizeof(Element));

    // Copy input to host for reference computation
    host_input.resize(total_elements);
    host_output.resize(total_elements);
    host_ref_output.resize(total_elements);

    cutlass::device_memory::copy_to_host(host_input.data(), block_input.get(), total_elements);
  }

  /// Compute reference softmax on host (standard two-pass for numerical stability)
  void compute_reference_() {

    int rows = options.rows;
    int cols = options.cols;

    for (int r = 0; r < rows; ++r) {
      Element const* row_in = host_input.data() + r * cols;
      Element* row_out = host_ref_output.data() + r * cols;

      // Pass 1: find max
      Element row_max = row_in[0];
      for (int c = 1; c < cols; ++c) {
        row_max = std::max(row_max, row_in[c]);
      }

      // Pass 2: exp and sum
      Element row_sum = Element(0);
      for (int c = 0; c < cols; ++c) {
        row_out[c] = std::exp(row_in[c] - row_max);
        row_sum += row_out[c];
      }

      // Pass 3: normalize
      Element inv_sum = Element(1) / row_sum;
      for (int c = 0; c < cols; ++c) {
        row_out[c] *= inv_sum;
      }
    }
  }

  /// Verify output against reference
  bool verify_() {

    int64_t total_elements = int64_t(options.rows) * options.cols;

    // Copy kernel output to host
    cutlass::device_memory::copy_to_host(host_output.data(), block_output.get(), total_elements);

    float abs_tol = 5e-5f;
    float rel_tol = 1e-3f;

    for (int64_t i = 0; i < total_elements; ++i) {
      float diff = std::fabs(host_output[i] - host_ref_output[i]);
      float abs_ref = std::fabs(host_ref_output[i]) + 1e-7f;
      float relative_diff = diff / abs_ref;

      if (std::isnan(host_output[i]) || std::isinf(host_output[i]) ||
          (diff > abs_tol && relative_diff > rel_tol)) {
        int row = int(i / options.cols);
        int col = int(i % options.cols);
        printf("[row=%d, col=%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}\n",
               row, col, diff, relative_diff, host_output[i], host_ref_output[i]);
        return false;
      }
    }

    return true;
  }

  /// Launch online softmax kernel
  cudaError_t launch_kernel_() {

    dim3 grid(options.rows);
    dim3 block(256);
    int num_warps = (block.x + 31) / 32;
    // shared memory: per-warp (max, sum) pair
    int smem_bytes = 2 * num_warps * sizeof(float);

    online_softmax_kernel<<<grid, block, smem_bytes>>>(
        block_input.get(),
        block_output.get(),
        options.rows,
        options.cols
    );

    return cudaGetLastError();
  }

public:

  /// Execute kernel and measure runtime
  Result profile() {

    Result result;
    result.passed = false;

    // Initialize the problem
    initialize_();

    // Launch kernel once for correctness check
    cudaError_t launch_err = launch_kernel_();
    if (launch_err != cudaSuccess) {
      std::cerr << "Kernel launch error: " << cudaGetErrorString(launch_err) << std::endl;
      result.error = launch_err;
      return result;
    }

    result.error = cudaDeviceSynchronize();
    if (result.error != cudaSuccess) {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Verify correctness
    //
    result.passed = true;

    if (options.reference_check) {
      compute_reference_();
      result.passed = verify_();
    }

    //
    // Warm-up run
    //

    launch_kernel_();
    cudaDeviceSynchronize();

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }
    }

    // Record start
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {
      launch_kernel_();
    }

    //
    // Stop profiling loop
    //

    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsedTime() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gbps = options.bandwidth_gbps(result.runtime_ms / 1000.0);

    //
    // Cleanup
    //

    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    std::cout << std::endl;
    std::cout << "Online Softmax:\n"
      << "====================================================" << std::endl;
    std::cout << "    " << "{rows, cols} = {" << options.rows
      << ", " << options.cols << "}" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "    " << "Bandwidth: " << result.gbps << " GB/s" << std::endl;

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  std::cout << "Device: " << props.name << " (SM" << props.major << props.minor << ")" << std::endl;

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run test
  //

  TestbedOnlineSoftmax testbed(options);

  Result result = testbed.profile();
  if (!result.passed) {
    std::cout << "\nFailed\n";
    return -1;
  }

  std::cout << "\nPassed\n";
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
