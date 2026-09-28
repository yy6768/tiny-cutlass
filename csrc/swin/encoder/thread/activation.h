#pragma once

#include <type_traits>

#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>

namespace tiny_cutlass::swin::encoder::thread {

// Custom encoded exponential from cuda_dldn_swin_enc0_kernel.ptx.txt,
// LF lines 12432-12504 (repeated at 27015-27081). This includes the
// original score scaling; do not apply another attention scale before it.
// The bit construction is part of the operator, not a conversion to integer.
template <typename Element>
struct EncodedExponential {
  static_assert(std::is_same<Element, cutlass::half_t>::value,
                "The PTX exponential contract supports cutlass::half_t only.");

  CUTLASS_DEVICE
  Element operator()(Element score) const {
    __half const scale = __float2half_rn(0.017186790704727173f);
    __half const bound = __float2half_rn(0.55615234375f);
    __half const coefficient = __float2half_rn(0.92724609375f);
    __half const offset = __float2half_rn(1.375f);

    __half scaled = __hmul_rn(score.to_half(), scale);
    __half clipped = __hmin(__hmax(scaled, __hneg(bound)), bound);
    __half quadratic = __hfma(clipped, __hneg(clipped), coefficient);
    __half encoded = __hfma(clipped, quadratic, offset);
    unsigned short bits = static_cast<unsigned short>(
        (static_cast<unsigned int>(__half_as_ushort(encoded)) << 5) & 0x7fe0u);
    return Element(__ushort_as_half(bits));
  }
};

// Custom clipped polynomial gate from the supplied PTX, LF lines
// 40947-41008 (last repeated fragment at 42840-42868). This is neither
// CUTLASS GELU (erf) nor CUTLASS GELU_taylor (tanh). Each arithmetic step
// rounds to half separately; the multiply/add pairs must not contract.
template <typename Element>
struct ClippedGelu {
  static_assert(std::is_same<Element, cutlass::half_t>::value,
                "The PTX activation contract supports cutlass::half_t only.");

  CUTLASS_DEVICE
  Element operator()(Element x) const {
    __half const slope = __float2half_rn(0.4121621549129486f);
    __half const curvature = __float2half_rn(0.0810810774564743f);
    __half const midpoint = __float2half_rn(0.5f);
    __half const bound = __float2half_rn(2.0f);

    __half clipped = __hmin(__hmax(x.to_half(), __hneg(bound)), bound);
    __half magnitude_term = __hmul_rn(curvature, __habs(clipped));
    __half linear_term = __hsub_rn(slope, magnitude_term);
    __half signed_term = __hmul_rn(clipped, linear_term);
    __half gate = __hadd_rn(midpoint, signed_term);
    return Element(__hmul_rn(x.to_half(), gate));
  }
};

}  // namespace tiny_cutlass::swin::encoder::thread
