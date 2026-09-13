/*
  Explicit instantiation of the PatchMerging public API. The arch / dtype choice
  lives here and nowhere else.
*/

#include "swin/patch_merging/ops/patch_merging.h"

#include "cutlass/arch/arch.h"
#include "cutlass/half.h"

#include "swin/patch_merging/device/patch_merging.h"

namespace tiny_cutlass::swin::patch_merging {

namespace {

template <typename Element, typename ElementCompute>
using Operator =
    device::PatchMerging<cutlass::arch::Sm80, Element, ElementCompute>;

template <typename Element, typename ElementCompute>
typename Operator<Element, ElementCompute>::Arguments to_device_arguments(
    PatchMergingArguments<Element, ElementCompute> const& args) {
  typename Operator<Element, ElementCompute>::Arguments out;
  out.problem = args.problem;
  out.input = args.input;
  out.gather = args.gather;
  out.gamma = args.gamma;
  out.beta = args.beta;
  out.weight = args.weight;
  out.bias = args.bias;
  out.output = args.output;
  out.normalized = args.normalized;
  out.epsilon = args.epsilon;
  return out;
}

}  // namespace

template <typename Element, typename ElementCompute>
cutlass::Status patch_merging(
    PatchMergingArguments<Element, ElementCompute> const& args) {
  Operator<Element, ElementCompute> op;
  return op(to_device_arguments(args), args.stream);
}

template <typename Element, typename ElementCompute>
cutlass::Status patch_merging_can_implement(
    PatchMergingArguments<Element, ElementCompute> const& args) {
  return Operator<Element, ElementCompute>::can_implement(
      to_device_arguments(args));
}

template <typename Element, typename ElementCompute>
size_t patch_merging_normalized_size(PatchMergingProblem const& problem) {
  return Operator<Element, ElementCompute>::normalized_size(problem);
}

//
// Explicit instantiations
//

template cutlass::Status patch_merging<cutlass::half_t, float>(
    PatchMergingArguments<cutlass::half_t, float> const&);

template cutlass::Status patch_merging_can_implement<cutlass::half_t, float>(
    PatchMergingArguments<cutlass::half_t, float> const&);

template size_t patch_merging_normalized_size<cutlass::half_t, float>(
    PatchMergingProblem const&);

}  // namespace tiny_cutlass::swin::patch_merging
