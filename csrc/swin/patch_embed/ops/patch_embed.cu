/*
  Explicit instantiation of the PatchEmbed public API.

  The arch / dtype / tile-shape choice lives here (and only here), keeping those
  names out of the primary operator and the public header, per the workspace
  rules. Adding a precision means adding an instantiation at the bottom of this
  file, not renaming anything.
*/

#include "swin/patch_embed/ops/patch_embed.h"

#include "cutlass/arch/arch.h"
#include "cutlass/half.h"

#include "swin/patch_embed/device/patch_embed.h"

namespace tiny_cutlass::swin::patch_embed {

namespace {

// The one place the concrete arch is chosen. SM89 runs Sm80 TensorOp kernels.
template <typename Element, typename ElementCompute>
using Operator = device::PatchEmbed<cutlass::arch::Sm80, Element, ElementCompute>;

template <typename Element, typename ElementCompute>
typename Operator<Element, ElementCompute>::Arguments to_device_arguments(
    PatchEmbedArguments<Element, ElementCompute> const& args) {
  typename Operator<Element, ElementCompute>::Arguments out;
  out.problem = args.problem;
  out.input = args.input;
  out.filter = args.filter;
  out.bias = args.bias;
  out.gamma = args.gamma;
  out.beta = args.beta;
  out.output = args.output;
  out.epsilon = args.epsilon;
  return out;
}

}  // namespace

template <typename Element, typename ElementCompute>
cutlass::Status patch_embed(
    PatchEmbedArguments<Element, ElementCompute> const& args) {
  Operator<Element, ElementCompute> op;
  return op(to_device_arguments(args), nullptr, args.stream);
}

template <typename Element, typename ElementCompute>
cutlass::Status patch_embed_can_implement(
    PatchEmbedArguments<Element, ElementCompute> const& args) {
  return Operator<Element, ElementCompute>::can_implement(
      to_device_arguments(args));
}

template <typename Element, typename ElementCompute>
size_t patch_embed_shared_storage_size() {
  return Operator<Element, ElementCompute>::shared_storage_size();
}

//
// Explicit instantiations
//

template cutlass::Status patch_embed<cutlass::half_t, float>(
    PatchEmbedArguments<cutlass::half_t, float> const&);

template cutlass::Status patch_embed_can_implement<cutlass::half_t, float>(
    PatchEmbedArguments<cutlass::half_t, float> const&);

template size_t patch_embed_shared_storage_size<cutlass::half_t, float>();

}  // namespace tiny_cutlass::swin::patch_embed
