/*
  Explicit instantiation of the Swin MLP public API. The arch / dtype choice
  lives here and nowhere else.
*/

#include "swin/swin_mlp/ops/swin_mlp.h"

#include "cutlass/arch/arch.h"
#include "cutlass/half.h"

#include "swin/swin_mlp/device/swin_mlp.h"

namespace tiny_cutlass::swin::swin_mlp {

namespace {

template <typename Element, typename ElementCompute>
using Operator = device::SwinMlp<cutlass::arch::Sm80, Element, ElementCompute>;

template <typename Element, typename ElementCompute>
typename Operator<Element, ElementCompute>::Arguments to_device_arguments(
    SwinMlpArguments<Element, ElementCompute> const& args) {
  typename Operator<Element, ElementCompute>::Arguments out;
  out.rows = args.rows;
  out.channels = args.channels;
  out.hidden = args.hidden;
  out.input = args.input;
  out.gamma = args.gamma;
  out.beta = args.beta;
  out.fc1_weight = args.fc1_weight;
  out.fc1_bias = args.fc1_bias;
  out.fc2_weight = args.fc2_weight;
  out.fc2_bias = args.fc2_bias;
  out.output = args.output;
  out.normalized = args.normalized;
  out.hidden_buffer = args.hidden_buffer;
  out.epsilon = args.epsilon;
  return out;
}

}  // namespace

template <typename Element, typename ElementCompute>
cutlass::Status swin_mlp(SwinMlpArguments<Element, ElementCompute> const& args) {
  Operator<Element, ElementCompute> op;
  return op(to_device_arguments(args), args.stream);
}

template <typename Element, typename ElementCompute>
cutlass::Status swin_mlp_can_implement(
    SwinMlpArguments<Element, ElementCompute> const& args) {
  return Operator<Element, ElementCompute>::can_implement(
      to_device_arguments(args));
}

template <typename Element, typename ElementCompute>
size_t swin_mlp_normalized_size(int rows, int channels) {
  return Operator<Element, ElementCompute>::normalized_size(rows, channels);
}

template <typename Element, typename ElementCompute>
size_t swin_mlp_hidden_size(int rows, int hidden) {
  return Operator<Element, ElementCompute>::hidden_size(rows, hidden);
}

//
// Explicit instantiations
//

template cutlass::Status swin_mlp<cutlass::half_t, float>(
    SwinMlpArguments<cutlass::half_t, float> const&);

template cutlass::Status swin_mlp_can_implement<cutlass::half_t, float>(
    SwinMlpArguments<cutlass::half_t, float> const&);

template size_t swin_mlp_normalized_size<cutlass::half_t, float>(int, int);

template size_t swin_mlp_hidden_size<cutlass::half_t, float>(int, int);

}  // namespace tiny_cutlass::swin::swin_mlp
