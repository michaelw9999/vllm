// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Placeholder bridge for the upcoming SINQ WMMA CUDA kernels. The real
// implementation will live in the external SINQ repository and link against
// these entry points.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include <type_traits>

namespace vllm {
namespace sinq {

template <typename T>
struct NativeType {
  using type = T;
  static constexpr int64_t kVectorWidth = 16 / sizeof(T);
};

template <int Bits>
struct QuantTraits {
  static_assert(
      Bits == 1 || Bits == 2 || Bits == 3 || Bits == 4 || Bits == 6 ||
          Bits == 8,
      "Unsupported SINQ bit width");
  static constexpr int kBits = Bits;
  static constexpr int kElementsPerByte = 8 / Bits;
  static constexpr bool kVectorizable = (Bits % 2) == 0 || Bits == 1;
};

namespace {

template <typename Scalar, int Bits>
at::Tensor dispatch_placeholder(
    const at::Tensor& input,
    const at::Tensor& packed_weight,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& zero_point,
    const c10::optional<at::Tensor>& scale_2,
    const c10::optional<at::Tensor>& codebook,
    const c10::Dict<c10::IValue, c10::IValue>& runtime_meta,
    const c10::optional<at::Tensor>& bias) {
  static_assert(std::is_same_v<Scalar, at::Half> ||
                    std::is_same_v<Scalar, at::BFloat16>,
                "SINQ kernel expects half or bfloat16 activations");
  [[maybe_unused]] constexpr QuantTraits<Bits> kQuant{};
  [[maybe_unused]] constexpr int64_t kVectorWidth =
      NativeType<Scalar>::kVectorWidth;

  (void)input;
  (void)packed_weight;
  (void)scale;
  (void)zero_point;
  (void)scale_2;
  (void)codebook;
  (void)runtime_meta;
  (void)bias;

  TORCH_CHECK(
      false,
      "SINQ WMMA kernel was not linked into this build. Compile vLLM with the "
      "external SINQ CUDA sources to enable this path.");
  return at::Tensor();
}

}  // namespace

at::Tensor sinq_linear_forward(
    const at::Tensor& input,
    const at::Tensor& packed_weight,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& zero_point,
    const c10::optional<at::Tensor>& scale_2,
    const c10::optional<at::Tensor>& codebook,
    const c10::Dict<c10::IValue, c10::IValue>& runtime_meta,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(input.is_cuda(), "SINQ expects CUDA input activations.");
  TORCH_CHECK(packed_weight.is_cuda(),
              "SINQ expects CUDA-resident packed weights.");
  TORCH_CHECK(input.dim() == 2, "SINQ linear expects rank-2 input tensors.");

  const auto bit_entry = runtime_meta.at(c10::IValue("weight_bits"));
  TORCH_CHECK(bit_entry.isInt(),
              "runtime_meta['weight_bits'] must be provided as an integer.");
  const int64_t weight_bits = bit_entry.toInt();

  at::Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "sinq_linear_forward", [&] {
        switch (weight_bits) {
          case 1:
            output = dispatch_placeholder<scalar_t, 1>(
                input, packed_weight, scale, zero_point, scale_2, codebook,
                runtime_meta, bias);
            break;
          case 2:
            output = dispatch_placeholder<scalar_t, 2>(
                input, packed_weight, scale, zero_point, scale_2, codebook,
                runtime_meta, bias);
            break;
          case 3:
            output = dispatch_placeholder<scalar_t, 3>(
                input, packed_weight, scale, zero_point, scale_2, codebook,
                runtime_meta, bias);
            break;
          case 4:
            output = dispatch_placeholder<scalar_t, 4>(
                input, packed_weight, scale, zero_point, scale_2, codebook,
                runtime_meta, bias);
            break;
          case 6:
            output = dispatch_placeholder<scalar_t, 6>(
                input, packed_weight, scale, zero_point, scale_2, codebook,
                runtime_meta, bias);
            break;
          case 8:
            output = dispatch_placeholder<scalar_t, 8>(
                input, packed_weight, scale, zero_point, scale_2, codebook,
                runtime_meta, bias);
            break;
          default:
            TORCH_CHECK(false, "Unsupported SINQ bit width: ", weight_bits);
        }
      });

  return output;
}

}  // namespace sinq
}  // namespace vllm

TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("sinq_linear_forward", &vllm::sinq::sinq_linear_forward);
}

TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CPU, m) {
  m.impl(
      "sinq_linear_forward",
      [](const at::Tensor&, const at::Tensor&,
         const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
         const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&,
         const c10::Dict<c10::IValue, c10::IValue>&,
         const c10::optional<at::Tensor>&) -> at::Tensor {
        TORCH_CHECK(false, "SINQ kernels are only implemented for CUDA builds.");
      });
}

