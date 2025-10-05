#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace sinq {
namespace wmma {

// WMMA tile constants ---------------------------------------------------------
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kThreadsPerWarp = 32;
constexpr int kBlockRowTiles = 2;
constexpr int kBlockColTiles = 2;

static_assert(kWmmaM == 16 && kWmmaN == 16 && kWmmaK == 16,
              "This kernel is specialized for 16x16x16 WMMA tiles.");

// Enumerations ----------------------------------------------------------------
enum class BitPacking : int {
  k1bit_u8 = 0,
  k2bit_u8,
  k3bit_32,
  k4bit_u8,
  k5bit_32,
  k6bit_32,
  k8bit_u8,
};

enum class QuantMode : int {
  kUniform = 0,
  kNFCodebook,
};

enum class TilingMode : int {
  k1D = 0,
  k2D,
};

namespace detail {

template <typename T>
struct NativeType {
  using type = T;
  static constexpr bool kIsSupported = false;
  static constexpr int kVectorWidth = 1;
};

template <>
struct NativeType<half> {
  using type = half;
  static constexpr bool kIsSupported = true;
  static constexpr int kVectorWidth = 8;  // 8 half values == 16 bytes.
};

template <int Bits>
struct QuantTraits;

template <>
struct QuantTraits<1> {
  static constexpr int kBits = 1;
  static constexpr BitPacking kPacking = BitPacking::k1bit_u8;
  static constexpr int kValuesPerWord = 32;
};

template <>
struct QuantTraits<2> {
  static constexpr int kBits = 2;
  static constexpr BitPacking kPacking = BitPacking::k2bit_u8;
  static constexpr int kValuesPerWord = 16;
};

template <>
struct QuantTraits<3> {
  static constexpr int kBits = 3;
  static constexpr BitPacking kPacking = BitPacking::k3bit_32;
  static constexpr int kValuesPerWord = 10;
};

template <>
struct QuantTraits<4> {
  static constexpr int kBits = 4;
  static constexpr BitPacking kPacking = BitPacking::k4bit_u8;
  static constexpr int kValuesPerWord = 8;
};

template <>
struct QuantTraits<5> {
  static constexpr int kBits = 5;
  static constexpr BitPacking kPacking = BitPacking::k5bit_32;
  static constexpr int kValuesPerWord = 6;
};

template <>
struct QuantTraits<6> {
  static constexpr int kBits = 6;
  static constexpr BitPacking kPacking = BitPacking::k6bit_32;
  static constexpr int kValuesPerWord = 5;
};

template <>
struct QuantTraits<8> {
  static constexpr int kBits = 8;
  static constexpr BitPacking kPacking = BitPacking::k8bit_u8;
  static constexpr int kValuesPerWord = 4;
};

}  // namespace detail

struct KernelLaunchParams {
  const void* activation_ptr = nullptr;
  const void* packed_weights = nullptr;
  const void* scale_ptr = nullptr;
  const void* zero_ptr = nullptr;
  const void* scale2_ptr = nullptr;
  const void* codebook_ptr = nullptr;
  void* output_ptr = nullptr;

  int m = 0;
  int n = 0;
  int k = 0;
  int group_size = 0;
  int tile_dim = 0;
  int codebook_size = 0;
  int weight_bits = 0;

  BitPacking packing = BitPacking::k4bit_u8;
  QuantMode quant_mode = QuantMode::kUniform;
  TilingMode tiling_mode = TilingMode::k1D;

  bool has_dual_scale = false;
};

struct KernelExecutionConfig {
  dim3 grid;
  dim3 block;
  size_t shared_mem_bytes;
};

void launch_sinq_wmma_forward(const KernelLaunchParams& params,
                              const KernelExecutionConfig& config,
                              cudaStream_t stream);

}  // namespace wmma
}  // namespace sinq

