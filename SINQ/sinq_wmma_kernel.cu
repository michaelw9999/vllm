// Copyright (c) 2025
//
// Skeleton for the SINQ fused WMMA CUDA kernel. The implementation will follow
// the design outlined in the planning notes: support SINQ's packed weight
// formats (uniform + NF3/NF4), variable group sizes, and optional dual-scale
// metadata.

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <type_traits>

#include "sinq_wmma_kernel.h"

namespace sinq {
namespace wmma {

namespace {

using namespace ::nvcuda;

// Shared-memory staging helpers ------------------------------------------------

struct SharedWorkspace {
  half* scale = nullptr;
  half* zero = nullptr;
  half* scale2 = nullptr;
  float* codebook = nullptr;
  half* a_tile = nullptr;
  half* tile_b = nullptr;
};

__device__ SharedWorkspace make_shared_workspace(int groups_per_block,
                                                 int tile_k_dim,
                                                 int tile_cols,
                                                 int codebook_size) {
  extern __shared__ char shared_raw[];
  SharedWorkspace ws;

  size_t offset = 0;
  auto align = [&](size_t alignment) {
    offset = (offset + alignment - 1) & ~(alignment - 1);
  };

  align(alignof(half));
  ws.scale = reinterpret_cast<half*>(shared_raw + offset);
  offset += sizeof(half) * groups_per_block;

  align(alignof(half));
  ws.zero = reinterpret_cast<half*>(shared_raw + offset);
  offset += sizeof(half) * groups_per_block;

  align(alignof(half));
  ws.scale2 = reinterpret_cast<half*>(shared_raw + offset);
  offset += sizeof(half) * tile_k_dim;

  if (codebook_size > 0) {
    align(alignof(float));
    ws.codebook = reinterpret_cast<float*>(shared_raw + offset);
    offset += sizeof(float) * codebook_size;
  } else {
    ws.codebook = nullptr;
  }

  const int tile_rows = kBlockRowTiles * kWmmaM;
  align(alignof(half));
  ws.a_tile = reinterpret_cast<half*>(shared_raw + offset);
  offset += sizeof(half) * tile_rows * tile_k_dim;

  align(alignof(half));
  ws.tile_b = reinterpret_cast<half*>(shared_raw + offset);
  offset += sizeof(half) * tile_cols * tile_k_dim;

  return ws;
}

__device__ inline void load_group_metadata(const KernelLaunchParams& params,
                                           int block_group_start,
                                           int groups_per_block,
                                           int groups_per_row,
                                           SharedWorkspace shared) {
  const half zero_half = __float2half(0.0f);

  const half* scale_global = reinterpret_cast<const half*>(params.scale_ptr);
  const half* zero_global = reinterpret_cast<const half*>(params.zero_ptr);

  for (int idx = threadIdx.x; idx < groups_per_block; idx += blockDim.x) {
    const int global_idx = block_group_start + idx;
    shared.scale[idx] = zero_half;
    shared.zero[idx] = zero_half;

    if (global_idx < 0) {
      continue;
    }

    int metadata_idx = -1;

    if (params.tiling_mode == TilingMode::k1D) {
      const int total_groups = params.n * groups_per_row;
      if (global_idx < total_groups) {
        metadata_idx = global_idx;
      }
    } else if (params.tiling_mode == TilingMode::k2D && params.tile_dim > 0) {
      const int tile_dim = params.tile_dim;
      const int groups_per_matrix_row = params.k / tile_dim;
      const int num_col_tiles = params.n / tile_dim;
      const int total_groups = (params.n * params.k) / tile_dim;

      if (global_idx < total_groups && groups_per_matrix_row > 0 && num_col_tiles > 0) {
        const int matrix_row = global_idx / groups_per_matrix_row;
        const int group_in_row = global_idx % groups_per_matrix_row;

        const int tile_row_idx = matrix_row / tile_dim;
        const int tile_col_idx = group_in_row;

        const int linear_tile_idx = tile_row_idx * num_col_tiles + tile_col_idx;
        const int row_in_tile = matrix_row % tile_dim;
        metadata_idx = linear_tile_idx * tile_dim + row_in_tile;
      }
    }

    if (metadata_idx >= 0) {
      if (scale_global != nullptr) {
        shared.scale[idx] = scale_global[metadata_idx];
      }
      if (zero_global != nullptr) {
        shared.zero[idx] = zero_global[metadata_idx];
      }
    }
  }
  __syncthreads();
}

__device__ inline int decode_weight_code(const uint8_t* packed, int64_t element_idx,
                                         BitPacking packing) {
  switch (packing) {
    case BitPacking::k8bit_u8: {
      return static_cast<int>(packed[static_cast<size_t>(element_idx)]);
    }
    case BitPacking::k4bit_u8: {
      const int64_t byte_idx = element_idx >> 1;
      const uint8_t byte = packed[static_cast<size_t>(byte_idx)];
      if ((element_idx & 1) == 0) {
        return (byte >> 4) & 0xF;
      }
      return byte & 0xF;
    }
    case BitPacking::k2bit_u8: {
      const int64_t byte_idx = element_idx >> 2;
      const uint8_t byte = packed[static_cast<size_t>(byte_idx)];
      const int lane = static_cast<int>(element_idx & 3);
      const int shift = 6 - (lane * 2);
      return (byte >> shift) & 0x3;
    }
    case BitPacking::k1bit_u8: {
      const int64_t byte_idx = element_idx >> 3;
      const uint8_t byte = packed[static_cast<size_t>(byte_idx)];
      const int bit = static_cast<int>(7 - (element_idx & 7));
      return (byte >> bit) & 0x1;
    }
    case BitPacking::k3bit_32: {
      const uint32_t* words = reinterpret_cast<const uint32_t*>(packed);
      constexpr uint64_t kDivBy10Magic = 0xCCCCCCCDull;
      const uint64_t mul = static_cast<uint64_t>(element_idx) * kDivBy10Magic;
      const int64_t word_idx = static_cast<int64_t>(mul >> 35);
      const int lane = static_cast<int>(element_idx - word_idx * 10);
      const uint32_t word = words[static_cast<size_t>(word_idx)];
      const int shift = (lane < 9) ? (27 - 3 * lane) : 0;
      return (word >> shift) & 0x7;
    }
    case BitPacking::k5bit_32: {
      const uint32_t* words = reinterpret_cast<const uint32_t*>(packed);
      // Correct magic multiplier and shift for unsigned division by 6.
      constexpr uint64_t kDivBy6Magic = 0x2AAAAAABull;
      const uint64_t mul = static_cast<uint64_t>(element_idx) * kDivBy6Magic;
      const int64_t word_idx = static_cast<int64_t>(mul >> 33);
      const int lane = static_cast<int>(element_idx - word_idx * 6);
      const uint32_t word = words[static_cast<size_t>(word_idx)];
      const int shift = 25 - 5 * lane;
      return (word >> shift) & 0x1F;
    }
    case BitPacking::k6bit_32: {
      const uint32_t* words = reinterpret_cast<const uint32_t*>(packed);
      constexpr uint64_t kDivBy5Magic = 0xCCCCCCCDull;
      const uint64_t mul = static_cast<uint64_t>(element_idx) * kDivBy5Magic;
      const int64_t word_idx = static_cast<int64_t>(mul >> 34);
      const int lane = static_cast<int>(element_idx - word_idx * 5);
      const uint32_t word = words[static_cast<size_t>(word_idx)];
      const int shift = 24 - 6 * lane;
      return (word >> shift) & 0x3F;
    }
    default:
      return 0;
  }
}

template <typename T>
struct FragmentTraits;

template <>
struct FragmentTraits<half> {
  using A = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major>;
  using B = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major>;
  using C = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half>;
};

}  // namespace

__global__ void sinq_wmma_fwd_kernel(KernelLaunchParams params) {
  const int block_tile_m = blockIdx.y;
  const int block_tile_n = blockIdx.x;

  const int tile_cols = kBlockColTiles * kWmmaN;
  const int tile_k_dim = kWmmaK;

  const int groups_per_row = (params.k + params.group_size - 1) / params.group_size;
  const int groups_per_block = tile_cols * groups_per_row;
  const int block_group_start = (block_tile_n * tile_cols) * groups_per_row;

  SharedWorkspace shared = make_shared_workspace(groups_per_block, tile_k_dim, tile_cols, params.codebook_size);

  const float* codebook_global = reinterpret_cast<const float*>(params.codebook_ptr);
  const bool use_nf = (params.quant_mode == QuantMode::kNFCodebook) &&
                      (codebook_global != nullptr) &&
                      (params.codebook_size > 0);

  if (use_nf) {
    for (int idx = threadIdx.x; idx < params.codebook_size; idx += blockDim.x) {
      shared.codebook[idx] = codebook_global[idx];
    }
  }
  __syncthreads();

  load_group_metadata(params, block_group_start, groups_per_block, groups_per_row, shared);

  typename FragmentTraits<half>::C c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane_id = threadIdx.x % kThreadsPerWarp;

  const int warp_tile_m = warp_id % kBlockRowTiles;
  const int warp_tile_n = warp_id / kBlockRowTiles;

  const int c_row = (block_tile_m * kBlockRowTiles + warp_tile_m) * kWmmaM;
  const int c_col = (block_tile_n * kBlockColTiles + warp_tile_n) * kWmmaN;

  typename FragmentTraits<half>::A a_frag;
  typename FragmentTraits<half>::B b_frag;
  wmma::fill_fragment(a_frag, 0.0f);
  wmma::fill_fragment(b_frag, 0.0f);

  const half* activation_base = reinterpret_cast<const half*>(params.activation_ptr);
  const uint8_t* packed_base = reinterpret_cast<const uint8_t*>(params.packed_weights);
  const half* scale2_global = reinterpret_cast<const half*>(params.scale2_ptr);
  const bool has_group_scale = (params.scale_ptr != nullptr);
  const bool has_zero = (params.zero_ptr != nullptr);
  const bool has_scale2 = params.has_dual_scale && (scale2_global != nullptr);

  const int tile_rows = kBlockRowTiles * kWmmaM;
  const int steady_state_k_iterations = params.k / tile_k_dim;
  const bool has_k_tail = (steady_state_k_iterations * tile_k_dim) != params.k;
  const int64_t total_elements = static_cast<int64_t>(params.n) * params.k;
  const int64_t total_packed_bytes = [&]() -> int64_t {
    switch (params.packing) {
      case BitPacking::k8bit_u8:
        return total_elements;
      case BitPacking::k4bit_u8:
        return (total_elements + 1) >> 1;
      case BitPacking::k2bit_u8:
        return (total_elements + 3) >> 2;
      case BitPacking::k1bit_u8:
        return (total_elements + 7) >> 3;
      case BitPacking::k3bit_32: {
        const int64_t words = (total_elements + 9) / 10;
        return words * static_cast<int64_t>(sizeof(uint32_t));
      }
      case BitPacking::k5bit_32: {
        const int64_t words = (total_elements + 5) / 6;
        return words * static_cast<int64_t>(sizeof(uint32_t));
      }
      case BitPacking::k6bit_32: {
        const int64_t words = (total_elements + 4) / 5;
        return words * static_cast<int64_t>(sizeof(uint32_t));
      }
      default:
        return total_elements;
    }
  }();
  const int total_tile_elems = tile_cols * tile_k_dim;
  const int row_base = block_tile_n * tile_cols;

  auto stage_scale2 = [&](int tile_k, bool tail) {
    if (!has_scale2) {
      return;
    }
    const int k_base = tile_k * tile_k_dim;
    for (int k_local = threadIdx.x; k_local < tile_k_dim; k_local += blockDim.x) {
      const int global_col = k_base + k_local;
      if (!tail || global_col < params.k) {
        shared.scale2[k_local] = scale2_global[global_col];
      } else {
        shared.scale2[k_local] = __float2half(1.0f);
      }
    }
  };

  auto stage_weights = [&](int tile_k, bool tail) {
    const BitPacking packing = params.packing;
    const bool byte_packing = (packing == BitPacking::k8bit_u8 ||
                               packing == BitPacking::k4bit_u8 ||
                               packing == BitPacking::k2bit_u8 ||
                               packing == BitPacking::k1bit_u8);
    const int word_shift = (packing == BitPacking::k8bit_u8)
                               ? 2
                               : (packing == BitPacking::k4bit_u8)
                                     ? 3
                                     : (packing == BitPacking::k2bit_u8)
                                           ? 4
                                           : (packing == BitPacking::k1bit_u8) ? 5 : 0;
    const int lane_mask = byte_packing ? ((1 << word_shift) - 1) : 0;
    const uint32_t* packed_words32 = reinterpret_cast<const uint32_t*>(packed_base);
    int64_t cached_word_idx = -1;
    uint32_t cached_word = 0;
    int cached_valid_bytes = 0;

    for (int elem_idx_in_tile = threadIdx.x; elem_idx_in_tile < total_tile_elems; elem_idx_in_tile += blockDim.x) {
      const int n_local = elem_idx_in_tile / tile_k_dim;
      const int k_local = elem_idx_in_tile % tile_k_dim;

      const int global_row = row_base + n_local;
      const int global_col = tile_k * tile_k_dim + k_local;

      half stored_val = __float2half(0.0f);
      if (global_row < params.n && (!tail || global_col < params.k)) {
        const int64_t flat_index = static_cast<int64_t>(global_row) * params.k + global_col;

        int code = 0;
        bool decoded = false;

        if (byte_packing) {
          const int64_t word_idx = flat_index >> word_shift;
          const int lane = static_cast<int>(flat_index & lane_mask);
          const int64_t byte_offset = word_idx << 2;
          if (word_idx != cached_word_idx || cached_valid_bytes == 0) {
            if (byte_offset + static_cast<int64_t>(sizeof(uint32_t)) <= total_packed_bytes) {
              cached_word = packed_words32[word_idx];
              cached_word_idx = word_idx;
              cached_valid_bytes = sizeof(uint32_t);
            } else {
              cached_word_idx = -1;
              cached_valid_bytes = 0;
            }
          }
          if (cached_valid_bytes == sizeof(uint32_t) && word_idx == cached_word_idx) {
            const uint32_t word = cached_word;
            switch (packing) {
              case BitPacking::k8bit_u8: {
                const int shift = lane * 8;
                code = static_cast<int>((word >> shift) & 0xFFu);
                decoded = true;
                break;
              }
              case BitPacking::k4bit_u8: {
                const int byte_lane = lane >> 1;
                const int shift = byte_lane * 8;
                const uint8_t byte = static_cast<uint8_t>((word >> shift) & 0xFFu);
                code = (lane & 1) ? (byte & 0xF) : ((byte >> 4) & 0xF);
                decoded = true;
                break;
              }
              case BitPacking::k2bit_u8: {
                const int byte_lane = lane >> 2;
                const int shift = byte_lane * 8;
                const uint8_t byte = static_cast<uint8_t>((word >> shift) & 0xFFu);
                const int lane_in_byte = lane & 3;
                const int bit_shift = 6 - lane_in_byte * 2;
                code = (byte >> bit_shift) & 0x3;
                decoded = true;
                break;
              }
              case BitPacking::k1bit_u8: {
                const int byte_lane = lane >> 3;
                const int shift = byte_lane * 8;
                const uint8_t byte = static_cast<uint8_t>((word >> shift) & 0xFFu);
                const int bit = 7 - (lane & 7);
                code = (byte >> bit) & 0x1;
                decoded = true;
                break;
              }
              default:
                break;
            }
          }
        } else if (packing == BitPacking::k3bit_32 ||
                   packing == BitPacking::k5bit_32 ||
                   packing == BitPacking::k6bit_32) {
          int lane = 0;
          int64_t word_idx = 0;
          if (packing == BitPacking::k3bit_32) {
            constexpr uint64_t kDivBy10Magic = 0xCCCCCCCDull;
            const uint64_t mul = static_cast<uint64_t>(flat_index) * kDivBy10Magic;
            word_idx = static_cast<int64_t>(mul >> 35);
            lane = static_cast<int>(flat_index - word_idx * 10);
          } else if (packing == BitPacking::k5bit_32) {
            constexpr uint64_t kDivBy6Magic = 0x2AAAAAABull;
            const uint64_t mul = static_cast<uint64_t>(flat_index) * kDivBy6Magic;
            word_idx = static_cast<int64_t>(mul >> 33);
            lane = static_cast<int>(flat_index - word_idx * 6);
          } else {
            constexpr uint64_t kDivBy5Magic = 0xCCCCCCCDull;
            const uint64_t mul = static_cast<uint64_t>(flat_index) * kDivBy5Magic;
            word_idx = static_cast<int64_t>(mul >> 34);
            lane = static_cast<int>(flat_index - word_idx * 5);
          }

          if (word_idx != cached_word_idx) {
            cached_word = packed_words32[word_idx];
            cached_word_idx = word_idx;
          }
          const uint32_t word = cached_word;
          if (packing == BitPacking::k3bit_32) {
            const int shift = (lane < 9) ? (27 - 3 * lane) : 0;
            code = static_cast<int>((word >> shift) & 0x7u);
          } else if (packing == BitPacking::k5bit_32) {
            const int shift = 25 - 5 * lane;
            code = static_cast<int>((word >> shift) & 0x1Fu);
          } else {
            const int shift = 24 - 6 * lane;
            code = static_cast<int>((word >> shift) & 0x3Fu);
          }
          decoded = true;
        }

        if (!decoded) {
          code = decode_weight_code(packed_base, flat_index, packing);
        }

        const int group_index = static_cast<int>(flat_index / params.group_size);
        const int group_local = group_index - block_group_start;

        float scale_val = 1.0f;
        float zero_val = 0.0f;
        if (group_local >= 0 && group_local < groups_per_block) {
          if (has_group_scale) {
            scale_val = __half2float(shared.scale[group_local]);
          }
          if (has_zero) {
            zero_val = __half2float(shared.zero[group_local]);
          }
        }

        const float column_scale = has_scale2 ? __half2float(shared.scale2[k_local]) : 1.0f;

        float decoded_val = static_cast<float>(code);
        if (use_nf && params.codebook_size > 0) {
          int cb_idx = code;
          if (cb_idx < 0) cb_idx = 0;
          if (cb_idx >= params.codebook_size) cb_idx = params.codebook_size - 1;
          decoded_val = shared.codebook[cb_idx];
        }

        const float value = (decoded_val - zero_val) * scale_val * column_scale;
        stored_val = __float2half(value);
      }

      shared.tile_b[k_local + n_local * tile_k_dim] = stored_val;
    }
  };

  // Steady-state loop over full K tiles
  for (int tile_k = 0; tile_k < steady_state_k_iterations; ++tile_k) {
    stage_scale2(tile_k, false);
    __syncthreads();

    stage_weights(tile_k, false);
    __syncthreads();

    const half* tile_b_ptr = shared.tile_b + warp_tile_n * kWmmaN * tile_k_dim;

    const bool valid_a = (activation_base != nullptr) && (c_row + kWmmaM <= params.m);
    if (valid_a) {
      const half* a_tile_ptr = activation_base + c_row * params.k + tile_k * tile_k_dim;
      wmma::load_matrix_sync(a_frag, a_tile_ptr, params.k);
    } else {
      wmma::fill_fragment(a_frag, 0.0f);
    }

    wmma::load_matrix_sync(b_frag, tile_b_ptr, tile_k_dim);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Epilogue for trailing partial K tile
  if (has_k_tail) {
    const int tile_k = steady_state_k_iterations;
    stage_scale2(tile_k, true);
    __syncthreads();

    stage_weights(tile_k, true);
    __syncthreads();

    const half* tile_b_ptr = shared.tile_b + warp_tile_n * kWmmaN * tile_k_dim;

    half* a_tile_shared = shared.a_tile;
    for (int idx = threadIdx.x; idx < tile_rows * tile_k_dim; idx += blockDim.x) {
      const int local_row = idx / tile_k_dim;
      const int local_col = idx % tile_k_dim;
      const int global_row = c_row + local_row;
      const int global_col = tile_k * tile_k_dim + local_col;

      half val = __float2half(0.0f);
      if (activation_base != nullptr && global_row < params.m && global_col < params.k) {
        val = activation_base[global_row * params.k + global_col];
      }
      a_tile_shared[idx] = val;
    }
    __syncthreads();

    const half* a_tile_ptr = a_tile_shared + warp_tile_m * kWmmaM * tile_k_dim;
    wmma::load_matrix_sync(a_frag, a_tile_ptr, tile_k_dim);

    wmma::load_matrix_sync(b_frag, tile_b_ptr, tile_k_dim);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  if (params.output_ptr != nullptr &&
      c_row + kWmmaM <= params.m &&
      c_col + kWmmaN <= params.n) {
    half* output_base = reinterpret_cast<half*>(params.output_ptr);
    half* c_tile_ptr = output_base + c_row * params.n + c_col;
    wmma::store_matrix_sync(c_tile_ptr, c_frag, params.n, wmma::mem_row_major);
  } else if (params.output_ptr != nullptr) {
    half* output_base = reinterpret_cast<half*>(params.output_ptr);
    alignas(16) half c_store[kWmmaM * kWmmaN];
    wmma::store_matrix_sync(c_store, c_frag, kWmmaN, wmma::mem_row_major);

    for (int idx = lane_id; idx < kWmmaM * kWmmaN; idx += kThreadsPerWarp) {
      const int local_row = idx / kWmmaN;
      const int local_col = idx % kWmmaN;
      const int global_row = c_row + local_row;
      const int global_col = c_col + local_col;
      if (global_row < params.m && global_col < params.n) {
        output_base[global_row * params.n + global_col] = c_store[idx];
      }
    }
  }
}

void launch_sinq_wmma_forward(const KernelLaunchParams& params,
                              const KernelExecutionConfig& config,
                              cudaStream_t stream) {
  sinq_wmma_fwd_kernel<<<config.grid, config.block, config.shared_mem_bytes, stream>>>(params);
}

}  // namespace wmma
}  // namespace sinq
