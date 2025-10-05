# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Python-side SINQ quantization helpers."""

from __future__ import annotations

import math
from typing import Dict

import torch

from vllm.model_executor.layers.quantization.sinq import SinqConfig


def _compute_group_shape(weight: torch.Tensor, group_size: int) -> tuple[int, int, int]:
    if weight.ndim < 2:
        raise ValueError(
            "SINQ quantization expects a 2D weight tensor (out_features, in_features)."
        )
    out_features, in_features = weight.shape[-2], weight.shape[-1]
    if in_features % group_size != 0:
        raise ValueError(
            "Input feature dimension {} is not divisible by group_size {}".format(
                in_features, group_size
            )
        )
    groups_per_row = in_features // group_size
    rows = weight.numel() // in_features
    return rows, groups_per_row, group_size


def _pack_groups(quantized: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer groups into a torch.uint8 tensor.

    Args:
        quantized: Tensor with shape (..., group_size) containing values in
            ``[0, 2 ** bits)``.
        bits: Bit-width of the quantized values. Must divide ``LCM(bits, 8)``.

    Returns:
        Packed tensor with the same leading dimensions as ``quantized`` and the
        last dimension collapsed to bytes.
    """

    if bits not in {1, 2, 3, 4, 6, 8}:
        raise ValueError(f"Unsupported weight bit-width: {bits}")

    if quantized.numel() == 0:
        return torch.empty_like(quantized, dtype=torch.uint8)

    chunk_bits = math.lcm(8, bits)
    values_per_chunk = chunk_bits // bits
    bytes_per_chunk = chunk_bits // 8

    if quantized.shape[-1] % values_per_chunk != 0:
        raise ValueError(
            "Group size {} is incompatible with {}-bit packing ({} values/chunk)".format(
                quantized.shape[-1], bits, values_per_chunk
            )
        )

    quantized_i64 = quantized.to(torch.int64)
    prefix_shape = quantized_i64.shape[:-1]
    groups = quantized_i64.reshape(-1, quantized_i64.shape[-1] // values_per_chunk, values_per_chunk)

    shifts = torch.arange(
        values_per_chunk - 1,
        -1,
        -1,
        device=quantized.device,
        dtype=torch.int64,
    ) * bits
    chunks = torch.sum(groups << shifts, dim=-1)

    byte_shifts = torch.arange(
        bytes_per_chunk - 1,
        -1,
        -1,
        device=quantized.device,
        dtype=torch.int64,
    ) * 8
    packed = ((chunks.unsqueeze(-1) >> byte_shifts) & 0xFF).to(torch.uint8)

    output = packed.reshape(*prefix_shape, -1)
    return output.contiguous()


def quantize_weights(weight: torch.Tensor, config: SinqConfig) -> Dict[str, torch.Tensor]:
    """Quantize a floating point weight tensor using SINQ.

    The function computes group-wise scale and zero-point values and then packs
    the quantized integers so they can be consumed by the CUDA kernels.
    """

    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "Expected floating point weight tensor, but received dtype {}".format(
                weight.dtype
            )
        )

    weight_f32 = weight.to(torch.float32)
    rows, groups_per_row, group_size = _compute_group_shape(weight_f32, config.group_size)
    total_groups = rows * groups_per_row

    reshaped = weight_f32.reshape(rows, groups_per_row, group_size)
    w_min = reshaped.amin(dim=-1)
    w_max = reshaped.amax(dim=-1)
    levels = (1 << config.weight_bits) - 1

    ranges = w_max - w_min
    safe_ranges = torch.where(ranges > 0, ranges, torch.ones_like(ranges))
    scales = safe_ranges / max(levels, 1)

    if not torch.isfinite(scales).all():
        raise FloatingPointError("Encountered non-finite scale values during SINQ quantization")

    inv_scales = torch.where(scales > 0, 1.0 / scales, torch.zeros_like(scales))
    zero_points = torch.round(-w_min * inv_scales)
    zero_points = zero_points.clamp_(0, levels).to(torch.int32)

    quantized = torch.round(reshaped * inv_scales.unsqueeze(-1) + zero_points.unsqueeze(-1))
    quantized = quantized.clamp_(0, levels).to(torch.int32)

    packed = _pack_groups(quantized, config.weight_bits)

    qbytes_per_group = packed.shape[-1] // groups_per_row if groups_per_row > 0 else 0
    packed = packed.reshape(rows, groups_per_row * qbytes_per_group)

    return {
        "packed_weights": packed.contiguous(),
        "scales": scales.to(weight.dtype),
        "zero_points": zero_points,
        "group_size": torch.tensor(group_size, dtype=torch.int32),
        "weight_bits": torch.tensor(config.weight_bits, dtype=torch.int32),
        "groups_per_row": torch.tensor(groups_per_row, dtype=torch.int32),
        "total_groups": torch.tensor(total_groups, dtype=torch.int32),
    }
