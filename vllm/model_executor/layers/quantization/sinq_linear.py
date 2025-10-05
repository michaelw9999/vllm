# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Linear method scaffolding for SINQ quantization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase

from .sinq import SinqConfig


logger = init_logger(__name__)


@dataclass
class _SinqWeights:
    """Container tracking tensors required by the SINQ kernel."""

    packed_weights: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None
    scale_2: Optional[torch.Tensor] = None
    codebook: Optional[torch.Tensor] = None


class SinqLinearMethod(LinearMethodBase):
    """Linear method stub that delegates to the fused SINQ CUDA kernel."""

    _STATE_ATTR = "_sinq_weights_state"

    def __init__(self, config: SinqConfig):
        self.config = config
        if not config.has_external_kernel:
            logger.warning(
                "SINQ CUDA sources were not located automatically. Place the "
                "active SINQ checkout next to the vLLM repository so the "
                "kernel can be built and linked."
            )

    @staticmethod
    def _ensure_state(layer: torch.nn.Module) -> _SinqWeights:
        state = getattr(layer, SinqLinearMethod._STATE_ATTR, None)
        if state is None:
            state = _SinqWeights()
            setattr(layer, SinqLinearMethod._STATE_ATTR, state)
        return state

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Prepare the layer to receive externally quantized SINQ weights."""

        self._ensure_state(layer)

        # The real weight loader is responsible for populating the state with
        # packed tensors. Until the kernel artifacts are linked in, we only
        # stash metadata on the module so downstream tooling can introspect it.
        layer.sinq_weight_bits = self.config.weight_bits
        layer.sinq_group_size = self.config.group_size
        layer.sinq_use_zero_point = self.config.use_zp
        layer.sinq_pack_factor = self.config.pack_factor
        layer.sinq_kernel_version = self.config.kernel_version

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weights = getattr(layer, self._STATE_ATTR, None)
        if not isinstance(weights, _SinqWeights) or weights.packed_weights is None:
            raise RuntimeError(
                "SINQ weights have not been loaded. Ensure the checkpoint "
                "provides pre-quantized tensors compatible with the SINQ kernel."
            )

        try:
            sinq_op = torch.ops._C.sinq_linear_forward
        except AttributeError as exc:  # pragma: no cover - depends on extension
            raise RuntimeError(
                "SINQ CUDA kernels are unavailable. Rebuild vLLM with the "
                "external SINQ sources to enable the fused WMMA path."
            ) from exc

        runtime_meta = {
            "weight_bits": int(self.config.weight_bits),
            "group_size": int(self.config.group_size),
            "use_zero_point": bool(self.config.use_zp),
            "pack_factor": int(self.config.pack_factor or 0),
            "kernel_version": self.config.kernel_version,
        }

        return sinq_op(
            x,
            weights.packed_weights,
            weights.scale,
            weights.zero_point,
            weights.scale_2,
            weights.codebook,
            runtime_meta,
            bias,
        )
