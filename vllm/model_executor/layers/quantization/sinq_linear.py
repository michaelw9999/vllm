# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Placeholder implementation for SINQ linear method integration."""

from __future__ import annotations

from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase

from .sinq import SinqConfig


logger = init_logger(__name__)


class SinqLinearMethod(LinearMethodBase):
    """Linear method placeholder for SINQ quantization."""

    def __init__(self, config: SinqConfig):
        self.config = config
        if not config.has_external_kernel:
            logger.warning(
                "SINQ CUDA sources were not located automatically. The "
                "external repository is expected at a sibling 'SINQ' "
                "directory (e.g. /workspace/vllm/SINQ) so the kernel can "
                "be built."
            )

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
        raise NotImplementedError("SINQ linear method is not yet implemented")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SINQ linear method is not yet implemented")
