# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Placeholder implementation for SINQ linear method integration."""

from __future__ import annotations

import importlib
from typing import Optional, Protocol

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase

from .sinq import SinqConfig


logger = init_logger(__name__)


_RUNTIME_MODULE_CANDIDATES = (
    "sinq.torch",
    "sinq.runtime",
    "sinq",
)


class _RuntimeLinearAPI(Protocol):
    def create_linear_weights(
        self,
        layer: torch.nn.Module,
        config: SinqConfig,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        ...

    def linear_forward(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        config: SinqConfig,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


def _load_runtime_module() -> _RuntimeLinearAPI:
    """Import the runtime helper exposed by the external SINQ package."""

    for module_name in _RUNTIME_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        create_fn = getattr(module, "create_linear_weights", None)
        forward_fn = getattr(module, "linear_forward", None)
        if callable(create_fn) and callable(forward_fn):
            return module  # type: ignore[return-value]

    raise RuntimeError(
        "SINQ runtime integration requires an external Python package that "
        "exposes `create_linear_weights` and `linear_forward` helpers. "
        "Install the development SINQ package and ensure it is importable."
    )


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
        self._runtime_module: Optional[_RuntimeLinearAPI] = None

    def _get_runtime(self) -> _RuntimeLinearAPI:
        if self._runtime_module is None:
            self._runtime_module = _load_runtime_module()
        return self._runtime_module

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
        runtime = self._get_runtime()
        runtime.create_linear_weights(
            layer,
            self.config,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        runtime = self._get_runtime()
        return runtime.linear_forward(layer, x, self.config, bias=bias)
