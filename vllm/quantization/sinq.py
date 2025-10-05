# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration helpers for the external SINQ quantization runtime."""

from __future__ import annotations

import importlib
from typing import Dict, Protocol

import torch

from vllm.model_executor.layers.quantization.sinq import SinqConfig

_MODULE_CANDIDATES = (
    "sinq",
    "sinq_quantization",
)


class _ExternalQuantizeFn(Protocol):
    def __call__(self, weight: torch.Tensor, config: Dict[str, object]) -> Dict[str, torch.Tensor]:
        ...


def _load_external_quantizer() -> _ExternalQuantizeFn:
    """Load the quantization helper from the external SINQ package."""

    for module_name in _MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        quantize_fn = getattr(module, "quantize_weights", None)
        if callable(quantize_fn):
            return quantize_fn  # type: ignore[return-value]

    raise RuntimeError(
        "SINQ quantization support requires an external package exposing a "
        "`quantize_weights` function (e.g., the development SINQ repo). "
        "Install the package and ensure it is available on PYTHONPATH."
    )


def quantize_weights(weight: torch.Tensor, config: SinqConfig) -> Dict[str, torch.Tensor]:
    """Delegate quantization to the external SINQ runtime.

    The returned dictionary is treated as opaque metadata by vLLM and passed
    directly to :class:`~vllm.model_executor.layers.quantization.sinq_linear.SinqLinearMethod`.
    """

    quantize_fn = _load_external_quantizer()
    return quantize_fn(weight, config.to_external_config())
