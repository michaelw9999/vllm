# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration definition for SINQ quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)


_SUPPORTED_BITS = (1, 2, 3, 4, 6, 8)


@dataclass
class SinqConfig(QuantizationConfig):
    """Configuration for SINQ quantization.

    The configuration stores the parameters that are required to quantize the
    weights with the SINQ scheme. The CUDA kernel implementation expects the
    Python configuration to provide consistent metadata, so all validation is
    performed up-front.
    """

    weight_bits: int = 4
    """Bit width used to represent each quantized weight value."""

    group_size: int = 128
    """Number of consecutive weights that share scale and zero-point."""

    pack_factor: Optional[int] = None
    """Optional override for how many groups are fused during packing."""

    use_zp: bool = True
    """Whether zero-points are used in the quantization scheme."""

    kernel_version: str = "auto"
    """Identifier for the kernel variant that should be selected at runtime."""

    external_kernel_dir: Optional[Path] = field(init=False, default=None, repr=False)
    """Resolved path to the out-of-tree SINQ CUDA sources, when present."""

    def __post_init__(self) -> None:
        super().__init__()
        if self.weight_bits not in _SUPPORTED_BITS:
            raise ValueError(
                "SINQ only supports weight bit-widths {} but received {}".format(
                    _SUPPORTED_BITS, self.weight_bits
                )
            )
        if self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if self.pack_factor is not None and self.pack_factor <= 0:
            raise ValueError("pack_factor, when provided, must be positive")

        self.external_kernel_dir = self._discover_external_kernel_dir()

    # QuantizationConfig API -------------------------------------------------
    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "sinq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The fused SINQ kernels rely on Tensor Cores which are widely
        # available starting from compute capability 80 (Ampere) and newer.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        # SINQ checkpoints are configured via CLI/overrides. There is no
        # canonical JSON configuration file to load from disk yet.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SinqConfig":
        weight_bits = cls.get_from_keys(config, ["bits", "weight_bits"])
        group_size = cls.get_from_keys(config, ["group_size", "block_size"])
        pack_factor = cls.get_from_keys_or(config, ["pack_factor"], None)
        use_zp = cls.get_from_keys_or(config, ["use_zp", "use_zero_point"], True)
        kernel_version = cls.get_from_keys_or(
            config, ["kernel_version", "kernel"], "auto"
        )
        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            pack_factor=pack_factor,
            use_zp=use_zp,
            kernel_version=kernel_version,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            from vllm.model_executor.layers.quantization.sinq_linear import (
                SinqLinearMethod,
            )

            return SinqLinearMethod(self)
        return None

    @staticmethod
    def _discover_external_kernel_dir() -> Optional[Path]:
        """Detect the optional external SINQ kernel checkout.

        Several internal development workflows keep the CUDA implementation in a
        sibling ``SINQ`` repository (e.g. ``/workspace/vllm/SINQ``). When that
        directory is available we surface the resolved path so the build system
        or runtime can consume the out-of-tree sources without manual tweaks.
        """

        for parent in Path(__file__).resolve().parents:
            candidate = parent / "SINQ"
            if candidate.is_dir():
                return candidate
        return None

    @property
    def has_external_kernel(self) -> bool:
        """Return ``True`` when the optional CUDA sources are discoverable."""

        return self.external_kernel_dir is not None


__all__ = ["SinqConfig"]
