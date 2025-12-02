# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantization plugin for Wan2.2 native attention modules.

This module provides FP8 quantization support for WanSelfAttention and WanCrossAttention
classes used in the native Wan2.2 pipeline. It quantizes Q, K, V tensors before they
are passed to flash_attention.

The plugin is automatically loaded when Wan2.2 is installed and modelopt.torch.quantization
is imported.
"""

import logging
from contextlib import contextmanager
from functools import partial
from types import ModuleType
from typing import Callable

import torch

from modelopt.torch.quantization.nn import (
    QuantInputBase,
    QuantModule,
    QuantModuleRegistry,
    TensorQuantizer,
)

logger = logging.getLogger(__name__)

_original_flash_attention = None


def _get_wan_model_module():
    """Lazily import Wan model module and classes."""
    try:
        from wan.modules import model as wan_model
        return wan_model, wan_model.WanSelfAttention, wan_model.WanCrossAttention
    except ImportError:
        return None, None, None


@contextmanager
def replace_function(module: ModuleType, func_name: str, new_func: Callable):
    """Context manager to temporarily replace a function in a module."""
    original = getattr(module, func_name)
    setattr(module, func_name, new_func)
    try:
        yield
    finally:
        setattr(module, func_name, original)


def _quantized_flash_attention(
    attention_module: "QuantModule",
    *args,
    **kwargs,
) -> torch.Tensor:
    """Quantized flash attention wrapper that quantizes Q, K, V before attention.

    Handles both positional and keyword arguments for q, k, v:
    - WanSelfAttention: flash_attention(q=..., k=..., v=..., ...)  (keyword)
    - WanCrossAttention: flash_attention(q, k, v, ...)  (positional)
    """
    global _original_flash_attention

    if len(args) >= 3:
        q, k, v = args[0], args[1], args[2]
        remaining_args = args[3:]
    else:
        q = kwargs.pop('q')
        k = kwargs.pop('k')
        v = kwargs.pop('v')
        remaining_args = args

    if hasattr(attention_module, 'q_bmm_quantizer') and attention_module.q_bmm_quantizer.is_enabled:
        q = attention_module.q_bmm_quantizer(q)
    if hasattr(attention_module, 'k_bmm_quantizer') and attention_module.k_bmm_quantizer.is_enabled:
        k = attention_module.k_bmm_quantizer(k)
    if hasattr(attention_module, 'v_bmm_quantizer') and attention_module.v_bmm_quantizer.is_enabled:
        v = attention_module.v_bmm_quantizer(v)

    result = _original_flash_attention(q, k, v, *remaining_args, **kwargs)

    if hasattr(attention_module, 'bmm2_output_quantizer') and attention_module.bmm2_output_quantizer.is_enabled:
        result = attention_module.bmm2_output_quantizer(result)

    return result


class _QuantWanAttentionBase(QuantModule):
    """Base class for quantized Wan attention modules."""

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)

        # Disabled: Flash Attention is a fused kernel, cannot intercept internal softmax
        self.softmax_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.softmax_quantizer.disable()

        self.bmm2_output_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self._disable_fp8_mha = False

    def forward(self, *args, **kwargs):
        global _original_flash_attention

        wan_model, _, _ = _get_wan_model_module()
        if wan_model is None:
            return super().forward(*args, **kwargs)

        if _original_flash_attention is None:
            _original_flash_attention = wan_model.flash_attention

        quantized_fn = partial(_quantized_flash_attention, self)

        with replace_function(wan_model, 'flash_attention', quantized_fn):
            return super().forward(*args, **kwargs)


class _QuantWanSelfAttention(_QuantWanAttentionBase):
    """Quantized WanSelfAttention module."""
    pass


class _QuantWanCrossAttention(_QuantWanAttentionBase):
    """Quantized WanCrossAttention module."""
    pass


def register_wan_attention_quantization():
    """Register Wan attention modules with the quantization registry.

    Returns:
        bool: True if registration successful, False otherwise
    """
    _, WanSelfAttention, WanCrossAttention = _get_wan_model_module()

    if WanSelfAttention is None or WanCrossAttention is None:
        logger.warning(
            "Could not import Wan attention modules. "
            "Make sure Wan2.2 is in your Python path."
        )
        return False

    try:
        QuantModuleRegistry.register({WanSelfAttention: "WanSelfAttention"})(_QuantWanSelfAttention)
        logger.info("Registered WanSelfAttention for quantization")

        QuantModuleRegistry.register({WanCrossAttention: "WanCrossAttention"})(_QuantWanCrossAttention)
        logger.info("Registered WanCrossAttention for quantization")

        return True

    except Exception as e:
        logger.error(f"Failed to register Wan attention quantization: {e}")
        return False


def check_wan_attention_quantization(model: torch.nn.Module) -> dict:
    """Check if Wan attention modules have quantizers properly set up."""
    _, WanSelfAttention, WanCrossAttention = _get_wan_model_module()
    if WanSelfAttention is None:
        return {"error": "Wan modules not importable"}

    results = {
        "self_attention_modules": 0,
        "cross_attention_modules": 0,
        "modules_with_quantizers": 0,
        "quantizers_enabled": {
            "q_bmm": 0,
            "k_bmm": 0,
            "v_bmm": 0,
            "softmax": 0,
            "bmm2_output": 0,
        },
    }

    for name, module in model.named_modules():
        if isinstance(module, WanSelfAttention):
            results["self_attention_modules"] += 1
            if hasattr(module, 'q_bmm_quantizer'):
                results["modules_with_quantizers"] += 1
                if module.q_bmm_quantizer.is_enabled:
                    results["quantizers_enabled"]["q_bmm"] += 1
                if module.k_bmm_quantizer.is_enabled:
                    results["quantizers_enabled"]["k_bmm"] += 1
                if module.v_bmm_quantizer.is_enabled:
                    results["quantizers_enabled"]["v_bmm"] += 1
                if hasattr(module, 'softmax_quantizer') and module.softmax_quantizer.is_enabled:
                    results["quantizers_enabled"]["softmax"] += 1
                if hasattr(module, 'bmm2_output_quantizer') and module.bmm2_output_quantizer.is_enabled:
                    results["quantizers_enabled"]["bmm2_output"] += 1

        elif isinstance(module, WanCrossAttention):
            results["cross_attention_modules"] += 1
            if hasattr(module, 'q_bmm_quantizer'):
                results["modules_with_quantizers"] += 1
                if module.q_bmm_quantizer.is_enabled:
                    results["quantizers_enabled"]["q_bmm"] += 1
                if module.k_bmm_quantizer.is_enabled:
                    results["quantizers_enabled"]["k_bmm"] += 1
                if module.v_bmm_quantizer.is_enabled:
                    results["quantizers_enabled"]["v_bmm"] += 1

    return results


def disable_wan_sdpa_quantization(model: torch.nn.Module):
    """Disable SDPA quantization for all Wan attention modules."""
    _, WanSelfAttention, WanCrossAttention = _get_wan_model_module()
    if WanSelfAttention is None:
        return

    for name, module in model.named_modules():
        if isinstance(module, (WanSelfAttention, WanCrossAttention)):
            if hasattr(module, 'q_bmm_quantizer'):
                module.q_bmm_quantizer.disable()
            if hasattr(module, 'k_bmm_quantizer'):
                module.k_bmm_quantizer.disable()
            if hasattr(module, 'v_bmm_quantizer'):
                module.v_bmm_quantizer.disable()
            if hasattr(module, 'softmax_quantizer'):
                module.softmax_quantizer.disable()
            if hasattr(module, 'bmm2_output_quantizer'):
                module.bmm2_output_quantizer.disable()


_registration_attempted = False


def ensure_registration():
    """Ensure Wan attention quantization is registered (call once)."""
    global _registration_attempted
    if not _registration_attempted:
        _registration_attempted = True
        register_wan_attention_quantization()


ensure_registration()
