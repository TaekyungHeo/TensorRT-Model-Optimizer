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

"""
Quantization script for Wan2.2 using native pipeline.

This script quantizes Wan2.2 text-to-video models using the native Wan pipeline
instead of the diffusers pipeline, supporting both low_noise_model and high_noise_model.

Supported formats:
    - fp8: FP8 quantization (recommended for Ada/Hopper GPUs)
    - fp4: NVFP4 quantization (requires Blackwell GPUs, best compression)
    - int8: INT8 quantization

Usage:
    # FP8 quantization
    python quantize_wan_native.py \
        --ckpt-dir /path/to/wan/checkpoint \
        --format fp8 \
        --calib-size 32 \
        --batch-size 1 \
        --quantized-ckpt-save-path ./quantized_wan/

    # NVFP4 quantization (4x compression)
    python quantize_wan_native.py \
        --ckpt-dir /path/to/wan/checkpoint \
        --format fp4 \
        --calib-size 64 \
        --quantized-ckpt-save-path ./quantized_wan_fp4/

    # NVFP4 with SVDQuant (better quality)
    python quantize_wan_native.py \
        --ckpt-dir /path/to/wan/checkpoint \
        --format fp4 \
        --quant-algo svdquant \
        --lowrank 32 \
        --calib-size 64 \
        --quantized-ckpt-save-path ./quantized_wan_fp4_svd/

Example with custom prompts:
    python quantize_wan_native.py \
        --ckpt-dir /path/to/wan/checkpoint \
        --format fp4 \
        --prompts-file ./calib_prompts.txt \
        --quantized-ckpt-save-path ./quantized_wan/
"""

import argparse
import gc
import logging
import re
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add Wan2.2 to path if needed
import os
wan_path = os.path.join(os.path.dirname(__file__), "../../../Wan2.2")
if os.path.exists(wan_path):
    sys.path.insert(0, wan_path)

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

try:
    from modelopt.torch.quantization.plugins import wan as wan_attention_quant
    WAN_ATTENTION_QUANT_AVAILABLE = True
except ImportError:
    wan_attention_quant = None
    WAN_ATTENTION_QUANT_AVAILABLE = False

from config import (
    FP8_DEFAULT_CONFIG,
    INT8_DEFAULT_CONFIG,
    NVFP4_DEFAULT_CONFIG,
    NVFP4_FP8_MHA_CONFIG,
    set_quant_config_attr,
)


class QuantFormat(str, Enum):
    """Supported quantization formats."""
    INT8 = "int8"
    FP8 = "fp8"
    FP4 = "fp4"


class QuantAlgo(str, Enum):
    """Supported quantization algorithms."""
    MAX = "max"
    SMOOTHQUANT = "smoothquant"
    SVDQUANT = "svdquant"


class DataType(str, Enum):
    """Supported data types for model loading."""
    HALF = "Half"
    BFLOAT16 = "BFloat16"
    FLOAT = "Float"

    @property
    def torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "Half": torch.float16,
            "BFloat16": torch.bfloat16,
            "Float": torch.float32,
        }
        return dtype_map[self.value]


@dataclass
class WanQuantConfig:
    """Configuration for Wan model quantization."""
    ckpt_dir: str
    task: str = "t2v-A14B"
    format: QuantFormat = QuantFormat.FP8
    algo: QuantAlgo = QuantAlgo.MAX
    model_dtype: DataType = DataType.BFLOAT16
    trt_high_precision_dtype: DataType = DataType.BFLOAT16
    alpha: float = 1.0
    lowrank: int = 32
    batch_size: int = 1
    calib_size: int = 32
    n_steps: int = 30
    size: str = "1280*720"
    frame_num: int = 81
    prompts_file: Path | None = None
    quantized_ckpt_save_path: Path | None = None
    restore_from: Path | None = None
    device_id: int = 0
    offload_model: bool = True
    compress: bool = False
    quantize_low_noise: bool = True
    quantize_high_noise: bool = True
    quantize_sdpa: bool = True
    layer_range: tuple[int, int] | None = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.addHandler(console_handler)

    return logger


def filter_func_wan_native(name: str) -> bool:
    """Filter function for Wan native models.

    Disables quantization for embedding layers that are sensitive to quantization.
    """
    pattern = re.compile(r".*(patch_embedding|text_embedding|time_embedding|time_projection).*")
    return pattern.match(name) is not None


def create_quantization_filter(
    layer_range: tuple[int, int] | None = None,
):
    """Create a filter function that disables quantization based on layer range.

    Args:
        layer_range: Tuple of (start, end) layer indices. Layers in [start, end) are quantized.
                    If None, all layers are quantized.

    Returns:
        Filter function that returns True for layers to DISABLE quantization.
    """
    embedding_pattern = re.compile(
        r".*(patch_embedding|text_embedding|time_embedding|time_projection).*"
    )
    block_pattern = re.compile(r".*blocks\.(\d+)\..*")

    def filter_func(name: str) -> bool:
        if embedding_pattern.match(name) is not None:
            return True

        if layer_range is not None:
            start, end = layer_range
            match = block_pattern.match(name)
            if match:
                block_idx = int(match.group(1))
                if block_idx < start or block_idx >= end:
                    return True

        return False

    return filter_func


def load_calib_prompts(
    batch_size: int,
    calib_data_path: str | Path = "nkp37/OpenVid-1M",
    split: str = "train",
    column: str = "caption",
) -> list[list[str]]:
    """Load calibration prompts from file or dataset."""
    prompt_list: list[str] = []
    if isinstance(calib_data_path, Path):
        with open(calib_data_path) as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
    else:
        dataset = load_dataset(calib_data_path)
        prompt_list = list(dataset[split][column])
    return [prompt_list[i : i + batch_size] for i in range(0, len(prompt_list), batch_size)]


def get_quant_config(
    format: QuantFormat,
    algo: QuantAlgo,
    trt_high_precision_dtype: str = "BFloat16",
    alpha: float = 1.0,
    lowrank: int = 32,
) -> dict:
    """Get quantization configuration based on format.

    Args:
        format: Quantization format (int8, fp8, fp4)
        algo: Quantization algorithm (max, smoothquant, svdquant)
        trt_high_precision_dtype: TensorRT high precision dtype (Half, BFloat16, Float)
        alpha: SmoothQuant alpha parameter
        lowrank: SVDQuant lowrank parameter

    Returns:
        Quantization configuration dictionary
    """
    import copy

    if format == QuantFormat.INT8:
        if algo == QuantAlgo.SMOOTHQUANT:
            quant_config = copy.deepcopy(mtq.INT8_SMOOTHQUANT_CFG)
        else:
            quant_config = copy.deepcopy(INT8_DEFAULT_CONFIG)
    elif format == QuantFormat.FP8:
        quant_config = copy.deepcopy(FP8_DEFAULT_CONFIG)
    elif format == QuantFormat.FP4:
        if algo == QuantAlgo.SVDQUANT:
            quant_config = copy.deepcopy(NVFP4_FP8_MHA_CONFIG)
        else:
            quant_config = copy.deepcopy(NVFP4_DEFAULT_CONFIG)
    else:
        raise NotImplementedError(f"Unknown format {format}")

    set_quant_config_attr(
        quant_config,
        trt_high_precision_dtype,
        algo.value,
        alpha=alpha,
        lowrank=lowrank,
    )

    return quant_config


class WanNativeQuantizer:
    """Quantizer for Wan native pipeline."""

    def __init__(self, config: WanQuantConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device(f"cuda:{config.device_id}")
        self.wan_pipeline = None
        self.filter_func = create_quantization_filter(
            layer_range=config.layer_range,
        )

    def load_pipeline(self):
        """Load Wan native pipeline."""
        import copy

        self.logger.info(f"Loading Wan pipeline from {self.config.ckpt_dir}")
        self.logger.info(f"Task: {self.config.task}")
        self.logger.info(f"Model dtype: {self.config.model_dtype.value}")

        cfg = copy.deepcopy(WAN_CONFIGS[self.config.task])
        cfg.param_dtype = self.config.model_dtype.torch_dtype
        self.logger.info(f"  - param_dtype set to: {cfg.param_dtype}")

        self.wan_pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=self.config.ckpt_dir,
            device_id=self.config.device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            init_on_cpu=True,
            convert_model_dtype=True,
        )

        self.logger.info("Wan pipeline loaded successfully")
        self.logger.info(f"  - low_noise_model parameters: {sum(p.numel() for p in self.wan_pipeline.low_noise_model.parameters()):,}")
        self.logger.info(f"  - high_noise_model parameters: {sum(p.numel() for p in self.wan_pipeline.high_noise_model.parameters()):,}")
        if self.config.layer_range:
            start, end = self.config.layer_range
            self.logger.info(f"  - Layer range: blocks {start} to {end-1} (inclusive)")
        else:
            self.logger.info("  - Layer range: all layers")
        self.logger.info(f"  - Quantize low_noise_model: {self.config.quantize_low_noise}")
        self.logger.info(f"  - Quantize high_noise_model: {self.config.quantize_high_noise}")
        self.logger.info(f"  - Quantize SDPA: {self.config.quantize_sdpa}")
        if self.config.quantize_sdpa:
            if WAN_ATTENTION_QUANT_AVAILABLE:
                self.logger.info(f"  - SDPA quantization: ENABLED ({self.config.format.value.upper()} Q, K, V)")
            else:
                self.logger.warning("  - SDPA quantization: REQUESTED but wan_attention_quant not available!")
        else:
            self.logger.info("  - SDPA quantization: DISABLED (Q, K, V will be BF16)")

    def _disable_sdpa_quantizers(self, model: torch.nn.Module):
        """Disable SDPA quantizers (q_bmm, k_bmm, v_bmm) for a model."""
        if WAN_ATTENTION_QUANT_AVAILABLE:
            wan_attention_quant.disable_wan_sdpa_quantization(model)
        else:
            def sdpa_filter_func(name: str) -> bool:
                pattern = re.compile(
                    r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer|bmm2_output_quantizer).*"
                )
                return pattern.match(name) is not None
            mtq.disable_quantizer(model, sdpa_filter_func)

    def load_prompts(self) -> list[list[str]]:
        """Load calibration prompts."""
        if self.config.prompts_file:
            self.logger.info(f"Loading prompts from {self.config.prompts_file}")
            return load_calib_prompts(self.config.batch_size, self.config.prompts_file)
        else:
            self.logger.info("Loading prompts from OpenVid-1M dataset")
            return load_calib_prompts(
                self.config.batch_size,
                "nkp37/OpenVid-1M",
                "train",
                "caption"
            )

    def run_calibration(self, batched_prompts: list[list[str]]):
        """Run calibration by generating videos."""
        num_batches = self.config.calib_size // self.config.batch_size
        self.logger.info(f"Running calibration with {num_batches} batches")

        size = SIZE_CONFIGS[self.config.size]
        cfg = self.wan_pipeline.config
        sample_shift = getattr(cfg, 'sample_shift', 12.0)
        sample_guide_scale = getattr(cfg, 'sample_guide_scale', (3.0, 4.0))

        with tqdm(total=num_batches, desc="Calibration", unit="batch") as pbar:
            for i, prompt_batch in enumerate(batched_prompts):
                if i >= num_batches:
                    break

                prompt = prompt_batch[0] if isinstance(prompt_batch, list) else prompt_batch

                try:
                    _ = self.wan_pipeline.generate(
                        input_prompt=prompt,
                        size=size,
                        frame_num=self.config.frame_num,
                        shift=sample_shift,
                        sampling_steps=self.config.n_steps,
                        guide_scale=sample_guide_scale,
                        offload_model=self.config.offload_model,
                    )
                except Exception as e:
                    self.logger.warning(f"Calibration batch {i} failed: {e}")
                    continue

                pbar.update(1)

                if i % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        self.logger.info("Calibration completed")

    def quantize_model(self, model: torch.nn.Module, model_name: str, quant_config: dict):
        """Quantize a single model."""
        self.logger.info(f"Quantizing {model_name}...")

        model.to(self.device)
        model.eval()

        mtq.quantize(model, quant_config, forward_loop=lambda m: None)
        mtq.disable_quantizer(model, self.filter_func)
        mtq.print_quant_summary(model)

        self.logger.info(f"{model_name} quantization completed")

    def quantize(self):
        """Main quantization method."""
        if self.wan_pipeline is None:
            self.load_pipeline()

        quant_config = get_quant_config(
            self.config.format,
            self.config.algo,
            self.config.trt_high_precision_dtype.value,
            self.config.alpha,
            self.config.lowrank,
        )

        if self.config.restore_from:
            self.restore_checkpoints()
        else:
            batched_prompts = self.load_prompts()
            self.logger.info("Starting calibration phase...")

            cfg = self.wan_pipeline.config
            sample_shift = getattr(cfg, 'sample_shift', 12.0)
            sample_guide_scale = getattr(cfg, 'sample_guide_scale', (3.0, 4.0))
            self.logger.info(f"Using shift={sample_shift}, guide_scale={sample_guide_scale}")

            if self.config.quantize_low_noise:
                self.logger.info("=" * 50)
                self.logger.info("Quantizing low_noise_model")
                self.logger.info("=" * 50)

                self.wan_pipeline.low_noise_model.to(self.device)

                def low_noise_forward_loop(model):
                    self.logger.info("Running calibration for low_noise_model...")
                    num_batches = min(self.config.calib_size // self.config.batch_size, 16)

                    for i, prompt_batch in enumerate(batched_prompts[:num_batches]):
                        prompt = prompt_batch[0] if isinstance(prompt_batch, list) else prompt_batch
                        try:
                            _ = self.wan_pipeline.generate(
                                input_prompt=prompt,
                                size=SIZE_CONFIGS[self.config.size],
                                frame_num=self.config.frame_num,
                                shift=sample_shift,
                                sampling_steps=min(self.config.n_steps, 20),
                                guide_scale=sample_guide_scale,
                                offload_model=False,
                            )
                        except Exception as e:
                            self.logger.warning(f"Low noise calibration step {i} failed: {e}")

                        if i % 4 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()

                mtq.quantize(
                    self.wan_pipeline.low_noise_model,
                    quant_config.copy(),
                    forward_loop=low_noise_forward_loop
                )
                mtq.disable_quantizer(self.wan_pipeline.low_noise_model, self.filter_func)

                if not self.config.quantize_sdpa:
                    self._disable_sdpa_quantizers(self.wan_pipeline.low_noise_model)
                    self.logger.info("Disabled SDPA quantization for low_noise_model")

                self.logger.info("low_noise_model quantization summary:")
                mtq.print_quant_summary(self.wan_pipeline.low_noise_model)

                self.wan_pipeline.low_noise_model.to("cpu")
                torch.cuda.empty_cache()
            else:
                self.logger.info("⏭️  Skipping low_noise_model (disabled)")

            if self.config.quantize_high_noise:
                self.logger.info("=" * 50)
                self.logger.info("Quantizing high_noise_model")
                self.logger.info("=" * 50)

                self.wan_pipeline.high_noise_model.to(self.device)

                def high_noise_forward_loop(model):
                    self.logger.info("Running calibration for high_noise_model...")
                    num_batches = min(self.config.calib_size // self.config.batch_size, 16)
                    self.wan_pipeline.low_noise_model.to(self.device)

                    for i, prompt_batch in enumerate(batched_prompts[:num_batches]):
                        prompt = prompt_batch[0] if isinstance(prompt_batch, list) else prompt_batch
                        try:
                            _ = self.wan_pipeline.generate(
                                input_prompt=prompt,
                                size=SIZE_CONFIGS[self.config.size],
                                frame_num=self.config.frame_num,
                                shift=sample_shift,
                                sampling_steps=min(self.config.n_steps, 20),
                                guide_scale=sample_guide_scale,
                                offload_model=False,
                            )
                        except Exception as e:
                            self.logger.warning(f"High noise calibration step {i} failed: {e}")

                        if i % 4 == 0:
                            gc.collect()
                            torch.cuda.empty_cache()

                mtq.quantize(
                    self.wan_pipeline.high_noise_model,
                    quant_config.copy(),
                    forward_loop=high_noise_forward_loop
                )
                mtq.disable_quantizer(self.wan_pipeline.high_noise_model, self.filter_func)

                if not self.config.quantize_sdpa:
                    self._disable_sdpa_quantizers(self.wan_pipeline.high_noise_model)
                    self.logger.info("Disabled SDPA quantization for high_noise_model")

                self.logger.info("high_noise_model quantization summary:")
                mtq.print_quant_summary(self.wan_pipeline.high_noise_model)
            else:
                self.logger.info("⏭️  Skipping high_noise_model (disabled)")

        if self.config.compress:
            self.logger.info("Compressing quantized models...")
            if self.config.quantize_low_noise:
                mtq.compress(self.wan_pipeline.low_noise_model)
            if self.config.quantize_high_noise:
                mtq.compress(self.wan_pipeline.high_noise_model)
            self.logger.info("Compression completed")

        if self.config.quantized_ckpt_save_path:
            self.save_checkpoints()

    def save_checkpoints(self):
        """Save quantized model checkpoints."""
        save_path = self.config.quantized_ckpt_save_path
        save_path.mkdir(parents=True, exist_ok=True)

        if self.config.quantize_low_noise:
            low_noise_path = save_path / "low_noise_model_quantized.pt"
            self.logger.info(f"Saving low_noise_model to {low_noise_path}")
            mto.save(self.wan_pipeline.low_noise_model, str(low_noise_path))

        if self.config.quantize_high_noise:
            high_noise_path = save_path / "high_noise_model_quantized.pt"
            self.logger.info(f"Saving high_noise_model to {high_noise_path}")
            mto.save(self.wan_pipeline.high_noise_model, str(high_noise_path))

        self.logger.info("Checkpoints saved successfully")

    def restore_checkpoints(self):
        """Restore quantized model checkpoints."""
        restore_path = self.config.restore_from

        low_noise_path = restore_path / "low_noise_model_quantized.pt"
        high_noise_path = restore_path / "high_noise_model_quantized.pt"

        if low_noise_path.exists():
            self.logger.info(f"Restoring low_noise_model from {low_noise_path}")
            mto.restore(self.wan_pipeline.low_noise_model, str(low_noise_path))
        else:
            self.logger.warning(f"low_noise_model checkpoint not found at {low_noise_path}")

        if high_noise_path.exists():
            self.logger.info(f"Restoring high_noise_model from {high_noise_path}")
            mto.restore(self.wan_pipeline.high_noise_model, str(high_noise_path))
        else:
            self.logger.warning(f"high_noise_model checkpoint not found at {high_noise_path}")

        self.logger.info("Checkpoints restored successfully")

    def generate(self, prompt: str, **kwargs) -> torch.Tensor:
        """Generate video using quantized models."""
        if self.wan_pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        return self.wan_pipeline.generate(
            input_prompt=prompt,
            size=SIZE_CONFIGS.get(kwargs.get("size", self.config.size)),
            frame_num=kwargs.get("frame_num", self.config.frame_num),
            sampling_steps=kwargs.get("sampling_steps", 40),
            offload_model=kwargs.get("offload_model", self.config.offload_model),
        )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Quantize Wan2.2 models using native pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic FP8 quantization
    python quantize_wan_native.py --ckpt-dir /path/to/wan --format fp8 --quantized-ckpt-save-path ./quantized/

    # NVFP4 quantization (4x compression, requires Blackwell GPU)
    python quantize_wan_native.py --ckpt-dir /path/to/wan --format fp4 --calib-size 64 --quantized-ckpt-save-path ./quantized_fp4/

    # NVFP4 with SVDQuant (better quality for FP4)
    python quantize_wan_native.py --ckpt-dir /path/to/wan --format fp4 --quant-algo svdquant --lowrank 32 --quantized-ckpt-save-path ./quantized_fp4_svd/

    # INT8 quantization with SmoothQuant
    python quantize_wan_native.py --ckpt-dir /path/to/wan --format int8 --quant-algo smoothquant --quantized-ckpt-save-path ./quantized_int8/

    # Restore and generate with quantized models
    python quantize_wan_native.py --ckpt-dir /path/to/wan --restore-from ./quantized/
        """
    )

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Path to Wan checkpoint directory"
    )
    model_group.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="Wan task type"
    )
    model_group.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device ID"
    )
    model_group.add_argument(
        "--model-dtype",
        type=str,
        default="BFloat16",
        choices=[d.value for d in DataType],
        help="Precision for loading the model (Half, BFloat16, Float)"
    )
    model_group.add_argument(
        "--trt-high-precision-dtype",
        type=str,
        default="BFloat16",
        choices=[d.value for d in DataType],
        help="Precision for TensorRT high-precision layers"
    )

    quant_group = parser.add_argument_group("Quantization Configuration")
    quant_group.add_argument(
        "--format",
        type=str,
        default="fp8",
        choices=[f.value for f in QuantFormat],
        help="Quantization format"
    )
    quant_group.add_argument(
        "--quant-algo",
        type=str,
        default="max",
        choices=[a.value for a in QuantAlgo],
        help="Quantization algorithm"
    )
    quant_group.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="SmoothQuant alpha parameter"
    )
    quant_group.add_argument(
        "--lowrank",
        type=int,
        default=32,
        help="SVDQuant lowrank parameter (for FP4 with svdquant algorithm)"
    )
    quant_group.add_argument(
        "--compress",
        action="store_true",
        help="Compress quantized weights"
    )
    quant_group.add_argument(
        "--quantize-low-noise",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to quantize low_noise_model (default: true)"
    )
    quant_group.add_argument(
        "--quantize-high-noise",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to quantize high_noise_model (default: true)"
    )
    quant_group.add_argument(
        "--quantize-sdpa",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to quantize SDPA (Q, K, V in attention) with the specified format. "
             "If false, Q, K, V remain in BF16 (default: true)"
    )
    quant_group.add_argument(
        "--layer-range",
        type=str,
        default=None,
        help="Layer range to quantize (e.g., '0-20' for blocks 0-19). Wan A14B has 40 blocks (0-39)."
    )

    calib_group = parser.add_argument_group("Calibration Configuration")
    calib_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for calibration"
    )
    calib_group.add_argument(
        "--calib-size",
        type=int,
        default=32,
        help="Total number of calibration samples"
    )
    calib_group.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help="Number of denoising steps"
    )
    calib_group.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to calibration prompts file"
    )

    gen_group = parser.add_argument_group("Generation Configuration")
    gen_group.add_argument(
        "--size",
        type=str,
        default="1280*720",
        help="Video size (width*height)"
    )
    gen_group.add_argument(
        "--frame-num",
        type=int,
        default=81,
        help="Number of frames to generate"
    )
    gen_group.add_argument(
        "--offload-model",
        action="store_true",
        default=True,
        help="Offload models to CPU after forward pass"
    )

    export_group = parser.add_argument_group("Export Configuration")
    export_group.add_argument(
        "--quantized-ckpt-save-path",
        type=str,
        default=None,
        help="Path to save quantized checkpoints"
    )
    export_group.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path to restore quantized checkpoints from"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def parse_layer_range(layer_range_str: str | None) -> tuple[int, int] | None:
    """Parse layer range string (e.g., '0-20') into tuple."""
    if layer_range_str is None:
        return None
    parts = layer_range_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid layer range format: {layer_range_str}. Expected 'start-end'.")
    return (int(parts[0]), int(parts[1]))


def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    logger.info("Starting Wan Native Pipeline Quantization")

    start_time = time.time()

    try:
        layer_range = parse_layer_range(args.layer_range)
        if layer_range:
            logger.info(f"Layer range: blocks {layer_range[0]} to {layer_range[1]-1}")

        config = WanQuantConfig(
            ckpt_dir=args.ckpt_dir,
            task=args.task,
            format=QuantFormat(args.format),
            algo=QuantAlgo(args.quant_algo),
            model_dtype=DataType(args.model_dtype),
            trt_high_precision_dtype=DataType(args.trt_high_precision_dtype),
            alpha=args.alpha,
            lowrank=args.lowrank,
            batch_size=args.batch_size,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
            size=args.size,
            frame_num=args.frame_num,
            prompts_file=Path(args.prompts_file) if args.prompts_file else None,
            quantized_ckpt_save_path=Path(args.quantized_ckpt_save_path) if args.quantized_ckpt_save_path else None,
            restore_from=Path(args.restore_from) if args.restore_from else None,
            device_id=args.device_id,
            offload_model=args.offload_model,
            compress=args.compress,
            quantize_low_noise=args.quantize_low_noise.lower() == "true",
            quantize_high_noise=args.quantize_high_noise.lower() == "true",
            quantize_sdpa=args.quantize_sdpa.lower() == "true",
            layer_range=layer_range,
        )

        quantizer = WanNativeQuantizer(config, logger)
        quantizer.load_pipeline()
        quantizer.quantize()

        elapsed = time.time() - start_time
        logger.info(f"Quantization completed successfully in {elapsed:.2f} seconds")

    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

