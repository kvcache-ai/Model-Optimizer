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

"""Calibration utilities."""

import hashlib
import json
import math
import os
import time
import warnings
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypeAlias

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from modelopt.torch.opt.searcher import ForwardLoop
from modelopt.torch.quantization.utils.layerwise_calib import (
    LayerActivationCollector,
    _CheckpointState,
)
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.distributed import DistributedProcessGroup, ParallelState
from modelopt.torch.utils.network import bind_forward_method, unpatch_forward_method

from .calib import MseCalibrator, NVFP4MSECalibrator, _Calibrator
from .conversion import create_and_replace_svdquant_linear_on_the_fly, set_quantizer_by_cfg_context
from .nn import NVFP4StaticQuantizer, QuantModule, SequentialQuantizer, TensorQuantizer
from .utils import (
    disable_calib,
    enable_fake_quant,
    enable_quant,
    enable_weight_access_and_writeback,
    is_quantized_column_parallel_linear,
    is_quantized_linear,
    is_quantized_row_parallel_linear,
    persistent_materialization,
    promote_nvfp4_static_quantizers,
    quantizer_attr_names,
    reduce_amax,
    weight_attr_names,
)
from .utils.calib_utils import _GPTQ_HELPER_REGISTRY, GPTQHelper

__all__ = [
    "CalibratorFactory",
    "awq",
    "layerwise_calibrate",
    "local_hessian_calibrate",
    "max_calibrate",
    "smoothquant",
    "svdquant",
]

CalibratorFactory: TypeAlias = Callable[
    [torch.Tensor, int | tuple | list | None, Callable[..., torch.Tensor]], _Calibrator
]

_FP8_SWEEP_CALIBRATOR_REGISTRY: dict[str, CalibratorFactory] = {}


def _register_fp8_sweep_calibrator(backend: str, calibrator_factory: CalibratorFactory) -> None:
    """Register a custom calibrator factory for a quantization backend.

    When ``fp8_scale_sweep=True`` is passed to :func:`mse_calibrate`, any weight
    quantizer whose ``backend`` attribute matches a registered key will use the
    corresponding factory instead of the default :class:`MseCalibrator`.

    Args:
        backend: Backend name string (must match ``TensorQuantizer.backend``).
        calibrator_factory: Callable with signature
            ``(amax: Tensor, axis: int | tuple | list | None, quant_func: Callable)``
            that returns a :class:`_Calibrator` instance.
    """
    _FP8_SWEEP_CALIBRATOR_REGISTRY[backend] = calibrator_factory


class _CalibrationResumeCheckpoint:
    """Checkpoint helper for strict calibration resume algorithms."""

    _VERSION = 1

    def __init__(
        self,
        method: str,
        model: nn.Module,
        resume_checkpoint_dir: str | None,
        resume_save_interval: int = 1,
        resume_keep_checkpoint: bool = False,
        extra_signature_payload: dict | None = None,
    ):
        self.enabled = resume_checkpoint_dir is not None
        self.method = method
        self.save_interval = max(int(resume_save_interval), 1)
        self.keep_checkpoint = resume_keep_checkpoint
        self._state: dict[str, object] = {"stage": "start"}
        self._state_loaded = False

        if not self.enabled:
            self._checkpoint_dir = None
            self._state_path = None
            self._modelopt_path = None
            self._aux_path = None
            return

        self._checkpoint_dir = Path(resume_checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._checkpoint_dir / f"{method}_resume_state.json"
        self._modelopt_path = self._checkpoint_dir / f"{method}_resume_modelopt_state.pth"
        self._aux_path = self._checkpoint_dir / f"{method}_resume_aux_state.pth"

        payload = dict(extra_signature_payload or {})
        payload["method"] = method
        payload["quantizer_name_hash"] = self._quantizer_name_hash(model)
        payload["quantizer_count"] = len(
            [
                name
                for name, module in model.named_modules()
                if isinstance(module, (TensorQuantizer, SequentialQuantizer))
            ]
        )
        self._signature = self._build_signature(payload)
        self._load_state()

    @staticmethod
    def _build_signature(payload: dict) -> str:
        content = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _quantizer_name_hash(model: nn.Module) -> str:
        names = [
            name
            for name, module in model.named_modules()
            if isinstance(module, (TensorQuantizer, SequentialQuantizer))
        ]
        joined = "\n".join(names)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    @property
    def has_state(self) -> bool:
        return self.enabled and self._state_loaded

    @property
    def stage(self) -> str:
        return str(self._state.get("stage", "start"))

    @property
    def data(self) -> dict[str, object]:
        return self._state

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=True)
        os.replace(tmp, path)

    def _load_state(self) -> None:
        assert self._state_path is not None and self._modelopt_path is not None
        if not self._state_path.is_file() or not self._modelopt_path.is_file():
            return

        try:
            with open(self._state_path, encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            print_rank_0(
                f"Failed to parse calibration resume file {self._state_path}: {e}. Ignoring it."
            )
            return

        if state.get("version") != self._VERSION:
            print_rank_0(
                f"Resume state version mismatch for {self.method}: found {state.get('version')}, "
                f"expected {self._VERSION}. Ignoring checkpoint."
            )
            return
        if state.get("signature") != self._signature:
            print_rank_0(
                f"Resume state signature mismatch for {self.method}. "
                "Ignoring checkpoint and starting from scratch."
            )
            return

        self._state = state
        self._state_loaded = True
        print_rank_0(
            f"{self.method}: resume checkpoint loaded "
            f"(stage={self.stage}, path={self._checkpoint_dir})"
        )

    def restore_modelopt_state(self, model: nn.Module) -> nn.Module:
        if not self.has_state:
            return model
        assert self._modelopt_path is not None
        import modelopt.torch.opt as mto

        print_rank_0(f"{self.method}: restoring modelopt state from {self._modelopt_path}")
        checkpoint_state = mto.load_modelopt_state(self._modelopt_path)

        model_is_converted = False
        try:
            model_is_converted = mto.ModeloptStateManager.is_converted(model, is_root=True)
        except AssertionError:
            # Be conservative: if state consistency cannot be proven, fall back to
            # full architecture restore path below.
            model_is_converted = False

        if model_is_converted:
            # During PTQ calibration we already run on a quantized model (apply_mode has
            # inserted quantizers). In this case we only need to restore quantizer states.
            metadata = None
            for _, mode_state in reversed(checkpoint_state.get("modelopt_state_dict", [])):
                candidate = mode_state.get("metadata", {})
                if isinstance(candidate, dict) and "quantizer_state" in candidate:
                    metadata = candidate
                    break

            if metadata is None:
                print_rank_0(
                    f"{self.method}: resume checkpoint does not contain quantizer_state metadata. "
                    "Continuing without modelopt restore."
                )
                return model

            from .config import QuantizeConfig
            from .conversion import restore_quantizer_state

            return restore_quantizer_state(model, QuantizeConfig(), metadata)

        return mto.restore_from_modelopt_state(model, modelopt_state=checkpoint_state)

    def save(
        self,
        model: nn.Module,
        stage: str,
        *,
        aux_state: object | None = None,
        clear_aux_state: bool = False,
        **extra_state: object,
    ) -> None:
        if not self.enabled:
            return
        assert self._state_path is not None and self._modelopt_path is not None
        import modelopt.torch.opt as mto

        modelopt_tmp = self._modelopt_path.with_suffix(".pth.tmp")
        torch.save(mto.modelopt_state(model), modelopt_tmp)
        os.replace(modelopt_tmp, self._modelopt_path)

        if aux_state is not None:
            assert self._aux_path is not None
            aux_tmp = self._aux_path.with_suffix(self._aux_path.suffix + ".tmp")
            torch.save(aux_state, aux_tmp)
            os.replace(aux_tmp, self._aux_path)
        elif clear_aux_state and self._aux_path is not None and self._aux_path.is_file():
            self._aux_path.unlink()

        payload: dict[str, object] = {
            "version": self._VERSION,
            "signature": self._signature,
            "method": self.method,
            "stage": stage,
            "updated_at_unix": time.time(),
            **extra_state,
        }
        self._atomic_write_json(self._state_path, payload)
        self._state = payload
        self._state_loaded = True

    def load_aux_state(self):
        if not self.enabled:
            return None
        assert self._aux_path is not None
        if not self._aux_path.is_file():
            return None
        return torch.load(self._aux_path, map_location="cpu")

    def finalize(self, success: bool) -> None:
        if not self.enabled:
            return
        if not success or self.keep_checkpoint:
            return
        assert self._state_path is not None and self._modelopt_path is not None
        for path in (self._state_path, self._modelopt_path, self._aux_path):
            if path.is_file():
                path.unlink()


def weight_only_quantize(model: nn.Module):
    """Just quantize the weights of the model."""
    name_to_module = dict(model.named_modules())
    seen_modules = set()
    for module in name_to_module.values():
        if module in seen_modules:
            continue

        if isinstance(module, QuantModule):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                for weight, weight_quantizer in module.iter_weights_for_calibration():
                    weight_quantizer(weight)
        seen_modules.add(module)


def _has_expert_parallelism(module: nn.Module) -> bool:
    """Check if module has expert parallelism enabled."""
    ps = getattr(module, "parallel_state", None)
    return ps is not None and ps.expert_model_parallel_group.is_initialized()


def _check_moe_calibration_complete(quantizer, parallel_state):
    """Raise error if MoE calibration is incomplete (some ranks have amax, others don't)."""
    if isinstance(quantizer, SequentialQuantizer):
        for _q in quantizer:
            _check_moe_calibration_complete(_q, parallel_state)
        return
    for group in [
        parallel_state.data_parallel_group,
        parallel_state.expert_model_parallel_group,
        parallel_state.tensor_parallel_group,
    ]:
        if not group.is_initialized():
            continue
        has_amax = getattr(quantizer, "_amax", None) is not None
        amax_states = DistributedProcessGroup.get_dist_syncd_obj(has_amax, group, lambda objs: objs)
        if any(amax_states) and not all(amax_states):
            raise RuntimeError(
                "MoE calibration incomplete: some experts received no tokens during calibration. "
                "Increase --calib-size to ensure all experts see calibration data."
            )


def _snapshot_calibrator_amax_to_quantizers(model: nn.Module) -> None:
    """Materialize in-flight calibrator amax into quantizer `_amax` buffers.

    This allows checkpointing mid max-calibration without relying on calibrator
    internals being serialized in `modelopt_state`.
    """
    for _, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if module._disabled or getattr(module, "_dynamic", False):
            continue
        calibrator = getattr(module, "_calibrator", None)
        if calibrator is None:
            continue
        calib_amax = calibrator.compute_amax()
        if calib_amax is None:
            continue
        calib_amax = calib_amax.clone().detach()
        if getattr(module, "_amax", None) is None:
            module.register_buffer("_amax", calib_amax)
            continue
        if module._amax.shape != calib_amax.shape:
            continue
        module._amax.data.copy_(torch.max(module._amax.data, calib_amax.to(module._amax.device)))


def _seed_calibrator_from_quantizer_amax(model: nn.Module) -> list[str]:
    """Seed calibrator running state from restored quantizer `_amax` where possible.

    Returns names of unsupported quantizers that cannot be safely resumed from
    `_amax` snapshots.
    """
    unsupported: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if module._disabled or getattr(module, "_dynamic", False):
            continue
        calibrator = getattr(module, "_calibrator", None)
        if calibrator is None:
            continue
        amax = getattr(module, "_amax", None)
        if amax is None:
            continue

        if hasattr(calibrator, "_calib_amax"):
            calibrator._calib_amax = amax.detach().clone().to(amax.device)
        else:
            unsupported.append(name)
    return unsupported


def _run_forward_loop_with_progress(
    forward_loop: ForwardLoop,
    model: nn.Module,
    start_step: int = 0,
    step_callback: Callable[[int], None] | None = None,
) -> bool:
    """Run forward_loop with optional step-resume arguments when supported."""
    if getattr(forward_loop, "supports_step_resume", False):
        forward_loop(model, start_step=start_step, step_callback=step_callback)
        return True

    try:
        forward_loop(model, start_step=start_step, step_callback=step_callback)
        return True
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise

    forward_loop(model)
    return False


@torch.no_grad()
def max_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync=True,
    sync_expert_weight_amax=False,
    resume: _CalibrationResumeCheckpoint | None = None,
    max_forward_start_step: int = 0,
):
    """Calibrate the model using max.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.
        distributed_sync: Whether to sync input_quantizer amax across distributed processes.
        sync_expert_weight_amax: Whether to sync weight quantizer amax across MoE experts.
        resume: Optional strict-resume checkpoint helper. When provided, max-calibration
            progress is periodically checkpointed.
        max_forward_start_step: Number of already-processed forward batches to skip when
            resuming max calibration.

    See :class:`MaxCalibConfig <modelopt.torch.quantization.config.MaxCalibConfig>` for
    details on the remaining arguments.
    """
    enable_stats_collection(model)
    if forward_loop is None:
        weight_only_quantize(model)
    else:
        last_step = max(int(max_forward_start_step), 0)
        max_loop_save_interval = None
        if resume is not None and resume.enabled:
            total_batches = getattr(forward_loop, "num_batches", None)
            if isinstance(total_batches, int) and total_batches > 0:
                # Keep max-phase checkpointing lightweight: cap write frequency to
                # roughly <=64 saves per max-calibration pass unless user asks for less.
                max_loop_save_interval = max(resume.save_interval, max(total_batches // 64, 1))
            else:
                max_loop_save_interval = max(resume.save_interval, 64)

        def _on_step(step: int):
            nonlocal last_step
            last_step = step
            if resume is None or not resume.enabled:
                return
            assert max_loop_save_interval is not None
            if step % max_loop_save_interval != 0:
                return
            _snapshot_calibrator_amax_to_quantizers(model)
            resume.save(model, "max_loop", max_forward_step=step)

        supports_resume = _run_forward_loop_with_progress(
            forward_loop,
            model,
            start_step=max_forward_start_step,
            step_callback=_on_step,
        )
        if max_forward_start_step > 0 and not supports_resume:
            print_rank_0(
                "Forward loop does not support batch-level resume arguments. "
                "Restarting max calibration from the beginning."
            )

        if resume is not None and resume.enabled and last_step > 0:
            _snapshot_calibrator_amax_to_quantizers(model)
            assert max_loop_save_interval is not None
            if last_step % max_loop_save_interval != 0:
                resume.save(model, "max_loop", max_forward_step=last_step)
    finish_stats_collection(model)

    # Sync quantizer amax across local experts within each rank (for SequentialMLP)
    for name, module in model.named_modules():
        if hasattr(module, "layer_sync_moe_local_experts_amax"):
            module.layer_sync_moe_local_experts_amax(sync_weight_amax=sync_expert_weight_amax)

    if not distributed_sync:
        return

    # Check MoE calibration completeness before sync
    for name, module in model.named_modules():
        if isinstance(module, QuantModule) and _has_expert_parallelism(module):
            for child in module.children():
                if isinstance(child, (TensorQuantizer, SequentialQuantizer)):
                    _check_moe_calibration_complete(child, module.parallel_state)

    def sync_quantizer_amax_across_dp_ep(quantizer, parallel_state):
        """Synchronize the amax across all ranks in the data parallel and expert parallel groups."""
        if isinstance(quantizer, SequentialQuantizer):
            for _q in quantizer:
                sync_quantizer_amax_across_dp_ep(_q, parallel_state)
            return
        if getattr(quantizer, "_amax", None) is not None:
            quantizer.sync_amax_across_distributed_group(parallel_state.data_parallel_group)
            quantizer.sync_amax_across_distributed_group(parallel_state.expert_model_parallel_group)
        # TODO: create sync_bias_across_distributed_group

    # Step 2:Sync amax across data parallelism
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            for child in module.children():
                if isinstance(child, (TensorQuantizer, SequentialQuantizer)):
                    sync_quantizer_amax_across_dp_ep(child, module.parallel_state)
    # Step 3: TP sync
    # Objective: the quantization parameters when TP = 8 then changed to TP=4 then back to TP=8 should be the same

    # ColumnParallel: X @ [A_1, A_2] (weights split along Cout)
    #   activations:  TPG should have the same amax if axis in [None, -1]
    #   weights:      TPG should have the same amax if axis in [None, -1] (note: we dont use -1 axis for weights)

    # RowParallel:    [X_1, X_2] @  [A_1
    #                                A_2] (weights split along Cin)
    #   activations:  TPG should have the same amax if axis in [None]
    #   weights:      TPG should have the same amax if axis in [None, 0]

    def sync_quantizer_amax_across_tp(
        quantizer: TensorQuantizer | SequentialQuantizer,
        linear_name: str,
        quantizer_type: str,
        axes_for_sync: list,
        parallel_state: ParallelState,
    ):
        # Syncing amax across TP for sequential quantizer
        if isinstance(quantizer, SequentialQuantizer):
            for _q in quantizer:
                # Syncing amax across TP for sequential quantizer
                sync_quantizer_amax_across_tp(
                    _q, linear_name, quantizer_type, axes_for_sync, parallel_state
                )
            return
        # sync is not needed for block quantization
        if quantizer.block_sizes is not None:
            if hasattr(quantizer, "_padding"):
                warnings.warn(
                    f"Found block-quantized padded {quantizer_type} for {linear_name}, amax will"
                    " not be synced correctly."
                )
            # Skip amax sync for INT4 / W4A8 block quantization
            # Sync amax for NVFP4 (dynamic per-block, static per-tensor quantized scale)
            if getattr(quantizer.block_sizes, "type", None) == "dynamic":
                return

        if quantizer.axis in axes_for_sync and quantizer.amax is not None:
            quantizer.sync_amax_across_distributed_group(parallel_state.tensor_parallel_group)

    # Step 2: Sync amax across relevant parallelism (such as TP / EP)
    for name, module in model.named_modules():
        if getattr(module, "_parallel_state", None) is None:
            continue

        if is_quantized_column_parallel_linear(module):
            sync_quantizer_amax_across_tp(
                module.input_quantizer,
                name,
                "input_quantizer",
                axes_for_sync=[None, -1],
                parallel_state=module.parallel_state,
            )
            sync_quantizer_amax_across_tp(
                module.weight_quantizer,
                name,
                "weight_quantizer",
                axes_for_sync=[None, -1],
                parallel_state=module.parallel_state,
            )

        if is_quantized_row_parallel_linear(module):
            sync_quantizer_amax_across_tp(
                module.input_quantizer,
                name,
                "input_quantizer",
                axes_for_sync=[None],
                parallel_state=module.parallel_state,
            )

            sync_quantizer_amax_across_tp(
                module.weight_quantizer,
                name,
                "weight_quantizer",
                axes_for_sync=[None, 0],
                parallel_state=module.parallel_state,
            )

        # KV Cache Quantization
        if hasattr(module, "k_bmm_quantizer") and hasattr(module, "v_bmm_quantizer"):
            # We only support KVCache quantization with scalar per-tensor states for now (NVFP4 & FP8 KV cache)
            # So we should sync amax across DP and TP for these quantizers (DP is already synced from above)
            for quantizer in [module.k_bmm_quantizer, module.v_bmm_quantizer]:
                if isinstance(quantizer, TensorQuantizer) and quantizer.amax is not None:
                    quantizer.sync_amax_across_distributed_group(
                        module.parallel_state.tensor_parallel_group
                    )


def _mse_quant_func(x, amax, quantizer):
    """Quantization function for MSE calibration."""
    original_amax = quantizer._amax.clone() if hasattr(quantizer, "_amax") else None
    quantizer._amax = amax

    with (
        enable_quant(quantizer),
        disable_calib(quantizer),
        enable_fake_quant(quantizer),
    ):
        if hasattr(quantizer, "_original_shape"):
            x = quantizer._reset_to_original_shape(x)
        xq = quantizer(x)
        if hasattr(quantizer, "_block_reshape_size"):
            xq = xq.reshape(quantizer._block_reshape_size)

    if original_amax is not None:
        quantizer._amax = original_amax
    else:
        delattr(quantizer, "_amax")

    return xq


@torch.no_grad()
def mse_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync=True,
    step_size: float = 0.1,
    start_multiplier: float = 0.25,
    stop_multiplier: float = 4.0,
    fp8_scale_sweep: bool = False,
    resume_checkpoint_dir: str | None = None,
    resume_save_interval: int = 1,
    resume_keep_checkpoint: bool = False,
):
    """Calibrate the model using MSE-based amax search.

    This calibration method first uses max calibration to get initial amax values,
    then searches for better amax values by minimizing the MSE between original
    and quantized tensors.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.
        distributed_sync: Whether to sync amax across distributed processes.
        step_size: Step size for amax search (default: 0.1).
        start_multiplier: Starting multiplier for amax search (default: 0.25).
        stop_multiplier: Ending multiplier for amax search (default: 4.0).
        fp8_scale_sweep: If True, sweep over all 128 possible FP8 E4M3 scale values
            for NVFP4 per-block quantization instead of using multipliers.
            This is specifically designed for optimizing the FP8-quantized
            per-block scales in NVFP4 format (default: False).

    See :class:`MseCalibConfig <modelopt.torch.quantization.config.MseCalibConfig>` for
    details on the remaining arguments.
    """
    resume = _CalibrationResumeCheckpoint(
        method="mse",
        model=model,
        resume_checkpoint_dir=resume_checkpoint_dir,
        resume_save_interval=resume_save_interval,
        resume_keep_checkpoint=resume_keep_checkpoint,
        extra_signature_payload={
            "distributed_sync": bool(distributed_sync),
            "step_size": float(step_size),
            "start_multiplier": float(start_multiplier),
            "stop_multiplier": float(stop_multiplier),
            "fp8_scale_sweep": bool(fp8_scale_sweep),
        },
    )
    if resume.has_state:
        model = resume.restore_modelopt_state(model)
    stage = resume.stage
    valid_stages = {
        "start",
        "max_loop",
        "after_max",
        "after_missing_weight_amax",
        "after_calibrator_setup",
        "weight_loop",
        "done",
    }
    if stage not in valid_stages:
        print_rank_0(f"MSE resume stage {stage!r} is invalid. Restarting from scratch.")
        stage = "start"

    if stage == "done":
        print_rank_0("MSE calibration already completed in existing resume checkpoint.")
        return

    if stage in {"start", "max_loop"}:
        # Step 1: First get initial amax using max calibration
        max_start = int(resume.data.get("max_forward_step", 0)) if stage == "max_loop" else 0
        if max_start > 0:
            unsupported = _seed_calibrator_from_quantizer_amax(model)
            if unsupported:
                print_rank_0(
                    "MSE max-calibration resume encountered calibrators that cannot be resumed "
                    f"from amax snapshots ({len(unsupported)} quantizers). Restarting max phase."
                )
                max_start = 0
        max_calibrate(
            model,
            forward_loop,
            distributed_sync,
            resume=resume if resume.enabled else None,
            max_forward_start_step=max_start,
        )
        resume.save(model, "after_max")
        stage = "after_max"

    name_to_module = dict(model.named_modules())

    def _initialize_missing_weight_amax():
        """Initialize weight amax for quantizers skipped by routing during max calibration."""
        initialized = 0
        seen = set()

        for parent_module in name_to_module.values():
            if parent_module in seen:
                continue

            for weight_name in weight_attr_names(parent_module):
                weight_quantizer_name = quantizer_attr_names(weight_name).weight_quantizer
                weight_quantizer = getattr(parent_module, weight_quantizer_name, None)
                if not isinstance(weight_quantizer, TensorQuantizer):
                    continue
                if (
                    not weight_quantizer.is_enabled
                    or getattr(weight_quantizer, "_dynamic", False)
                    or getattr(weight_quantizer, "_use_constant_amax", False)
                    or getattr(weight_quantizer, "_calibrator", None) is None
                    or getattr(weight_quantizer, "_amax", None) is not None
                ):
                    continue

                was_quant_enabled = getattr(weight_quantizer, "_if_quant", True)
                was_calib_enabled = getattr(weight_quantizer, "_if_calib", False)
                weight_quantizer.disable_quant()
                weight_quantizer.enable_calib()

                try:
                    with enable_weight_access_and_writeback(parent_module, model, name_to_module):
                        try:
                            weight = getattr(parent_module, weight_name)
                        except AttributeError:
                            continue
                        weight_quantizer(weight)

                    cal = getattr(weight_quantizer, "_calibrator", None)
                    if cal is not None and cal.compute_amax() is not None:
                        weight_quantizer.load_calib_amax()
                        initialized += 1
                    if cal is not None and hasattr(cal, "reset"):
                        cal.reset()
                finally:
                    if was_quant_enabled:
                        weight_quantizer.enable_quant()
                    else:
                        weight_quantizer.disable_quant()
                    if was_calib_enabled:
                        weight_quantizer.enable_calib()
                    else:
                        weight_quantizer.disable_calib()

            seen.add(parent_module)

        return initialized

    if stage in {"after_max", "start"}:
        initialized_weight_amax = _initialize_missing_weight_amax()
        if initialized_weight_amax:
            print_rank_0(
                f"MSE calibration initialized weight amax for {initialized_weight_amax} "
                "weight quantizers without calibration hits."
            )
        resume.save(model, "after_missing_weight_amax")
        stage = "after_missing_weight_amax"

    if stage in {"after_missing_weight_amax", "after_max", "start"}:
        # Step 2: Replace calibrators with MseCalibrator
        for _, module in list(model.named_modules()):
            if isinstance(module, TensorQuantizer) and not module._disabled:
                if module._calibrator is not None and not module._dynamic and hasattr(module, "_amax"):
                    # Get the initial amax from max calibration
                    initial_amax = module._amax.clone().detach()

                    is_nvfp4_static = (
                        module.is_static_block_quant
                        and module._num_bits == (2, 1)
                        and module._block_sizes is not None
                        and module._block_sizes.get("scale_bits") == (4, 3)
                    )

                    if is_nvfp4_static:
                        # Compute and set global_amax
                        global_amax = reduce_amax(initial_amax, axis=None)
                        # Convert to NVFP4StaticQuantizer in-place
                        NVFP4StaticQuantizer.from_tensor_quantizer(module, global_amax=global_amax)

                    if fp8_scale_sweep:
                        # Check if backend has a registered custom calibrator factory.
                        _backend: str | None = getattr(module, "backend", None)
                        backend_factory = (
                            _FP8_SWEEP_CALIBRATOR_REGISTRY.get(_backend)
                            if _backend is not None
                            else None
                        )
                        if backend_factory is not None:
                            module._calibrator = backend_factory(
                                initial_amax,
                                module._calibrator._axis,
                                partial(_mse_quant_func, quantizer=module),
                            )
                            continue

                    if fp8_scale_sweep and is_nvfp4_static:
                        # Replace calibrator with NVFP4MSECalibrator
                        module._calibrator = NVFP4MSECalibrator(
                            amax=initial_amax,
                            axis=module._calibrator._axis,
                            global_amax=module.global_amax,
                            quant_func=partial(_mse_quant_func, quantizer=module),
                        )
                        continue

                    # Create MSE calibrator with quant_func
                    module._calibrator = MseCalibrator(
                        amax=initial_amax,
                        axis=module._calibrator._axis,
                        step_size=step_size,
                        start_multiplier=start_multiplier,
                        stop_multiplier=stop_multiplier,
                        quant_func=partial(_mse_quant_func, quantizer=module),
                    )

        resume.save(model, "after_calibrator_setup")
        stage = "after_calibrator_setup"

    # Identify weight quantizers by checking if they have corresponding weight parameters.
    weight_quantizers: list[tuple[str, nn.Module, str, TensorQuantizer]] = []
    seen_modules = set()
    for module_name, parent_module in name_to_module.items():
        if parent_module in seen_modules:
            continue
        for weight_name in weight_attr_names(parent_module):
            weight_quantizer_name = quantizer_attr_names(weight_name).weight_quantizer
            weight_quantizer = getattr(parent_module, weight_quantizer_name, None)
            if isinstance(weight_quantizer, TensorQuantizer) and weight_quantizer.is_enabled:
                if getattr(weight_quantizer, "_calibrator", None) is not None:
                    weight_quantizers.append(
                        (module_name, parent_module, weight_name, weight_quantizer)
                    )
        seen_modules.add(parent_module)

    weight_keys = [f"{module_name}:{weight_name}" for module_name, _, weight_name, _ in weight_quantizers]
    start_idx = 0
    if stage == "weight_loop":
        if resume.data.get("weight_keys") is not None and resume.data.get("weight_keys") != weight_keys:
            raise RuntimeError(
                "MSE resume checkpoint does not match current quantizer weight list. "
                "Use a fresh resume_checkpoint_dir."
            )
        start_idx = int(resume.data.get("next_weight_index", 0))
    elif stage in {"after_calibrator_setup", "after_missing_weight_amax", "after_max", "start"}:
        start_idx = 0
    elif stage == "done":
        return

    # Step 3: Calibrate weight quantizers ONE AT A TIME with immediate amax computation
    # This prevents massive memory accumulation seen in large models
    for idx in tqdm(
        range(start_idx, len(weight_quantizers)),
        desc="MSE weight calibration",
        total=max(len(weight_quantizers) - start_idx, 0),
    ):
        _, parent_module, weight_name, weight_quantizer = weight_quantizers[idx]
        # Enable calibration mode for the weight quantizer
        weight_quantizer.disable_quant()
        weight_quantizer.enable_calib()
        with enable_weight_access_and_writeback(parent_module, model, name_to_module):
            weight = getattr(parent_module, weight_name)
            weight_quantizer(weight)

        # IMMEDIATELY compute amax and reset calibrator to free memory
        cal = getattr(weight_quantizer, "_calibrator", None)
        if cal is not None and cal.compute_amax() is not None:
            weight_quantizer.load_calib_amax()

        weight_quantizer.enable_quant()
        weight_quantizer.disable_calib()

        # Synchronize ALL CUDA devices before resetting to ensure all async operations complete
        # This is critical for multi-GPU setups where tensors may be on different devices
        if torch.cuda.is_available():
            for dev_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(torch.device(f"cuda:{dev_id}"))

        if cal is not None and hasattr(cal, "reset"):
            cal.reset()

        if resume.enabled and (
            (idx + 1) % resume.save_interval == 0 or (idx + 1) == len(weight_quantizers)
        ):
            resume.save(
                model,
                "weight_loop",
                next_weight_index=idx + 1,
                weight_keys=weight_keys,
            )

        if (idx + 1) % 10 == 0 and torch.cuda.is_available():
            for dev_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(torch.device(f"cuda:{dev_id}"))
            torch.cuda.empty_cache()

    resume.save(model, "done", next_weight_index=len(weight_quantizers), weight_keys=weight_keys)
    resume.finalize(success=True)

    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            torch.cuda.synchronize(torch.device(f"cuda:{dev_id}"))
        torch.cuda.empty_cache()

    # TODO: Sync amax across distributed processes


@torch.no_grad()
def local_hessian_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    distributed_sync: bool = True,
    step_size: float = 0.1,
    start_multiplier: float = 0.25,
    stop_multiplier: float = 4.0,
    fp8_scale_sweep: bool = True,
    block_size: int = 16,
    debug: bool = False,
    resume_checkpoint_dir: str | None = None,
    resume_save_interval: int = 1,
    resume_keep_checkpoint: bool = False,
):
    """Calibrate the model using local Hessian-weighted MSE search.

    Instead of minimizing weight error ``||W - Wq||²``, this minimizes Hessian-weighted error
    ``loss = (W - Wq)ᵀ H (W - Wq)`` where ``H = X @ X.T`` approximates output reconstruction
    error ``||WX - WqX||²``.

    Per-block Hessians of shape ``(cin // block_size, block_size, block_size)`` are accumulated
    during forward pass and used to weight the MSE loss during scale search.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model. Required for this algorithm.
        distributed_sync: Whether to sync amax across distributed processes.
        step_size: Step size for amax search (default: 0.1).
        start_multiplier: Starting multiplier for amax search (default: 0.25).
        stop_multiplier: Ending multiplier for amax search (default: 4.0).
        fp8_scale_sweep: If True, sweep over all 128 possible FP8 E4M3 scale values
            for NVFP4 per-block quantization (default: True).
        block_size: Block size for local Hessian computation (default: 16).
        debug: If True, keep the local Hessian metadata on modules.

    See :class:`LocalHessianCalibConfig <modelopt.torch.quantization.config.LocalHessianCalibConfig>`
    for details on the configuration options.
    """
    if forward_loop is None:
        warnings.warn("forward_loop must be provided for local_hessian; skipping local_hessian")
        return

    resume = _CalibrationResumeCheckpoint(
        method="local_hessian",
        model=model,
        resume_checkpoint_dir=resume_checkpoint_dir,
        resume_save_interval=resume_save_interval,
        resume_keep_checkpoint=resume_keep_checkpoint,
        extra_signature_payload={
            "distributed_sync": bool(distributed_sync),
            "step_size": float(step_size),
            "start_multiplier": float(start_multiplier),
            "stop_multiplier": float(stop_multiplier),
            "fp8_scale_sweep": bool(fp8_scale_sweep),
            "block_size": int(block_size),
        },
    )
    if resume.has_state:
        model = resume.restore_modelopt_state(model)
    stage = resume.stage
    valid_stages = {
        "start",
        "max_loop",
        "after_max",
        "hessian_cache_loop",
        "after_hessian_cache",
        "weight_loop",
        "done",
    }
    if stage not in valid_stages:
        print_rank_0(f"local_hessian resume stage {stage!r} is invalid. Restarting from scratch.")
        stage = "start"
    if stage == "done":
        print_rank_0("local_hessian calibration already completed in existing resume checkpoint.")
        return

    def _get_module_tensor_attr(module: nn.Module, attr_name: str) -> torch.Tensor | None:
        attr = module._parameters.get(attr_name)
        if attr is None:
            attr = module._buffers.get(attr_name)
        if attr is None:
            attr = module.__dict__.get(attr_name)
        return attr if isinstance(attr, torch.Tensor) else None

    def _normalize_weight_shape(weight_shape) -> tuple[int, int] | None:
        if isinstance(weight_shape, torch.Tensor):
            weight_shape = weight_shape.detach().cpu().tolist()
        if isinstance(weight_shape, torch.Size):
            weight_shape = tuple(weight_shape)
        if isinstance(weight_shape, (list, tuple)) and len(weight_shape) >= 2:
            return int(weight_shape[-2]), int(weight_shape[-1])
        return None

    def _infer_weight_shape(module: nn.Module) -> tuple[int, int]:
        get_shape = getattr(module, "get_uncompressed_weight_shape", None)
        if callable(get_shape):
            shape = _normalize_weight_shape(get_shape())
            if shape is not None:
                return shape

        weight = _get_module_tensor_attr(module, "weight")
        if weight is not None:
            shape = _normalize_weight_shape(weight.shape)
            if shape is not None:
                return shape

        weight_shape = getattr(module, "weight_shape", None)
        shape = _normalize_weight_shape(weight_shape)
        if shape is not None:
            return shape

        out_features = getattr(module, "out_features", None)
        in_features = getattr(module, "in_features", None)
        if out_features is not None and in_features is not None:
            return int(out_features), int(in_features)

        raise RuntimeError(f"Cannot infer weight shape for local_hessian module {module}.")

    def _infer_weight_device(module: nn.Module) -> torch.device:
        for attr_name in ("weight", "weight_packed", "weight_scale"):
            attr = _get_module_tensor_attr(module, attr_name)
            if attr is not None:
                return attr.device
        for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
            return tensor.device
        return torch.device("cpu")

    def _is_local_hessian_linear(module: nn.Module) -> bool:
        if is_quantized_linear(module):
            return True
        return (
            isinstance(module, QuantModule)
            and isinstance(getattr(module, "input_quantizer", None), TensorQuantizer)
            and hasattr(module, "weight_quantizer")
            and (
                _get_module_tensor_attr(module, "weight") is not None
                or
                callable(getattr(module, "get_uncompressed_weight_shape", None))
                or _normalize_weight_shape(getattr(module, "weight_shape", None)) is not None
                or _get_module_tensor_attr(module, "weight_packed") is not None
            )
        )

    def _is_nvfp4_static_quantizer(weight_quantizer: TensorQuantizer) -> bool:
        block_sizes = getattr(weight_quantizer, "_block_sizes", None)
        return (
            weight_quantizer.is_static_block_quant
            and weight_quantizer._num_bits == (2, 1)
            and block_sizes is not None
            and block_sizes.get("scale_bits") == (4, 3)
        )

    def _promote_nvfp4_static_quantizer(
        weight_quantizer: TensorQuantizer, initial_amax: torch.Tensor | None
    ) -> None:
        if not _is_nvfp4_static_quantizer(weight_quantizer):
            return
        global_amax = reduce_amax(initial_amax, axis=None) if initial_amax is not None else None
        NVFP4StaticQuantizer.from_tensor_quantizer(weight_quantizer, global_amax=global_amax)

    def _promote_all_nvfp4_static_weight_quantizers() -> int:
        promoted = 0
        for module in name_to_module.values():
            weight_quantizer = getattr(module, "weight_quantizer", None)
            if not isinstance(weight_quantizer, TensorQuantizer):
                continue
            if not getattr(weight_quantizer, "is_enabled", False):
                continue
            if not _is_nvfp4_static_quantizer(weight_quantizer):
                continue
            was_static = isinstance(weight_quantizer, NVFP4StaticQuantizer)
            has_amax = hasattr(weight_quantizer, "_amax") and weight_quantizer._amax is not None
            initial_amax = weight_quantizer._amax.clone().detach() if has_amax else None
            _promote_nvfp4_static_quantizer(weight_quantizer, initial_amax)
            if not was_static:
                promoted += 1
        return promoted

    class LocalHessianHelper:
        """Helper class to collect activations and compute local Hessian per module."""

        cache_mode: bool = False

        def __init__(self, module, name):
            self.name = name
            self.module = module
            self.weight_shape = _infer_weight_shape(module)  # (cout, cin)
            self.cout, self.cin = self.weight_shape
            self.block_size = block_size
            self.num_blocks_per_cin = self.cin // block_size
            self.is_enabled = True
            self.weight_device = _infer_weight_device(module)

            # Accumulated Hessian per block: (cin // block_size, block_size, block_size)
            self.hessian_per_block = torch.zeros(
                self.num_blocks_per_cin,
                block_size,
                block_size,
                dtype=torch.float32,
                device=self.weight_device,
            )
            self.num_samples = 0

        def setup(self):
            """Set up the forward hook to collect activations."""
            module = self.module
            bind_forward_method(module, forward, "_forward_no_local_hessian")

            # Check if cin is divisible by block_size
            if self.cin % self.block_size != 0:
                warnings.warn(
                    f"Module {self.name}: input features ({self.cin}) not divisible by "
                    f"block_size ({self.block_size}). Skipping local Hessian for this module."
                )
                self.is_enabled = False

        def cleanup(self):
            """Clean up the forward hook."""
            unpatch_forward_method(self.module, "_forward_no_local_hessian")
            if not debug:
                if hasattr(self.module, "hessian_helper"):
                    delattr(self.module, "hessian_helper")

        def accumulate_hessian(self, input_tensor: torch.Tensor):
            """Accumulate local Hessian from input activations.

            Args:
                input_tensor: Input tensor of shape (..., cin)
            """
            if not self.is_enabled:
                return

            # Flatten to (num_tokens, cin)
            x = input_tensor.reshape(-1, self.cin).T  # (cin, num_tokens)
            x = x.reshape(self.num_blocks_per_cin, self.block_size, -1)  # (num_blocks, bs, n)

            # Compute H = X @ X.T for each block and accumulate
            hessian_batch = (x @ x.transpose(-1, -2)).to(torch.float32)
            if hessian_batch.device != self.hessian_per_block.device:
                hessian_batch = hessian_batch.to(self.hessian_per_block.device)
            self.hessian_per_block += hessian_batch
            self.num_samples += input_tensor.numel() // self.cin

        def get_error_func(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """Get the local Hessian error function for MSE calibration."""
            cout = self.cout
            bs = self.block_size
            # Normalize hessian by number of samples
            hessian = self.hessian_per_block / max(self.num_samples, 1)
            hessian_cache: dict[torch.device, torch.Tensor] = {}

            def local_hessian_error(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
                """Compute local Hessian-weighted error."""
                original_shape = x.shape
                # Reshape to (cout, num_blocks_per_cin, block_size)
                dw = (x - xq).view(cout, -1, bs)
                hessian_for_x = hessian
                if hessian_for_x.device != x.device:
                    hessian_for_x = hessian_cache.get(x.device)
                    if hessian_for_x is None:
                        hessian_for_x = hessian.to(x.device)
                        hessian_cache[x.device] = hessian_for_x
                # Use einsum to avoid materializing cout-repeated Hessian
                # dw: (cout, n_blocks, bs), hessian: (n_blocks, bs, bs) -> (cout, n_blocks)
                block_loss = torch.einsum("cnb,nbd,cnd->cn", dw, hessian_for_x, dw)
                block_loss = block_loss.reshape(-1)
                error = block_loss.unsqueeze(-1).expand(-1, bs).reshape(original_shape)
                return error

            return local_hessian_error

    def forward(self, input, *args, **kwargs):
        """Custom forward that collects activations in cache mode."""
        if LocalHessianHelper.cache_mode and self.hessian_helper.is_enabled:
            # Get local tensor from DTensor if applicable
            input_local = input.to_local() if hasattr(input, "to_local") else input
            self.hessian_helper.accumulate_hessian(input_local)

        # Forward without quantization during caching
        if LocalHessianHelper.cache_mode:
            self.weight_quantizer.disable()
            try:
                out = self._forward_no_local_hessian(input, *args, **kwargs)
            finally:
                self.weight_quantizer.enable()
            return out

        return self._forward_no_local_hessian(input, *args, **kwargs)

    def _reset_hessian_helpers() -> None:
        for _, module in weight_quantizers_info:
            helper = module.hessian_helper
            helper.hessian_per_block.zero_()
            helper.num_samples = 0

    def _collect_hessian_cache_state() -> dict[str, dict[str, object]]:
        state: dict[str, dict[str, object]] = {}
        for name, module in weight_quantizers_info:
            helper = module.hessian_helper
            state[name] = {
                "hessian_per_block": helper.hessian_per_block.detach().cpu(),
                "num_samples": int(helper.num_samples),
            }
        return state

    def _restore_hessian_cache_state(cache_state: dict) -> int:
        helpers_state = cache_state.get("hessian_helpers", {})
        if not isinstance(helpers_state, dict):
            return 0

        restored = 0
        for name, module in weight_quantizers_info:
            helper_state = helpers_state.get(name)
            if not isinstance(helper_state, dict):
                continue

            hessian_tensor = helper_state.get("hessian_per_block")
            num_samples = helper_state.get("num_samples", 0)
            if not isinstance(hessian_tensor, torch.Tensor):
                continue

            helper = module.hessian_helper
            if hessian_tensor.shape != helper.hessian_per_block.shape:
                continue

            helper.hessian_per_block.copy_(
                hessian_tensor.to(device=helper.hessian_per_block.device, dtype=torch.float32)
            )
            helper.num_samples = int(num_samples)
            restored += 1
        return restored

    # First, run max_calibrate on the whole model to get initial amax for all quantizers.
    # This calibrates both weight_quantizer and input_quantizer with max calibration.
    if stage in {"start", "max_loop"}:
        print_rank_0("local_hessian: Running max calibration for all quantizers...")
        max_start = int(resume.data.get("max_forward_step", 0)) if stage == "max_loop" else 0
        if max_start > 0:
            unsupported = _seed_calibrator_from_quantizer_amax(model)
            if unsupported:
                print_rank_0(
                    "local_hessian max-calibration resume encountered calibrators that cannot be "
                    f"resumed from amax snapshots ({len(unsupported)} quantizers). "
                    "Restarting max phase."
                )
                max_start = 0
        max_calibrate(
            model,
            forward_loop,
            distributed_sync,
            resume=resume if resume.enabled else None,
            max_forward_start_step=max_start,
        )
        resume.save(model, "after_max")
        stage = "after_max"

    # Setup helpers for all quantized linear modules
    name_to_module = dict(model.named_modules())
    weight_quantizers_info = []
    all_patched_modules = []  # Track all modules for cleanup (including disabled ones)

    try:
        for name, module in name_to_module.items():
            if _is_local_hessian_linear(module) and module.weight_quantizer.is_enabled:
                module.hessian_helper = LocalHessianHelper(module, name)
                all_patched_modules.append((name, module))
                module.hessian_helper.setup()
                if module.hessian_helper.is_enabled:
                    weight_quantizers_info.append((name, module))

        if stage in {"after_max", "hessian_cache_loop"}:
            if len(weight_quantizers_info) == 0:
                print_rank_0(
                    "local_hessian: no eligible modules for Hessian cache; skipping cache pass."
                )
                if resume.enabled:
                    resume.save(
                        model,
                        "after_hessian_cache",
                        hessian_cache_step=0,
                        clear_aux_state=True,
                    )
                stage = "after_hessian_cache"
            else:
                # Cache activations by running forward loop
                LocalHessianHelper.cache_mode = True
                print_rank_0("local_hessian: Caching activations and computing local Hessian...")

                cache_start_step = (
                    int(resume.data.get("hessian_cache_step", 0))
                    if stage == "hessian_cache_loop"
                    else 0
                )
                cache_last_step = max(cache_start_step, 0)

                if cache_start_step > 0:
                    aux_state = resume.load_aux_state()
                    if aux_state is None:
                        print_rank_0(
                            "local_hessian: cache resume metadata exists but no aux cache state file "
                            "found. Restarting cache pass from step 0."
                        )
                        cache_start_step = 0
                    else:
                        restored_count = _restore_hessian_cache_state(aux_state)
                        if restored_count == 0:
                            cached_entries = 0
                            if isinstance(aux_state, dict):
                                helpers_state = aux_state.get("hessian_helpers", {})
                                if isinstance(helpers_state, dict):
                                    cached_entries = len(helpers_state)
                            print_rank_0(
                                "local_hessian: failed to restore cached hessian accumulators "
                                f"(cached_entries={cached_entries}, helpers={len(weight_quantizers_info)}); "
                            "restarting cache pass from step 0."
                            )
                            cache_start_step = 0
                        else:
                            print_rank_0(
                                f"local_hessian: restored hessian cache for {restored_count} modules "
                                f"at step {cache_start_step}."
                            )

                cache_save_interval = None
                if resume.enabled:
                    total_batches = getattr(forward_loop, "num_batches", None)
                    if isinstance(total_batches, int) and total_batches > 0:
                        # Limit hessian-cache checkpoint IO overhead.
                        cache_save_interval = max(resume.save_interval, max(total_batches // 16, 1))
                    else:
                        cache_save_interval = max(resume.save_interval, 128)

                def _on_cache_step(step: int) -> None:
                    nonlocal cache_last_step
                    cache_last_step = step
                    if resume is None or not resume.enabled:
                        return
                    assert cache_save_interval is not None
                    if step % cache_save_interval != 0:
                        return
                    helper_state = _collect_hessian_cache_state()
                    if not helper_state:
                        return
                    resume.save(
                        model,
                        "hessian_cache_loop",
                        hessian_cache_step=step,
                        aux_state={"hessian_helpers": helper_state},
                    )

                supports_cache_resume = _run_forward_loop_with_progress(
                    forward_loop,
                    model,
                    start_step=cache_start_step,
                    step_callback=_on_cache_step,
                )
                if cache_start_step > 0 and not supports_cache_resume:
                    print_rank_0(
                        "Forward loop does not support batch-level resume arguments for local_hessian "
                        "cache. Restarting cache pass from step 0."
                    )
                    _reset_hessian_helpers()
                    cache_last_step = 0
                    _run_forward_loop_with_progress(
                        forward_loop,
                        model,
                        start_step=0,
                        step_callback=_on_cache_step,
                    )

                if resume.enabled:
                    resume.save(
                        model,
                        "after_hessian_cache",
                        hessian_cache_step=cache_last_step,
                        clear_aux_state=True,
                    )
                stage = "after_hessian_cache"
                LocalHessianHelper.cache_mode = False

        # TODO(fridah-nv): Sync Hessian across distributed processes if needed

        # Replace calibrators with MseCalibrator using local Hessian error function
        print_rank_0("local_hessian: Running MSE calibration with local Hessian loss...")
        skip_weight_quantizer_ids: set[int] = set()
        for name, module in weight_quantizers_info:
            weight_quantizer = module.weight_quantizer
            helper = module.hessian_helper

            has_amax = hasattr(weight_quantizer, "_amax") and weight_quantizer._amax is not None
            initial_amax = weight_quantizer._amax.clone().detach() if has_amax else None

            def quant_func(x, amax, quantizer=weight_quantizer):
                original_amax = quantizer._amax.clone() if hasattr(quantizer, "_amax") else None
                quantizer._amax = amax

                with (
                    enable_quant(quantizer),
                    disable_calib(quantizer),
                    enable_fake_quant(quantizer),
                ):
                    if hasattr(quantizer, "_original_shape"):
                        x = quantizer._reset_to_original_shape(x)
                    xq = quantizer(x)
                    if hasattr(quantizer, "_block_reshape_size"):
                        xq = xq.reshape(quantizer._block_reshape_size)

                if original_amax is not None:
                    quantizer._amax = original_amax
                else:
                    delattr(quantizer, "_amax")

                return xq

            is_nvfp4_static = _is_nvfp4_static_quantizer(weight_quantizer)

            if is_nvfp4_static:
                _promote_nvfp4_static_quantizer(weight_quantizer, initial_amax)

            if initial_amax is None:
                warnings.warn(
                    f"Module {name}: no max-calibrated weight amax; "
                    "falling back to weight-derived NVFP4 scales during export."
                )
                skip_weight_quantizer_ids.add(id(weight_quantizer))
                continue

            if helper.num_samples == 0:
                warnings.warn(
                    f"Module {name}: no calibration tokens reached this module; "
                    "falling back to max-calibrated weight scale."
                )
                skip_weight_quantizer_ids.add(id(weight_quantizer))
                continue

            error_func = helper.get_error_func()

            if fp8_scale_sweep and is_nvfp4_static:
                weight_quantizer._calibrator = NVFP4MSECalibrator(
                    amax=initial_amax,
                    axis=weight_quantizer._calibrator._axis if weight_quantizer._calibrator else None,
                    global_amax=weight_quantizer.global_amax,
                    quant_func=quant_func,
                    error_func=error_func,
                )
            else:
                weight_quantizer._calibrator = MseCalibrator(
                    amax=initial_amax,
                    axis=weight_quantizer._calibrator._axis if weight_quantizer._calibrator else None,
                    step_size=step_size,
                    start_multiplier=start_multiplier,
                    stop_multiplier=stop_multiplier,
                    quant_func=quant_func,
                    error_func=error_func,
                )

        # Process weights ONE AT A TIME with immediate amax computation and cleanup
        weight_list = [
            (name, module)
            for name, module in weight_quantizers_info
            if id(module.weight_quantizer) not in skip_weight_quantizer_ids
            and module.weight_quantizer._calibrator is not None
        ]
        weight_keys = [name for name, _ in weight_list]
        start_idx = 0
        if stage == "weight_loop":
            saved_weight_keys = resume.data.get("weight_keys")
            if saved_weight_keys is not None and saved_weight_keys != weight_keys:
                raise RuntimeError(
                    "local_hessian resume checkpoint does not match current weight list. "
                    "Use a fresh resume_checkpoint_dir."
                )
            start_idx = int(resume.data.get("next_weight_index", 0))
            if start_idx < 0 or start_idx > len(weight_list):
                raise RuntimeError(
                    f"Invalid local_hessian next_weight_index={start_idx} for "
                    f"{len(weight_list)} weights."
                )

        for idx in range(start_idx, len(weight_list)):
            name, module = weight_list[idx]
            weight_quantizer = module.weight_quantizer
            cal = weight_quantizer._calibrator

            try:
                # Step 1: Calibrate this weight
                weight_quantizer.disable_quant()
                weight_quantizer.enable_calib()
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    weight = module.weight
                    weight_quantizer(weight)

                # Step 2: IMMEDIATELY compute amax before calibration data grows
                if cal.compute_amax() is not None:
                    weight_quantizer.load_calib_amax()
            finally:
                weight_quantizer.enable_quant()
                weight_quantizer.disable_calib()

            if torch.cuda.is_available():
                for dev_id in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(torch.device(f"cuda:{dev_id}"))

            if hasattr(cal, "reset"):
                cal.reset()

            if resume.enabled and (
                (idx + 1) % resume.save_interval == 0 or (idx + 1) == len(weight_list)
            ):
                resume.save(
                    model,
                    "weight_loop",
                    next_weight_index=idx + 1,
                    weight_keys=weight_keys,
                    clear_aux_state=True,
                )

        if torch.cuda.is_available():
            for dev_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(torch.device(f"cuda:{dev_id}"))

        promoted = _promote_all_nvfp4_static_weight_quantizers()
        if promoted:
            print_rank_0(
                f"local_hessian: Promoted {promoted} NVFP4 static weight quantizers "
                "for export fallback."
            )
        resume.save(
            model,
            "done",
            next_weight_index=len(weight_list),
            weight_keys=weight_keys,
            clear_aux_state=True,
        )
        resume.finalize(success=True)
    finally:
        LocalHessianHelper.cache_mode = False
        for _name, module in all_patched_modules:
            helper = getattr(module, "hessian_helper", None)
            if helper is not None:
                helper.cleanup()

    print_rank_0("local_hessian: Calibration complete.")


def enable_stats_collection(model: nn.Module):
    """Enable stats collection for all quantizers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module._disabled:
            if module._use_constant_amax:
                # use_constant_amax quantizers use a fixed amax and don't need calibration.
                # Disable quantization during calibration so it doesn't affect other quantizers.
                module.disable_quant()
                continue
            elif module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()


def finish_stats_collection(model: nn.Module, method: str | None = None, **kwargs):
    """Finish stats collection for all quantizers in the model."""
    for _, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or module._disabled:
            continue

        if module._use_constant_amax:
            # Re-enable quantization for use_constant_amax quantizers disabled in enable_stats_collection.
            module.enable_quant()
            continue

        cal = getattr(module, "_calibrator", None)
        if cal and not getattr(module, "_dynamic", False):
            if method in {"entropy"}:
                if cal.compute_amax(method) is not None:
                    module.load_calib_amax("entropy", **kwargs)
            elif cal.compute_amax(**kwargs) is not None:
                module.load_calib_amax(**kwargs)

        if module.bias_calibrator is not None and module.bias_type == "static":
            module.load_calib_bias()

        module.enable_quant()
        module.disable_calib()


@torch.no_grad()
def disable_pre_quant_scale_and_resmooth(linear: nn.Module, delete_pre_quant_scale: bool = False):
    """Disable pre_quant_scale and resmooth the quantized linear weights."""
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer._enable_pre_quant_scale, (
        "pre_quant_scale should be enabled first!"
    )
    assert hasattr(linear.input_quantizer, "_pre_quant_scale"), (
        "pre_quant_scale should be available"
    )

    pre_quant_scale = linear.input_quantizer._pre_quant_scale.to(torch.float32)

    linear.weight.copy_(
        (linear.weight * pre_quant_scale.squeeze()[None, :]).to(linear.weight.dtype)
    )
    linear.weight_quantizer.reset_amax()
    max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))

    # Lets not delete the _pre_quant_scale, it might useful later; Instead we will disable it
    linear.input_quantizer._enable_pre_quant_scale = False

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        device, dtype = linear.weight.device, linear.weight.dtype
        linear.input_quantizer.amax = linear.input_quantizer._amax_for_smoothing.amax().to(
            device=device, dtype=dtype
        )

    if delete_pre_quant_scale:
        delattr(linear.input_quantizer, "_pre_quant_scale")
        linear.input_quantizer._enable_pre_quant_scale = False


# A global variable used during auto_quantize to avoid folding pre_quant_scale to weights
_ENABLE_FOLDING_PQS_TO_WEIGHTS = True


@torch.no_grad()
def _apply_weight_pre_quant_scale(linear, pre_quant_scale):
    apply_weight_smooth_scale = getattr(linear, "apply_weight_smooth_scale", None)
    if _ENABLE_FOLDING_PQS_TO_WEIGHTS and callable(apply_weight_smooth_scale):
        apply_weight_smooth_scale(pre_quant_scale)
    elif _ENABLE_FOLDING_PQS_TO_WEIGHTS:
        linear.weight.data.copy_(
            (linear.weight * pre_quant_scale.to(linear.weight.device).squeeze()[None, :]).to(
                linear.weight.dtype
            )
        )
    else:
        linear.weight_quantizer._enable_pre_quant_scale = True
        linear.weight_quantizer.pre_quant_scale = pre_quant_scale.squeeze()[None, :].to(
            linear.weight.dtype
        )

    linear.weight_quantizer.reset_amax()
    enable_weight_access = getattr(linear, "enable_weight_access_and_writeback", None)
    if callable(enable_weight_access):
        with enable_weight_access():
            max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))
    else:
        max_calibrate(linear, lambda linear: linear.weight_quantizer(linear.weight))


@torch.no_grad()
def apply_pre_quant_scale_and_smooth(
    linear: nn.Module, pre_quant_scale: torch.Tensor | None = None
):
    """Apply pre_quant_scale and smooth the quantized linear weights.

    If pre_quant_scale is not provided, the existing pre_quant_scale of input_quantizer will be used.
    """
    assert is_quantized_linear(linear), "Only quantized linear modules are supported"
    assert linear.input_quantizer.pre_quant_scale is None, "pre_quant_scale should be None first!"

    if pre_quant_scale is None:
        pre_quant_scale = (
            linear.input_quantizer._pre_quant_scale
            if hasattr(linear.input_quantizer, "_pre_quant_scale")
            else None
        )

    assert pre_quant_scale is not None, "pre_quant_scale should be provided or already set"

    assert torch.all(pre_quant_scale > 0), "pre_quant_scale should be positive"

    # pre_quant_scale should be in fp32 for the scaling math to be numerically safe
    pre_quant_scale = pre_quant_scale.to(torch.float32)

    linear.input_quantizer._enable_pre_quant_scale = True
    linear.input_quantizer.pre_quant_scale = pre_quant_scale.to(linear.weight.dtype)

    inv_scale = 1.0 / pre_quant_scale
    _apply_weight_pre_quant_scale(linear, inv_scale)

    if linear.input_quantizer.amax is not None:
        assert hasattr(linear.input_quantizer, "_amax_for_smoothing")
        device, dtype = linear.weight.device, linear.weight.dtype
        _amax_for_smoothing = linear.input_quantizer._amax_for_smoothing.to(
            device=device, dtype=dtype
        )
        linear.input_quantizer.amax = (
            (_amax_for_smoothing * pre_quant_scale.to(device)).amax().to(dtype)
        )

        if is_quantized_column_parallel_linear(linear) or is_quantized_row_parallel_linear(linear):
            linear.input_quantizer.sync_amax_across_distributed_group(
                linear.parallel_state.tensor_parallel_group
            )


@torch.no_grad()
def smoothquant(model: nn.Module, forward_loop: ForwardLoop | None = None, alpha=1.0):
    """Smooth-Quant variant with per-channel weight scaling.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`SmoothQuantCalibConfig <modelopt.torch.quantization.config.SmoothQuantCalibConfig>` for
    details on the remaining arguments.
    """
    # distributed synchronization
    # max_calibrate performs amax sync for data parallel

    # Column parallel:
    # activations:  TPG should have the same pre_quant_scale
    #               This is achieved by syncing act_amax and weight_scale across TPG which is used to
    #               compute pre_quant_scale
    # weights:      no-op

    # Row parallel:
    # activations:  TPG should have same activation amax
    # weights:      TPG should have the same weight amax

    assert forward_loop is not None, "forward_loop must be provided for smoothquant"
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.input_quantizer.is_enabled
            and module.input_quantizer.axis is None
        ):
            module.input_quantizer.axis = -1

    max_calibrate(model, forward_loop)

    def postprocess(module):
        # It is important to keep scaling math in fp32 to be numerically safe
        act_amax = module.input_quantizer.amax.float()
        weight_scale = module.weight.abs().amax(dim=0, keepdim=True)
        device, dtype = module.weight.device, module.weight.dtype

        parallel_group = module.parallel_state.tensor_parallel_group
        if is_quantized_column_parallel_linear(module) and parallel_group.is_initialized():
            dist.all_reduce(act_amax, op=dist.ReduceOp.MAX, group=parallel_group.group)
            dist.all_reduce(weight_scale, op=dist.ReduceOp.MAX, group=parallel_group.group)

        scale_a = (weight_scale.pow(1 - alpha) / act_amax.pow(alpha)).squeeze()

        # Now that activation per-channel amax have been collected, use per-tensor quantization for activation
        # TODO: make this a buffer after we support only heterogeneous checkpointing for MCore
        module.input_quantizer._amax_for_smoothing = act_amax.cpu()
        module.input_quantizer.reset_amax()
        module.input_quantizer.axis = None
        module.input_quantizer.amax = act_amax.amax().to(dtype=dtype, device=device)

        # Some channel could have 0 amax which causes scale_a to overflow. Explicitly mask them out here
        epsilon = 1.0 / (1 << 31)
        if scale_a.min() <= epsilon:
            zero_mask = act_amax <= epsilon
            scale_a[zero_mask] = 1
        scale_a = scale_a.clamp(min=1e-4, max=1e4)
        apply_pre_quant_scale_and_smooth(module, scale_a)

    name_to_module = dict(model.named_modules())
    smoothed_modules = 0
    for name, module in name_to_module.items():
        if is_quantized_linear(module):
            if not hasattr(module.input_quantizer, "_amax"):
                warnings.warn(f"{name} is not calibrated, skip smoothing")
                continue
            if module.input_quantizer.num_bits != 8 or module.weight_quantizer.num_bits != 8:
                warnings.warn(f"Only int8 smoothing is supported, skip {name}")
                continue
            if module.input_quantizer.axis != -1:
                warnings.warn(f"Only per-channel smoothing is supported, skip {name}")
                continue

            assert module.input_quantizer._amax.numel() > 1, (
                f"Error: {name} has only one channel to smooth"
            )

            with enable_weight_access_and_writeback(module, model, name_to_module):
                postprocess(module)

            smoothed_modules += 1
    print_rank_0(f"Smoothed {smoothed_modules} modules")


def awq(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    algorithm: str = "awq_lite",
    **kwargs,
):
    """Apply AWQ to the model.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`AWQFullCalibConfig <modelopt.torch.quantization.config.AWQFullCalibConfig>` for
    details on the remaining arguments.
    """
    with SequentialQuantizer.convert_to_single_quantizer(model):
        if algorithm in ["awq_full", "awq_lite"]:
            awq_lite(model, forward_loop, **kwargs)

        if algorithm in ["awq_full", "awq_clip"]:
            awq_clip(model, forward_loop, **kwargs)

    # Special handling for SequentialQuantizer
    # Pre-compute name_to_module dict to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in model.named_modules():
        if is_quantized_linear(module) and isinstance(module.weight_quantizer, SequentialQuantizer):
            with enable_weight_access_and_writeback(module, model, name_to_module):
                max_calibrate(module, lambda linear: linear.weight_quantizer(module.weight))


@torch.no_grad()
def awq_lite(
    model: nn.Module,
    forward_loop: ForwardLoop,
    alpha_step: float = 0.1,
    debug: bool = False,
    **kwargs,
):
    """Lite version of AWQ.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`AWQLiteCalibConfig <modelopt.torch.quantization.config.AWQLiteCalibConfig>` for
    details on the remaining arguments.
    """
    if forward_loop is None:
        warnings.warn("forward_loop must be provided for awq_lite; skipping awq_lite")
        return

    class AWQLiteHelper:
        cache_mode: bool = False

        def __init__(self, module, name):
            self.name = name
            self.act_scale = 0.0
            self.num_cache_steps = 0
            self.num_search_steps = 0
            self.weight_dtype = module.weight.dtype
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)
            self.weight_scale = get_weight_scale(module.weight, self.block_size)
            self.loss = {
                k.item(): torch.zeros((), device=module.weight.device, dtype=torch.float32)
                for k in torch.arange(0, 1.0 + alpha_step, alpha_step)
            }
            self.best_scale = None
            self.best_alpha = None
            self.is_input_quantized = module.input_quantizer.is_enabled
            self.num_tokens = 0
            self.module = module
            self.is_enabled = True

        def setup(self):
            module = self.module
            bind_forward_method(module, forward, "_forward_no_awq")
            if module.input_quantizer.is_enabled:
                module.input_quantizer.disable()
                if module.input_quantizer.axis not in [None, -1]:
                    self.is_enabled = False
                    return
                module.input_quantizer.axis = -1

        def cleanup(self):
            module = self.module
            if hasattr(module, "_if_calib"):
                delattr(module, "_if_calib")
            unpatch_forward_method(module, "_forward_no_awq")

    def get_weight_scale(weight, block_size=None):
        org_shape = weight.shape
        slice_after_padding = None
        if block_size:
            if org_shape[-1] % block_size != 0:
                slice_after_padding = slice(org_shape[-1])
                weight = F.pad(weight, (0, block_size - org_shape[-1] % block_size), "constant", 0)
                org_shape = weight.shape
            weight = weight.contiguous().view(-1, block_size)
        weight_abs = weight.abs()  # Cache to avoid redundant computation
        weight_abs_amax = weight_abs.amax(dim=1, keepdim=True)
        scale = weight_abs / (weight_abs_amax + torch.finfo(weight.dtype).tiny)
        scale = scale.view(org_shape)
        if slice_after_padding is not None:
            scale = scale[..., slice_after_padding]
        scale = scale.mean(0).to(torch.float32)
        return scale

    def get_act_scale(x):
        return x.abs().contiguous().view(-1, x.shape[-1]).mean(0).to(torch.float32)

    def get_scale(x_max, w_max, alpha, tensor_parallel_group=None):
        scales = (
            (
                x_max.pow(alpha)
                / (w_max.to(x_max.device).pow(1 - alpha) + torch.finfo(torch.float32).tiny)
            )
            .clamp(min=1e-4, max=1e4)
            .view(-1)
        )
        scales = (scales / (scales.max() * scales.min()).sqrt()).view(-1)
        if tensor_parallel_group and tensor_parallel_group.is_initialized():
            dist.all_reduce(scales, op=dist.ReduceOp.SUM, group=tensor_parallel_group.group)
            scales /= tensor_parallel_group.world_size()
        return scales

    def update_loss(self, out, out_actual, alpha):
        out_actual = out_actual[0] if isinstance(out_actual, tuple) else out_actual
        out = out[0] if isinstance(out, tuple) else out
        out = out.to_local() if hasattr(out, "to_local") else out
        out_actual = out_actual.to_local() if hasattr(out_actual, "to_local") else out_actual
        loss = (out - out_actual).float().pow(2).mean()
        self.awq_lite.loss[alpha] += loss.to(self.awq_lite.loss[alpha].device)

    def update_best_params(self):
        if not self.awq_lite.is_enabled:
            return
        self.awq_lite.loss.update({k: float(v) for k, v in self.awq_lite.loss.items()})
        self.awq_lite.best_alpha = min(self.awq_lite.loss, key=self.awq_lite.loss.get)
        self.awq_lite.best_scale = get_scale(
            self.awq_lite.act_scale,
            self.awq_lite.weight_scale,
            self.awq_lite.best_alpha,
            (
                self.parallel_state.tensor_parallel_group
                if is_quantized_column_parallel_linear(self)
                else None
            ),
        )

    def forward(self, input, *args, **kwargs):
        # Collect actual output without quantization
        self.weight_quantizer.disable()
        if hasattr(self.input_quantizer, "_pre_quant_scale"):
            delattr(self.input_quantizer, "_pre_quant_scale")
        if hasattr(self.weight_quantizer, "_pre_quant_scale"):
            delattr(self.weight_quantizer, "_pre_quant_scale")
        out_actual = self._forward_no_awq(input, *args, **kwargs)
        self.weight_quantizer.enable()

        if input.numel() == 0 or not self.awq_lite.is_enabled:
            # For MoEs, some experts might see 0 tokens
            return out_actual

        if AWQLiteHelper.cache_mode:
            # Get local tensor from Dtensor
            input = input.to_local() if hasattr(input, "to_local") else input

            self.awq_lite.act_scale += get_act_scale(self.input_quantizer(input))
            self.awq_lite.num_cache_steps += 1
            self.awq_lite.num_tokens += input.numel() / input.shape[-1]
            if self.awq_lite.is_input_quantized:
                with set_quantizer_by_cfg_context(
                    self.input_quantizer, [{"quantizer_name": "*", "enable": True}]
                ):
                    max_calibrate(self.input_quantizer, lambda quantizer: quantizer(input), False)
            return out_actual

        for alpha in self.awq_lite.loss:
            awq_scale = get_scale(
                self.awq_lite.act_scale,
                self.awq_lite.weight_scale,
                alpha,
                (
                    self.parallel_state.tensor_parallel_group
                    if is_quantized_column_parallel_linear(self)
                    else None
                ),
            )
            weight_dtype = getattr(self.awq_lite, "weight_dtype", None)
            if weight_dtype is None:
                weight_dtype = self.weight.dtype
            self.input_quantizer.pre_quant_scale = (1 / awq_scale).to(weight_dtype)
            self.weight_quantizer.pre_quant_scale = awq_scale.to(weight_dtype)
            out = self._forward_no_awq(input, *args, **kwargs)
            update_loss(self, out, out_actual, alpha)

        self.awq_lite.num_search_steps += 1

        # Now forward the actual output without any quantization
        return out_actual

    def is_awq_linear(module):
        if is_quantized_linear(module):
            return True
        return (
            isinstance(module, QuantModule)
            and isinstance(getattr(module, "input_quantizer", None), TensorQuantizer)
            and hasattr(module, "weight_quantizer")
            and callable(getattr(module, "enable_weight_access_and_writeback", None))
        )

    # Pre-compute name_to_module dict ONCE to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in name_to_module.items():
        if is_awq_linear(module) and module.weight_quantizer.is_enabled:
            with enable_weight_access_and_writeback(module, model, name_to_module):
                if not is_quantized_linear(module):
                    continue
                module.awq_lite = AWQLiteHelper(module, name)
            module.awq_lite.setup()

    # Collect activation scale values
    AWQLiteHelper.cache_mode = True
    print_rank_0("awq_lite: Caching activation statistics...")

    # Lets enable stats collection
    # This will collect amax for input_quantizers and KV quantizers during the caching mode forward pass
    enable_stats_collection(model)
    forward_loop(model)

    # Call max_calibrate to load the amax values collected during the caching mode forward pass
    # This will also perform distributed amax sync for input_quantizers
    max_calibrate(model, lambda model: None)

    def sync_act_scale_across_dp(module, data_parallel_group):
        """Sync activation scale across Data Parallel (DP)."""
        if data_parallel_group.is_initialized():
            dist.all_reduce(
                module.awq_lite.act_scale, op=dist.ReduceOp.AVG, group=data_parallel_group.group
            )

    for name, module in model.named_modules():
        if (
            is_awq_linear(module)
            and hasattr(module, "awq_lite")
            and module.awq_lite.num_cache_steps > 0
        ):
            # Hack: MoEs forward all tokens through all experts if _if_calib is True
            module._if_calib = True
            module.awq_lite.act_scale = module.awq_lite.act_scale / module.awq_lite.num_cache_steps

            has_nan_local = torch.any(torch.isnan(module.awq_lite.act_scale)) or torch.any(
                torch.isnan(module.awq_lite.weight_scale)
            )
            has_nan = DistributedProcessGroup.get_dist_syncd_obj(
                has_nan_local, module.parallel_state.data_parallel_group, lambda objs: any(objs)
            )

            if has_nan:
                module.awq_lite.is_enabled = False
            else:
                sync_act_scale_across_dp(
                    module,
                    module.parallel_state.data_parallel_group,
                )

    # Disable AWQ search for uncalibrated experts (num_cache_steps == 0) to
    # prevent get_scale() crash on float act_scale. Max calibration and neutral
    # pre_quant_scale are applied in the postprocessing loop below.
    for name, module in model.named_modules():
        if (
            is_awq_linear(module)
            and hasattr(module, "awq_lite")
            and module.awq_lite.num_cache_steps == 0
        ):
            module.awq_lite.is_enabled = False

    AWQLiteHelper.cache_mode = False
    print_rank_0("awq_lite: Searching parameters...")
    with torch.no_grad():
        forward_loop(model)

    def postprocess(module, name):
        update_best_params(module)
        if hasattr(module.weight_quantizer, "_pre_quant_scale"):
            delattr(module.weight_quantizer, "_pre_quant_scale")
        if hasattr(module.input_quantizer, "_pre_quant_scale"):
            delattr(module.input_quantizer, "_pre_quant_scale")
        if module.awq_lite.is_input_quantized:
            if module.input_quantizer.amax is not None:
                act_amax = module.input_quantizer.amax
                # TODO: make this a buffer after we support only heterogeneous checkpointing for MCore
                module.input_quantizer._amax_for_smoothing = act_amax.cpu()
                module.input_quantizer.reset_amax()
                module.input_quantizer.axis = None
                module.input_quantizer.amax = act_amax.amax()
                module.input_quantizer.enable()
            # for dynamic quantization, there is no amax, so we just enable the quantizer
            else:
                module.input_quantizer.enable()

        if module.awq_lite.is_enabled:
            apply_pre_quant_scale_and_smooth(module, 1.0 / module.awq_lite.best_scale)
        else:
            warnings.warn(f"awq_lite: Disabling for {name}, quantizing with max calibration.")
            max_calibrate(module, lambda module: module.weight_quantizer(module.weight))

    def restore_input_pre_quant_scale(module, pre_quant_scale):
        module.input_quantizer._enable_pre_quant_scale = True
        module.input_quantizer.pre_quant_scale = pre_quant_scale

    for name, module in model.named_modules():
        if hasattr(module, "awq_lite"):
            if module.awq_lite.num_cache_steps == 0:
                # Uncalibrated expert: max calibrate weights and apply neutral
                # (all-ones) pre_quant_scale for export consistency.
                # NOTE: ones_scale must be registered OUTSIDE enable_weight_access_and_writeback
                # because HF accelerate post_forward drops newly-registered submodule buffers.
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    max_calibrate(module, lambda module: module.weight_quantizer(module.weight))
                    w_shape, w_dtype, w_device = (
                        module.weight.shape[1],
                        module.weight.dtype,
                        module.weight.device,
                    )
                module.input_quantizer._enable_pre_quant_scale = True
                module.input_quantizer.pre_quant_scale = torch.ones(
                    w_shape,
                    dtype=w_dtype,
                    device=w_device,
                )
            else:
                pre_quant_scale_to_restore = None
                w_shape, w_dtype, w_device = None, None, None
                if module.awq_lite.num_search_steps == 0:
                    module.awq_lite.is_enabled = False
                    warnings.warn(
                        "awq_lite: Calling `forward_loop(model)` the second time did not forward"
                        f" data through the {name}. Please provide a valid `forward_loop` function"
                        " that can be used to forward data through the model many times."
                    )
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    w_shape, w_dtype, w_device = (
                        module.weight.shape[1],
                        module.weight.dtype,
                        module.weight.device,
                    )
                    postprocess(module, name)
                    if hasattr(module.input_quantizer, "_pre_quant_scale"):
                        pre_quant_scale_to_restore = (
                            module.input_quantizer._pre_quant_scale.detach().clone()
                        )

                # Some HF accelerate/offload hooks drop buffers registered inside
                # weight-access contexts. Re-register the input-side AWQ scale
                # outside that context so export sees NVFP4_AWQ rather than NVFP4.
                if pre_quant_scale_to_restore is None:
                    if module.awq_lite.is_enabled and module.awq_lite.best_scale is not None:
                        pre_quant_scale_to_restore = (1.0 / module.awq_lite.best_scale).to(
                            dtype=w_dtype,
                            device=w_device,
                        )
                    else:
                        pre_quant_scale_to_restore = torch.ones(
                            w_shape,
                            dtype=w_dtype,
                            device=w_device,
                        )
                restore_input_pre_quant_scale(module, pre_quant_scale_to_restore)

            module.awq_lite.cleanup()
            if not debug:
                delattr(module, "awq_lite")


@torch.no_grad()
def awq_clip(
    model: nn.Module,
    forward_loop: ForwardLoop,
    max_co_batch_size: int = 1024,
    max_tokens_per_batch: int = 64,
    min_clip_ratio: float = 0.5,
    shrink_step: float = 0.05,
    debug: bool = False,
    **kwargs,
):
    """AWQ-Clip variant.

    Args:
        model: Model to calibrate.
        forward_loop: A callable that runs the forward pass of the model.

    See :class:`AWQClipCalibConfig <modelopt.torch.quantization.config.AWQClipCalibConfig>` for
    details on the remaining arguments.
    """
    assert forward_loop is not None, "forward_loop must be provided for awq_clip"

    class AWQClipHelper:
        def __init__(self, module):
            self.num_tokens = 0
            self.block_size = _get_awq_quantizer_block_size(module.weight, module.weight_quantizer)

            # Cache the original amax
            module.weight_quantizer.reset_amax()
            enable_stats_collection(module.weight_quantizer)
            module.weight_quantizer(module.weight)
            finish_stats_collection(module.weight_quantizer)
            self.w_amax = module.weight_quantizer.amax.clone()

            co, ci = module.weight.shape
            clip_ratios = [
                round(float(k), 2) for k in torch.arange(min_clip_ratio, 1.0, shrink_step)
            ] + [1.0]
            if self.is_per_tensor_clip(module):
                self.loss = {k: torch.tensor(0.0, device=module.weight.device) for k in clip_ratios}
            else:
                self.loss = {
                    k: torch.zeros(
                        (co, math.ceil(ci / self.block_size)),
                        device=module.weight.device,
                    )
                    for k in clip_ratios
                }
            self.best_clip_val = None
            self.best_loss = None

            self.is_input_quantized = module.input_quantizer.is_enabled
            module.weight_quantizer.disable()

        def is_per_tensor_clip(self, module):
            quantizer = module.weight_quantizer
            is_dynamic_w_per_tensor = (
                hasattr(quantizer, "block_sizes")
                and quantizer.block_sizes.get("type", None) == "dynamic"
                and quantizer.axis is None
            )
            is_per_tensor = quantizer.axis is None and quantizer.block_sizes is None
            return is_dynamic_w_per_tensor or is_per_tensor

    def update_best_params(self):
        self.awq_clip.best_loss = torch.ones_like(self.awq_clip.w_amax) * float("inf")
        self.awq_clip.best_clip_val = torch.zeros_like(self.awq_clip.w_amax)

        for shrink, loss in self.awq_clip.loss.items():
            loss = loss.view_as(self.awq_clip.w_amax)
            indices = loss < self.awq_clip.best_loss
            self.awq_clip.best_loss = torch.where(indices, loss, self.awq_clip.best_loss)
            self.awq_clip.best_clip_val = torch.where(
                indices, self.awq_clip.w_amax * shrink, self.awq_clip.best_clip_val
            )

    def _clip_search(self, inputs, co_bsz=256, max_tokens=16):
        weight = self.weight
        self.weight_quantizer.enable()

        if self.awq_clip.is_per_tensor_clip(self):
            # In NVFP4, only the per-tensor amax is clipped
            out_actual = inputs @ self.weight.T
            original_amax = self.weight_quantizer.amax.clone()
            self.awq_clip.num_tokens += inputs.shape[0]
            for shrink in self.awq_clip.loss:
                self.weight_quantizer.amax = original_amax * shrink
                out = inputs @ self.weight_quantizer(self.weight).T
                loss = (out - out_actual).float().pow(2).mean()
                self.awq_clip.loss[shrink] += loss
        else:
            # weight  [co, ci] -> [co, 1, n_block, block_size]
            # inputs  [..., ci] -> [1, max_tokens, n_block, block_size]

            inputs = inputs.view(-1, inputs.shape[-1])  # _, ci
            # Select max_tokens from the total input tokens of count batch * n_token
            inputs = inputs[0 :: max(1, inputs.shape[0] // max_tokens)]  # max_tokens, ci
            self.awq_clip.num_tokens += inputs.shape[0]

            block_size = self.awq_clip.block_size
            co, ci = weight.shape
            if ci % block_size != 0:
                weight = F.pad(weight, (0, block_size - ci % block_size), "constant", 0)
                inputs = F.pad(inputs, (0, block_size - ci % block_size), "constant", 0)
                ci = weight.shape[-1]

            weight = weight.reshape(co, 1, -1, block_size)  # co, 1, n_block, block_size

            # 1, max_tokens, n_block, block_size
            inputs = inputs.reshape(1, inputs.shape[0], -1, block_size)

            for co_batch in range(math.ceil(co / co_bsz)):
                w = weight[co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)]

                org_out = (inputs * w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                for shrink in self.awq_clip.loss:
                    self.weight_quantizer.amax = self.awq_clip.w_amax * shrink
                    quantized_clipped_weight = self.weight_quantizer(self.weight)
                    cur_w = quantized_clipped_weight[
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ]
                    if cur_w.shape[-1] % block_size != 0:
                        cur_w = F.pad(
                            cur_w,
                            (0, block_size - cur_w.shape[-1] % block_size),
                            "constant",
                            0,
                        )
                    cur_w = cur_w.reshape(w.shape)
                    cur_out = (inputs * cur_w).sum(dim=-1)  # co_bsz, max_tokens, n_block

                    # co_bsz, n_block
                    loss = (cur_out - org_out).float().pow(2).mean(dim=1)

                    parallel_group = self.parallel_state.data_parallel_group
                    if parallel_group.is_initialized():
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=parallel_group.group)
                        loss /= parallel_group.world_size()

                    del cur_out, cur_w
                    self.awq_clip.loss[shrink][
                        co_batch * co_bsz : min((co_batch + 1) * co_bsz, co)
                    ] += loss
                del org_out

    def forward(name, self, input, *args, **kwargs):
        # input shape : (..., cin)
        # weight shape : (cout, cin)
        if self.awq_clip.is_input_quantized:
            self.input_quantizer.enable()
            max_calibrate(self.input_quantizer, lambda input_quantizer: input_quantizer(input))
            self.input_quantizer.disable()
        try:
            _clip_search(
                self,
                self.input_quantizer(input),
                max_co_batch_size,
                max_tokens_per_batch,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError(
                    f"Clip search on {name} failed due to CUDA out of memory, try reducing"
                    " max_co_batch_size"
                ) from e
            raise RuntimeError(e)

        # Disable quantization
        self.weight_quantizer.disable()
        return self._forward_no_awq(input, *args, **kwargs)

    # Pre-compute name_to_module dict to avoid O(n^2) complexity in enable_weight_access_and_writeback
    name_to_module = dict(model.named_modules())
    for name, module in model.named_modules():
        if (
            is_quantized_linear(module)
            and module.weight_quantizer.is_enabled
            and module.weight_quantizer.block_sizes is not None
        ):
            bind_forward_method(module, partial(forward, name), "_forward_no_awq")
            with enable_weight_access_and_writeback(module, model, name_to_module):
                module.awq_clip = AWQClipHelper(module)

    print_rank_0("awq_clip: Estimating parameters...")
    # Lets enable stats collection
    # This will collect amax for input_quantizers and KV quantizers during the caching mode forward pass
    enable_stats_collection(model)
    forward_loop(model)
    # Call max_calibrate to load the amax values collected during the caching mode forward pass
    # This will also perform distributed amax sync for input_quantizers
    max_calibrate(model, lambda model: None)

    def postprocess(module):
        update_best_params(module)

        # Load the best clip value (amax)
        module.weight_quantizer.amax = module.awq_clip.best_clip_val
        module.weight_quantizer.enable()
        if module.awq_clip.is_input_quantized:
            module.input_quantizer.enable()

    for name, module in model.named_modules():
        if is_quantized_linear(module) and hasattr(module, "awq_clip"):
            if module.awq_clip.num_tokens > 0:
                with enable_weight_access_and_writeback(module, model, name_to_module):
                    postprocess(module)

            if not debug:
                delattr(module, "awq_clip")

            unpatch_forward_method(module, "_forward_no_awq")


def _get_awq_quantizer_block_size(tensor: torch.Tensor, quantizer: TensorQuantizer):
    if quantizer.block_sizes is None:
        return None
    if -1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[-1]
    elif 1 in quantizer.block_sizes:
        blocksize = quantizer.block_sizes[1]
    else:
        raise ValueError("AWQ requires block quantization along -1 axis")
    return blocksize


def svd(weight, rank):
    original_device = weight.device
    original_dtype = weight.dtype
    weight_f64 = weight.to(dtype=torch.float64, device=original_device)
    u, s, vt = torch.linalg.svd(weight_f64, full_matrices=False)
    us = u[:, :rank] * s[:rank]
    vt = vt[:rank]
    us = us.to(device=original_device, dtype=original_dtype)
    vt = vt.to(device=original_device, dtype=original_dtype)
    if us.shape[1] < rank or vt.shape[0] < rank:
        warnings.warn(
            "The low-rank dimensions do not match the layer dimensions. "
            "Please verify your configuration and model settings. "
            f"Rank is {us.shape[1]} and {vt.shape[0]}"
        )
        us_temp = torch.zeros((us.shape[0], rank), dtype=us.dtype, device=us.device)
        vt_temp = torch.zeros((rank, vt.shape[1]), dtype=vt.dtype, device=vt.device)
        us_temp[:, : us.shape[1]] = us
        vt_temp[: vt.shape[0], :] = vt
        us = us_temp
        vt = vt_temp
    return us, vt


@torch.no_grad()
def svdquant(
    model: nn.Module,
    forward_loop: ForwardLoop | None = None,
    lowrank: int = 32,
    **kwargs,
):
    """Lite version of SVDQuant.

    Args:
        model: Model to be calibrated.
        forward_loop: A callable which takes the model as argument and
            forwards calibration data through the model.

    See :class:`SVDQuantConfig <modelopt.torch.quantization.config.SVDQuantConfig>` for
    details on the remaining arguments.
    """

    def postprocess(module, name):
        print_rank_0(f"SVD {name}")
        weight = module.weight.data
        us, vt = svd(weight, lowrank)
        module.weight_quantizer.svdquant_lora_a = vt
        module.weight_quantizer.svdquant_lora_b = us
        module.weight.data.sub_(
            module.weight_quantizer.svdquant_lora_b @ module.weight_quantizer.svdquant_lora_a
        )
        module.weight_quantizer.reset_amax()
        module.input_quantizer.reset_amax()

    create_and_replace_svdquant_linear_on_the_fly(model=model)
    awq(model, forward_loop, "awq_lite", **kwargs)

    name_to_module = dict(model.named_modules())
    for name, module in name_to_module.items():
        if is_quantized_linear(module) and module.weight_quantizer.is_enabled:
            with enable_weight_access_and_writeback(module, model, name_to_module):
                postprocess(module, name)
    max_calibrate(model, forward_loop)


@torch.no_grad()
def layerwise_calibrate(
    model: nn.Module,
    forward_loop: ForwardLoop,
    calib_func: Callable,
    **calib_kwargs,
):
    """Layerwise calibration - a layer-by-layer calibration algorithm.

    Runs the full model forward per layer but patches decoder layers with a
    skip / run / capture strategy so that inter-layer logic in parent modules
    (e.g. mask construction) executes naturally without model-specific hooks.

    If ``checkpoint_dir`` is passed (via ``calib_kwargs``), per-layer checkpoints
    are saved after each layer completes. On restart, calibration resumes from
    the last completed layer.
    """
    checkpoint_dir = calib_kwargs.pop("checkpoint_dir", None)

    if forward_loop is None:
        raise ValueError(
            "forward_loop must not be None for layerwise calibration. "
            "Please provide a valid forward_loop callable."
        )

    transformer_layers = LayerActivationCollector.get_decoder_layers(model)
    if transformer_layers is None or len(transformer_layers) == 0:
        raise ValueError(
            "Could not find transformer layers in model. "
            "Layerwise calibration requires a model with identifiable transformer layers."
        )

    num_layers = len(transformer_layers)
    print_rank_0(f"Layerwise calibration: Found {num_layers} transformer layers")

    ckpt = _CheckpointState.from_folder(checkpoint_dir, num_layers)
    start_layer = ckpt.start_layer if ckpt else 0

    input_getter = LayerActivationCollector(model)
    input_getter._patch_all_layers(decoder_layers=transformer_layers)

    resumed_inputs = ckpt.setup_resume(transformer_layers) if ckpt and start_layer > 0 else None

    try:
        # Bootstrap: get first layer's inputs (or use resumed inputs).
        layer_inputs = input_getter.get_first_layer_inputs(
            start_layer, resumed_inputs, forward_loop
        )

        for layer_idx in range(start_layer, num_layers):
            layer = transformer_layers[layer_idx]

            def _layer_forward_loop(m, _inputs=layer_inputs):
                for args, kwargs_input in _inputs:
                    # Reset past_key_values to prevent the KV cache from
                    # accumulating across multiple forward replays (e.g.
                    # max_calibrate then Hessian collection in GPTQ).
                    # The layer doesn't need stale KV data — each replay
                    # should start with a fresh cache.
                    if (
                        "past_key_values" in kwargs_input
                        and kwargs_input["past_key_values"] is not None
                    ):
                        kwargs_input = dict(kwargs_input)
                        cache = kwargs_input["past_key_values"]
                        if hasattr(cache, "reset"):
                            cache.reset()
                        else:
                            kwargs_input["past_key_values"] = None
                    m(*args, **kwargs_input)

            with persistent_materialization(layer):
                calib_func(layer, _layer_forward_loop, **calib_kwargs)

            # Run one more forward to get next layer's inputs and set
            # output_meta on the just-calibrated layer (via "run" mode).
            is_last = layer_idx + 1 >= num_layers
            if not is_last:
                next_inputs = input_getter.cache_outputs_for_next_layer_calib(layer, forward_loop)
            else:
                next_inputs = None

            if ckpt:
                ckpt.save(layer_idx, layer, model, transformer_layers, next_inputs)

            del layer_inputs
            torch.cuda.empty_cache()
            layer_inputs = next_inputs  # noqa: F841 (used in next iteration's closure)
    finally:
        input_getter._unpatch_all_layers()

    if ckpt:
        ckpt.full_restore(transformer_layers, model)

    print_rank_0("Layerwise calibration completed")


@torch.no_grad()
def gptq(
    model: nn.Module,
    forward_loop: ForwardLoop,
    perc_damp: float = 0.01,
    block_size: int = 128,
    fused: bool = False,
):
    """GPTQ quantization.

    Works in two modes depending on ``layerwise`` in the config:

    * **Layerwise** (``layerwise=True``): ``layerwise_calibrate`` calls this
      function once per decoder layer with updated activations, producing more
      accurate Hessian estimates.
    * **Non-layerwise** (``layerwise=False``): called once on the full model.
      All layers are quantized in parallel from the original activations.

    Per-module steps:

    1. ``max_calibrate`` to set amax values from the current activations.
    2. Promote eligible quantizers to ``NVFP4StaticQuantizer`` (two-level scaling).
    3. Collect per-linear-layer Hessian matrices via forward hooks.
    4. Blockwise weight updates using the inverse Hessian to compensate for
       rounding error (the core GPTQ column-wise update).

    Args:
        model: The module to quantize — either the full model or a single decoder
            layer when invoked by ``layerwise_calibrate``.
        forward_loop: Callable that replays calibration inputs through *model*.
        perc_damp: Percentage of avg Hessian diagonal for damping (default: 0.01).
        block_size: Block size for GPTQ weight update.
        fused: If True, use fused Triton kernel for NVFP4 static quantization.
    """
    total_start = time.time()

    # TODO: Add support for other scale setting strateiges like weight-mse or local-hessian
    max_calibrate(model, forward_loop=forward_loop)
    promote_nvfp4_static_quantizers(model)

    quantized_layers = [
        (n, m)
        for n, m in model.named_modules()
        if is_quantized_linear(m) and m.weight_quantizer.is_enabled
    ]
    if not quantized_layers:
        print_rank_0("No quantized linear layers found, skipping GPTQ")
        return

    def _make_gptq_handle(name, m):
        backend = getattr(m.weight_quantizer, "backend", None)
        if backend is None:
            cls = GPTQHelper
        else:
            cls = _GPTQ_HELPER_REGISTRY.get(backend, GPTQHelper)
        return cls(m, name, offload_to_cpu=True, fused=fused)

    gptq_handles = {name: _make_gptq_handle(name, m) for name, m in quantized_layers}
    for handle in gptq_handles.values():
        handle.setup()

    print_rank_0(f"Computing Hessians for {len(gptq_handles)} linear layers...")

    with set_quantizer_by_cfg_context(
        model, [{"quantizer_name": "*weight_quantizer", "enable": False}]
    ):
        forward_loop(model)

    for handle in gptq_handles.values():
        handle.cleanup()

    print_rank_0("Updating weights using GPTQ algorithm...")
    name_to_module = dict(model.named_modules())
    for handle in gptq_handles.values():
        with enable_weight_access_and_writeback(handle.module, model, name_to_module):
            handle.update_weights(block_size, perc_damp)
        handle.free()
    del gptq_handles

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_rank_0(f"GPTQ time: {time.time() - total_start:.2f}s")
