#!/usr/bin/env python3
"""Merge local_hessian resume checkpoints collected on calibration shards."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


METHOD = "local_hessian"
STATE_NAME = f"{METHOD}_resume_state.json"
MODELOPT_NAME = f"{METHOD}_resume_modelopt_state.pth"
AUX_NAME = f"{METHOD}_resume_aux_state.pth"


def _load_resume_dir(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    state_path = path / STATE_NAME
    modelopt_path = path / MODELOPT_NAME
    missing = [str(p) for p in (state_path, modelopt_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is missing required resume files: {missing}")

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    if state.get("method") != METHOD:
        raise ValueError(f"{state_path} method={state.get('method')!r}, expected {METHOD!r}")
    if state.get("stage") not in {"after_max", "after_hessian_cache"}:
        raise ValueError(f"{state_path} stage={state.get('stage')!r}, expected 'after_max' or 'after_hessian_cache'")

    modelopt = torch.load(modelopt_path, map_location="cpu")
    aux = None
    aux_path = path / AUX_NAME
    if state.get("stage") == "after_hessian_cache":
        if not aux_path.is_file():
            raise FileNotFoundError(f"{path} is missing required resume file: {aux_path}")
        aux = torch.load(aux_path, map_location="cpu")
        if not isinstance(aux, dict) or not isinstance(aux.get("hessian_helpers"), dict):
            raise ValueError(f"{aux_path} does not contain hessian_helpers")
    return state, modelopt, aux


def _merge_tensor_state(dst: dict[str, Any], src: dict[str, Any], *, key_path: str) -> None:
    for key, src_value in src.items():
        if key not in dst:
            continue
        dst_value = dst[key]
        next_key_path = f"{key_path}.{key}" if key_path else str(key)
        if key in {"_amax", "_global_amax"}:
            if dst_value is None and isinstance(src_value, torch.Tensor):
                dst[key] = src_value.clone()
                continue
            if isinstance(dst_value, torch.Tensor) and src_value is None:
                continue
            if isinstance(dst_value, torch.Tensor) and isinstance(src_value, torch.Tensor):
                if dst_value.shape != src_value.shape:
                    raise ValueError(
                        f"{key} shape mismatch at {next_key_path}: "
                        f"{dst_value.shape} vs {src_value.shape}"
                    )
                dst[key] = torch.maximum(dst_value, src_value.to(dtype=dst_value.dtype))
                continue
            if dst_value is None and src_value is None:
                continue
        if isinstance(dst_value, torch.Tensor) and isinstance(src_value, torch.Tensor):
            continue
        if isinstance(dst_value, dict) and isinstance(src_value, dict):
            _merge_tensor_state(dst_value, src_value, key_path=next_key_path)


def _merge_modelopt_amax(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst_q = dst.get("modelopt_quantizer_state_dict")
    src_q = src.get("modelopt_quantizer_state_dict")
    if not isinstance(dst_q, dict) or not isinstance(src_q, dict):
        raise ValueError("modelopt checkpoint is missing modelopt_quantizer_state_dict")
    if set(dst_q) != set(src_q):
        raise ValueError("quantizer state key mismatch between shards")
    for name in dst_q:
        _merge_tensor_state(dst_q[name], src_q[name], key_path=name)


def _merge_hessian_aux(dst: dict[str, Any], src: dict[str, Any]) -> None:
    dst_helpers = dst["hessian_helpers"]
    src_helpers = src["hessian_helpers"]
    if set(dst_helpers) != set(src_helpers):
        raise ValueError("hessian helper key mismatch between shards")

    for name in dst_helpers:
        dst_entry = dst_helpers[name]
        src_entry = src_helpers[name]
        dst_h = dst_entry.get("hessian_per_block")
        src_h = src_entry.get("hessian_per_block")
        if not isinstance(dst_h, torch.Tensor) or not isinstance(src_h, torch.Tensor):
            raise ValueError(f"{name} is missing hessian_per_block tensor")
        if dst_h.shape != src_h.shape:
            raise ValueError(f"{name} hessian shape mismatch: {dst_h.shape} vs {src_h.shape}")
        dst_entry["hessian_per_block"] = dst_h + src_h.to(dtype=dst_h.dtype)
        dst_entry["num_samples"] = int(dst_entry.get("num_samples", 0)) + int(src_entry.get("num_samples", 0))

    dst["hessian_cache_step"] = int(dst.get("hessian_cache_step", 0)) + int(src.get("hessian_cache_step", 0))


def merge_resume_dirs(inputs: list[Path], output: Path, *, shared_after_max: bool = False) -> None:
    if len(inputs) < 1:
        raise ValueError("At least one input resume directory is required")

    states: list[dict[str, Any]] = []
    modelopts: list[dict[str, Any]] = []
    auxes: list[dict[str, Any] | None] = []
    for path in inputs:
        state, modelopt, aux = _load_resume_dir(path)
        states.append(state)
        modelopts.append(modelopt)
        auxes.append(aux)

    signature = states[0].get("signature")
    if any(state.get("signature") != signature for state in states):
        raise ValueError("resume signature mismatch; shards must use identical model/config/calibration settings")

    stages = {state.get("stage") for state in states}
    if len(stages) != 1:
        raise ValueError(f"resume stage mismatch between shards: {sorted(stages)}")
    stage = states[0].get("stage")

    merged_state = dict(states[0])
    merged_modelopt = modelopts[0]
    merged_aux = auxes[0]

    for modelopt, aux in zip(modelopts[1:], auxes[1:]):
        _merge_modelopt_amax(merged_modelopt, modelopt)
        if stage == "after_hessian_cache":
            assert merged_aux is not None and aux is not None
            _merge_hessian_aux(merged_aux, aux)

    merged_state["stage"] = stage
    merged_state["updated_at_unix"] = time.time()
    merged_state["merged_from"] = [str(path) for path in inputs]
    max_steps = [int(state.get("max_forward_step", 0)) for state in states]
    if stage == "after_hessian_cache" and shared_after_max:
        if len(set(max_steps)) != 1:
            raise ValueError(
                "--shared-after-max requires all shards to carry the same max_forward_step "
                "from a previously merged after_max checkpoint"
            )
        # Two-stage distributed local_hessian can copy the merged after_max checkpoint
        # into every Hessian shard; in that case max_forward_step is already global.
        merged_state["max_forward_step"] = max_steps[0]
    else:
        # Default: each input is an independent calibration shard, so progress is additive.
        merged_state["max_forward_step"] = sum(max_steps)
    if stage == "after_hessian_cache":
        merged_state["hessian_cache_step"] = sum(int(state.get("hessian_cache_step", 0)) for state in states)
    else:
        merged_state.pop("hessian_cache_step", None)

    output.mkdir(parents=True, exist_ok=True)
    with open(output / STATE_NAME, "w", encoding="utf-8") as f:
        json.dump(merged_state, f, indent=2, sort_keys=True)
    torch.save(merged_modelopt, output / MODELOPT_NAME)
    aux_path = output / AUX_NAME
    if stage == "after_hessian_cache":
        torch.save(merged_aux, aux_path)
    elif aux_path.exists():
        aux_path.unlink()
    print(f"Merged {len(inputs)} local_hessian {stage} resume checkpoints into {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Output merged resume checkpoint directory.")
    parser.add_argument(
        "--shared-after-max",
        action="store_true",
        help=(
            "Use only for two-stage runs where a merged after_max checkpoint was copied "
            "into every Hessian shard; keeps max_forward_step from the shared checkpoint "
            "instead of summing it."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input local_hessian resume checkpoint directories.")
    args = parser.parse_args()
    merge_resume_dirs(args.inputs, args.output, shared_after_max=args.shared_after_max)


if __name__ == "__main__":
    main()
