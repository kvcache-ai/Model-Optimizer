# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from safetensors.torch import save_file

from modelopt.torch.export.model_config import QUANTIZATION_NONE
from modelopt.torch.export.quant_utils import postprocess_state_dict
from modelopt.torch.export.unified_export_hf import (
    _restore_excluded_source_tensors_from_checkpoint,
)


def test_postprocess_state_dict_preserves_compressed_tensors_weight_shape():
    """compressed_tensors INT4 weights need weight_shape metadata for lossless loading."""
    state_dict = {
        "model.layers.1.mlp.experts.0.gate_proj.weight_packed": torch.ones(
            2048, 896, dtype=torch.int32
        ),
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale": torch.ones(
            2048, 224, dtype=torch.bfloat16
        ),
        "model.layers.1.mlp.experts.0.gate_proj.weight_shape": torch.tensor(
            [2048, 7168], dtype=torch.int32
        ),
        "model.layers.1.mlp.experts.0.gate_proj.weight_quantizer._amax": torch.tensor(1.0),
    }

    processed = postprocess_state_dict(state_dict, maxbound=448.0, quantization=QUANTIZATION_NONE)

    assert "model.layers.1.mlp.experts.0.gate_proj.weight_packed" in processed
    assert "model.layers.1.mlp.experts.0.gate_proj.weight_scale" in processed
    assert "model.layers.1.mlp.experts.0.gate_proj.weight_shape" in processed
    assert "model.layers.1.mlp.experts.0.gate_proj.weight_quantizer._amax" not in processed
    torch.testing.assert_close(
        processed["model.layers.1.mlp.experts.0.gate_proj.weight_shape"],
        torch.tensor([2048, 7168], dtype=torch.int32),
    )


def test_restore_excluded_source_tensors_preserves_original_bias_dtype(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "model.safetensors"
    source_bias = torch.tensor([0.125, -0.5], dtype=torch.float32)
    source_shape = torch.tensor([2048, 7168], dtype=torch.int32)
    save_file(
        {
            "language_model.model.layers.60.mlp.gate.e_score_correction_bias": source_bias,
            "language_model.model.layers.60.mlp.experts.0.gate_proj.weight_shape": source_shape,
            "language_model.model.layers.5.mlp.gate.e_score_correction_bias": torch.tensor(
                [1.0], dtype=torch.float32
            ),
        },
        source_path,
    )

    state_dict = {
        "language_model.model.layers.60.mlp.gate.e_score_correction_bias": source_bias.to(
            torch.bfloat16
        ),
        "language_model.model.layers.5.mlp.gate.e_score_correction_bias": torch.tensor(
            [2.0], dtype=torch.bfloat16
        ),
    }
    quant_config = {
        "quantization": {
            "exclude_modules": ["language_model.model.layers.60.*"],
        }
    }

    restored = _restore_excluded_source_tensors_from_checkpoint(
        state_dict, quant_config, source_dir
    )

    restored_bias = restored["language_model.model.layers.60.mlp.gate.e_score_correction_bias"]
    assert restored_bias.dtype == torch.float32
    torch.testing.assert_close(restored_bias, source_bias)
    torch.testing.assert_close(
        restored["language_model.model.layers.60.mlp.experts.0.gate_proj.weight_shape"],
        source_shape,
    )
    assert (
        restored["language_model.model.layers.5.mlp.gate.e_score_correction_bias"].dtype
        == torch.bfloat16
    )
