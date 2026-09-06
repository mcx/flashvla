#!/usr/bin/env python
# Copyright 2025 FlashVLA team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint key spaces of the PI0 / PI0.5 policies.

Both families register every parameter under exactly one state_dict name, so
loading is lerobot's recipe: rename the openpi keys, then one strict
``load_state_dict``. The tables below are the two directions of that rename.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from torch import Tensor, nn

# openpi / lerobot -> flashvla ---------------------------------------------------
#
# The joint layer at depth N owns the VLM and action-expert sublayers of that
# depth side by side, which is where the two backbone layer stacks land.
# lerobot's ``save_pretrained`` nests the openpi names under ``model.``; the
# optional prefix covers both spellings. Keys this package wrote already carry
# the final names and fall through every rule unchanged.
_VLM_LAYER = r"^(?:model\.)?paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\."
_EXPERT_LAYER = r"^(?:model\.)?paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\."
_BACKBONE_RULES: tuple[tuple[str, str], ...] = (
    (_VLM_LAYER + r"self_attn\.", r"model.layers.\1.self_attn.vlm_attention."),
    (_VLM_LAYER + r"mlp\.", r"model.layers.\1.mlp.vlm_mlp."),
    (_VLM_LAYER + r"(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.0."),
    (_EXPERT_LAYER + r"self_attn\.", r"model.layers.\1.self_attn.action_expert_attention."),
    (_EXPERT_LAYER + r"mlp\.", r"model.layers.\1.mlp.action_expert_mlp."),
    (_EXPERT_LAYER + r"(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.1."),
    (r"^(?:model\.)?paligemma_with_expert\.paligemma\.", "model.vlm."),
    (r"^(?:model\.)?paligemma_with_expert\.gemma_expert\.", "model.action_expert."),
    (r"^(?:model\.)?action_out_proj\.", "model.action_out_proj."),
)
PI05_KEY_RULES: tuple[tuple[str, str], ...] = _BACKBONE_RULES + (
    (r"^(?:model\.)?(action_in_proj|time_mlp_in|time_mlp_out)\.", r"model.suffix_embedder.\1."),
)
PI0_KEY_RULES: tuple[tuple[str, str], ...] = _BACKBONE_RULES + (
    (
        r"^(?:model\.)?(action_in_proj|state_proj|action_time_mlp_in|action_time_mlp_out)\.",
        r"model.suffix_embedder.\1.",
    ),
    # Older openpi pi0 exports spell the time MLP without the ``action_`` prefix.
    (r"^(?:model\.)?time_mlp_(in|out)\.", r"model.suffix_embedder.action_time_mlp_\1."),
)
VLM_EMBED_KEY = "model.vlm.model.language_model.embed_tokens.weight"
VLM_HEAD_KEY = "model.vlm.lm_head.weight"

# Names only a checkpoint from before the alias-free layout can contain.
_PRE_ALIAS_FREE_MARKERS = (
    "model.prefix_embedder.lang_embedder.",
    "model.action_expert.model.embed_tokens.",
    "model.action_expert.model.layers.",
    "model.vlm.model.language_model.layers.",
)


def _rename(state_dict: dict[str, Tensor], rules: Iterable[tuple[str, str]]) -> dict[str, Tensor]:
    rules = tuple(rules)
    renamed: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        for pattern, replacement in rules:
            key, count = re.subn(pattern, replacement, key, count=1)
            if count:
                break
        renamed[key] = value
    return renamed


def remap_state_dict(state_dict: dict[str, Tensor], rules: Iterable[tuple[str, str]]) -> dict[str, Tensor]:
    """Rename an openpi/lerobot checkpoint into this package's key space."""
    remapped = _rename(state_dict, rules)
    # openpi ties the VLM input embedding to its lm_head and stores the head only.
    if VLM_EMBED_KEY not in remapped and VLM_HEAD_KEY in remapped:
        remapped[VLM_EMBED_KEY] = remapped[VLM_HEAD_KEY]
    return remapped


def load_checkpoint(
    model: nn.Module,
    model_file: str,
    rules: Iterable[tuple[str, str]],
    *,
    drop: Iterable[str] = (),
    fresh: Iterable[str] = (),
) -> list[str]:
    """Fill ``model`` from ``model.safetensors``: rename, then one strict load.

    ``drop`` holds regexes for renamed keys the model has no place for (an
    adaRMS expert has no plain norm scales); ``fresh`` holds regexes for
    parameters the file may leave at their initialization (the projections
    such a file lacks). The names actually left fresh are returned; any other
    missing or unexpected key is an error. The file is read on the CPU (every
    rank of a distributed job shares one default GPU at this point); the
    caller moves the policy afterwards.
    """
    from safetensors.torch import load_file

    drop, fresh = tuple(drop), tuple(fresh)
    state_dict = remap_state_dict(load_file(model_file), rules)
    if drop:
        state_dict = {k: v for k, v in state_dict.items() if not any(re.match(p, k) for p in drop)}

    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not any(re.match(p, k) for p in fresh)]
    if missing or result.unexpected_keys:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)}: {missing[:8]}")
        if result.unexpected_keys:
            detail.append(f"unexpected {len(result.unexpected_keys)}: {result.unexpected_keys[:8]}")
        message = f"{model_file} does not match {type(model).__name__}: " + "; ".join(detail)
        if any(key.startswith(_PRE_ALIAS_FREE_MARKERS) for key in state_dict):
            message += (
                ". The file was saved before the alias-free layout; convert it once with "
                "`python -m flashvla.policies.pi05.convert_legacy_checkpoint <src_dir> <dst_dir>`"
            )
        raise RuntimeError(message)
    return list(result.missing_keys)


def load_pi05_checkpoint(model: nn.Module, model_file: str) -> None:
    load_checkpoint(model, model_file, PI05_KEY_RULES)


def load_pi0_checkpoint(
    model: nn.Module, model_file: str, *, drop: Iterable[str] = (), fresh: Iterable[str] = ()
) -> list[str]:
    return load_checkpoint(model, model_file, PI0_KEY_RULES, drop=drop, fresh=fresh)


# flashvla -> lerobot 0.5.1 --------------------------------------------------------
#
# The inverse of the tables above, producing the ``model.``-prefixed layout that
# lerobot's own ``save_pretrained`` writes. Whether the result is complete for a
# given lerobot policy is the exporter's job (see export_lerobot_checkpoint).
_JOINT = r"^model\.layers\.(\d+)\."
_PALIGEMMA = "model.paligemma_with_expert.paligemma."
_EXPERT = "model.paligemma_with_expert.gemma_expert."
LEROBOT_KEY_RULES: tuple[tuple[str, str], ...] = (
    (_JOINT + r"self_attn\.vlm_attention\.", _PALIGEMMA + r"model.language_model.layers.\1.self_attn."),
    (_JOINT + r"mlp\.vlm_mlp\.", _PALIGEMMA + r"model.language_model.layers.\1.mlp."),
    (
        _JOINT + r"(input_layernorm|post_attention_layernorm)\.0\.",
        _PALIGEMMA + r"model.language_model.layers.\1.\2.",
    ),
    (_JOINT + r"self_attn\.action_expert_attention\.", _EXPERT + r"model.layers.\1.self_attn."),
    (_JOINT + r"mlp\.action_expert_mlp\.", _EXPERT + r"model.layers.\1.mlp."),
    (_JOINT + r"(input_layernorm|post_attention_layernorm)\.1\.", _EXPERT + r"model.layers.\1.\2."),
    (r"^model\.vlm\.", _PALIGEMMA),
    (r"^model\.action_expert\.", _EXPERT),
    (r"^model\.suffix_embedder\.", "model."),
)


def to_lerobot_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename a flashvla PI0 / PI0.5 state dict into lerobot 0.5.1's key space."""
    return _rename(state_dict, LEROBOT_KEY_RULES)


__all__ = [
    "PI05_KEY_RULES",
    "PI0_KEY_RULES",
    "LEROBOT_KEY_RULES",
    "VLM_EMBED_KEY",
    "VLM_HEAD_KEY",
    "remap_state_dict",
    "load_checkpoint",
    "load_pi05_checkpoint",
    "load_pi0_checkpoint",
    "to_lerobot_state_dict",
]
