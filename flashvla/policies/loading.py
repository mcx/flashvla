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

"""Checkpoint loading for the PI0 / PI0.5 policies.

The policies use lerobot's parameter tree, so a lerobot or FlashVLA checkpoint
loads with one strict ``load_state_dict``. Only openpi's raw exports (the
``lerobot/pi0_base`` and ``lerobot/pi05_base`` files) need the same few fixes
lerobot's ``_fix_pytorch_state_dict_keys`` applies.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from torch import Tensor, nn

PALIGEMMA = "model.paligemma_with_expert.paligemma."
GEMMA_EXPERT = "model.paligemma_with_expert.gemma_expert."
VLM_EMBED_KEY = PALIGEMMA + "model.language_model.embed_tokens.weight"
VLM_HEAD_KEY = PALIGEMMA + "lm_head.weight"
EXPERT_HEAD_KEY = GEMMA_EXPERT + "lm_head.weight"

_EXPERT_NORM_SCALE = re.compile(
    r"^paligemma_with_expert\.gemma_expert\.model\.(layers\.\d+\.(input_layernorm|post_attention_layernorm)|norm)\.weight$"
)
# Names only a FlashVLA checkpoint from before the lerobot tree can contain.
_PRE_LEROBOT_TREE_MARKERS = (
    "model.layers.",
    "model.vlm.",
    "model.action_expert.",
    "model.suffix_embedder.",
    "model.prefix_embedder.",
)


def fix_openpi_state_dict(
    state_dict: dict[str, Tensor], kind: str, *, adarms_expert: bool = False
) -> dict[str, Tensor]:
    """Bring a raw openpi export into lerobot's ``model.``-prefixed key space.

    Mirrors lerobot's ``_fix_pytorch_state_dict_keys``: keys that already carry
    the ``model.`` prefix (lerobot or FlashVLA saves) pass through untouched.
    Raw keys get the prefix; PI0.5 reads openpi's ``action_time_mlp_*`` as
    ``time_mlp_*`` and has no ``state_proj``; PI0 reads ``time_mlp_*`` as
    ``action_time_mlp_*``; an adaRMS action expert has no plain norm scales, so
    those are skipped. openpi ties the VLM input embedding to its lm_head and
    stores the head only, so the embedding is copied from it when absent.
    """
    if kind not in ("pi0", "pi05"):
        raise ValueError(f"kind must be 'pi0' or 'pi05', got {kind!r}")
    fixed: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            if kind == "pi05":
                if key.startswith("state_proj."):
                    continue
                key = re.sub(r"^action_time_mlp_(in|out)\.", r"time_mlp_\1.", key)
            else:
                key = re.sub(r"^time_mlp_(in|out)\.", r"action_time_mlp_\1.", key)
            if adarms_expert and _EXPERT_NORM_SCALE.match(key):
                continue
            key = "model." + key
        fixed[key] = value
    if VLM_EMBED_KEY not in fixed and VLM_HEAD_KEY in fixed:
        fixed[VLM_EMBED_KEY] = fixed[VLM_HEAD_KEY]
    return fixed


def load_checkpoint(
    model: nn.Module,
    model_file: str,
    *,
    kind: str,
    adarms_expert: bool = False,
    fresh: Iterable[str] = (),
) -> list[str]:
    """Fill ``model`` from ``model.safetensors`` with one strict load.

    ``fresh`` holds regexes for parameters the file may leave at their
    initialization (the adaRMS projections a plain PI0 base cannot provide);
    the names actually left fresh are returned. Any other missing or
    unexpected key is an error. The file is read on the CPU (every rank of a
    distributed job shares one default GPU at this point); the caller moves
    the policy afterwards.
    """
    from safetensors.torch import load_file

    fresh = tuple(fresh)
    state_dict = fix_openpi_state_dict(load_file(model_file), kind, adarms_expert=adarms_expert)
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not any(re.match(p, k) for p in fresh)]
    if missing or result.unexpected_keys:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)}: {missing[:8]}")
        if result.unexpected_keys:
            detail.append(f"unexpected {len(result.unexpected_keys)}: {result.unexpected_keys[:8]}")
        message = f"{model_file} does not match {type(model).__name__}: " + "; ".join(detail)
        if any(key.startswith(_PRE_LEROBOT_TREE_MARKERS) for key in state_dict):
            message += (
                ". The file uses FlashVLA's former parameter names; convert it once with "
                "`python -m flashvla.policies.pi05.convert_legacy_checkpoint <src_dir> <dst_dir>`"
            )
        raise RuntimeError(message)
    return list(result.missing_keys)


__all__ = [
    "PALIGEMMA",
    "GEMMA_EXPERT",
    "VLM_EMBED_KEY",
    "VLM_HEAD_KEY",
    "EXPERT_HEAD_KEY",
    "fix_openpi_state_dict",
    "load_checkpoint",
]
