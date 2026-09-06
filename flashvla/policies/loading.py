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

from __future__ import annotations

import re
from collections.abc import Collection, Iterable

from torch import Tensor, nn


def _module_name_by_id(root: nn.Module) -> dict[int, str]:
    """Map module identity to its first registered path under ``root``."""
    names: dict[int, str] = {}
    for name, module in root.named_modules():
        names.setdefault(id(module), name)
    return names


def _parameter_aliases(model: nn.Module) -> dict[int, list[str]]:
    """Map ``id(parameter)`` to every state_dict name it is registered under, in
    registration order."""
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    return aliases


def restore_untied_lm_head_embeddings(
    instance: nn.Module,
    mapped_sd: dict[str, Tensor],
    target_sd: dict[str, Tensor],
) -> list[str]:
    """Re-materialize input embeddings that a tied-weight checkpoint omitted.

    Mutates ``mapped_sd`` in place. For every submodule that owns both an
    ``lm_head`` and an input embedding, and whose two weights are *not* the same
    tensor object, copies the head weight into the embedding entry when the
    checkpoint supplied the head but not the embedding.

    The guard checks every name the embedding is registered under (e.g. the
    prefix embedder's ``lang_embedder`` view, which safetensors' storage dedup
    may have kept instead of ``embed_tokens``), so a checkpoint that already
    carries the embedding under any alias is left alone.

    Args:
        instance: The freshly constructed policy, before ``load_state_dict``.
        mapped_sd: Checkpoint tensors already renamed to this policy's key space.
        target_sd: ``instance.state_dict()``, used for shape validation.

    Returns:
        The embedding keys that were filled in, for logging.
    """
    module_names = _module_name_by_id(instance)
    aliases = _parameter_aliases(instance)
    restored: list[str] = []

    for name, module in instance.named_modules():
        head = getattr(module, "lm_head", None)
        if not isinstance(head, nn.Module):
            continue
        get_input_embeddings = getattr(module, "get_input_embeddings", None)
        if not callable(get_input_embeddings):
            continue
        try:
            embedding = get_input_embeddings()
        except (AttributeError, NotImplementedError):
            continue
        if not isinstance(embedding, nn.Module):
            continue

        head_weight = getattr(head, "weight", None)
        embedding_weight = getattr(embedding, "weight", None)
        if head_weight is None or embedding_weight is None:
            continue
        if head_weight is embedding_weight:
            # Still tied: load_state_dict reaches both through the one tensor.
            continue

        head_name = module_names.get(id(head))
        embedding_name = module_names.get(id(embedding))
        if head_name is None or embedding_name is None:
            continue
        head_key = f"{head_name}.weight"
        embedding_key = f"{embedding_name}.weight"

        embedding_present = any(
            alias in mapped_sd for alias in aliases.get(id(embedding_weight), [embedding_key])
        )
        if head_key not in mapped_sd or embedding_present:
            continue
        if embedding_key not in target_sd:
            continue
        if tuple(target_sd[embedding_key].shape) != tuple(mapped_sd[head_key].shape):
            continue

        mapped_sd[embedding_key] = mapped_sd[head_key].clone()
        restored.append(embedding_key)

    return restored


def find_unfilled_parameters(
    instance: nn.Module,
    mapped_sd: Collection[str],
    *,
    allow: Iterable[str] = (),
) -> list[str]:
    """Return one name per parameter tensor the checkpoint never wrote.

    Deduplicates by tensor identity. These policies register the same
    ``nn.Parameter`` under several names -- the FlashVLA joint layers alias the
    backbone decoder layers, and PaliGemma exposes the language model twice --
    so a name-based ``missing_keys`` check reports hundreds of false positives
    for tensors that were in fact filled under a different name.

    Args:
        instance: The policy after ``load_state_dict``.
        mapped_sd: The key set that was passed to ``load_state_dict``.
        allow: Parameter names that are deliberately left at their fresh
            initialization (for example adaRMS projections that a non-adaRMS
            base checkpoint cannot supply).

    Returns:
        Sorted parameter names, one per distinct unfilled tensor.
    """
    allow = set(allow)
    # remove_duplicate=False so every alias name of an aliased parameter is
    # visible: a tensor is "filled" if ANY of its names is in the checkpoint.
    # The FlashVLA joint layers alias the backbone decoder layers, and a
    # checkpoint may store a given tensor under EITHER alias (e.g. VLM layers
    # saved under the fused `model.layers.*` name while the deduped first name
    # is the un-fused `model.vlm.*` one). With the default remove_duplicate=True
    # the deduped name can miss the checkpoint's chosen alias and this check
    # falsely reports a loaded tensor as unfilled.
    named = list(instance.named_parameters(remove_duplicate=False))
    filled_ids = {id(parameter) for name, parameter in named if name in mapped_sd}
    allowed_ids = {id(parameter) for name, parameter in named if name in allow}

    unfilled: dict[int, str] = {}
    for name, parameter in named:
        identity = id(parameter)
        if identity in filled_ids or identity in allowed_ids:
            continue
        unfilled.setdefault(identity, name)
    return sorted(unfilled.values())


def assert_checkpoint_covers_parameters(
    instance: nn.Module,
    mapped_sd: Collection[str],
    *,
    source: str,
    allow: Iterable[str] = (),
) -> None:
    """Fail loudly when a checkpoint leaves parameters at random initialization.

    ``load_state_dict(strict=False)`` reports such tensors in ``missing_keys``,
    which these policies cannot check directly because of weight aliasing. This
    identity-based equivalent is safe to enforce.

    Raises:
        RuntimeError: If any parameter tensor was never written.
    """
    unfilled = find_unfilled_parameters(instance, mapped_sd, allow=allow)
    if not unfilled:
        return

    parameters = dict(instance.named_parameters())
    total = sum(parameters[name].numel() for name in unfilled)
    detail = ", ".join(
        f"{name} {tuple(parameters[name].shape)}" for name in unfilled[:10]
    )
    if len(unfilled) > 10:
        detail += f", ... (+{len(unfilled) - 10} more)"
    raise RuntimeError(
        f"Checkpoint {source!r} left {len(unfilled)} parameter tensors "
        f"({total:,} parameters) at random initialization: {detail}. "
        "Every trainable tensor must come from the checkpoint; pass its name via "
        "`allow=` only if it is intentionally freshly initialized."
    )


# PI0.5 ------------------------------------------------------------------------
#
# Both PI0.5 policies register every parameter under exactly one state_dict
# name, so loading is lerobot's recipe: rename the openpi keys, then one strict
# ``load_state_dict``. The joint layer at depth N owns the VLM and action-expert
# sublayers of that depth side by side, which is where the two backbone layer
# stacks land. lerobot's ``save_pretrained`` nests the openpi names under
# ``model.``; the optional prefix covers both spellings. Keys this package wrote
# already carry the final names and fall through every rule unchanged.
_VLM_LAYER = r"^(?:model\.)?paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\."
_EXPERT_LAYER = r"^(?:model\.)?paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\."
PI05_KEY_RULES: tuple[tuple[str, str], ...] = (
    (_VLM_LAYER + r"self_attn\.", r"model.layers.\1.self_attn.vlm_attention."),
    (_VLM_LAYER + r"mlp\.", r"model.layers.\1.mlp.vlm_mlp."),
    (_VLM_LAYER + r"(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.0."),
    (_EXPERT_LAYER + r"self_attn\.", r"model.layers.\1.self_attn.action_expert_attention."),
    (_EXPERT_LAYER + r"mlp\.", r"model.layers.\1.mlp.action_expert_mlp."),
    (_EXPERT_LAYER + r"(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.1."),
    (r"^(?:model\.)?paligemma_with_expert\.paligemma\.", "model.vlm."),
    (r"^(?:model\.)?paligemma_with_expert\.gemma_expert\.", "model.action_expert."),
    (r"^(?:model\.)?(action_in_proj|time_mlp_in|time_mlp_out)\.", r"model.suffix_embedder.\1."),
    (r"^(?:model\.)?action_out_proj\.", "model.action_out_proj."),
)
PI05_VLM_EMBED_KEY = "model.vlm.model.language_model.embed_tokens.weight"
PI05_VLM_HEAD_KEY = "model.vlm.lm_head.weight"


def remap_pi05_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename an openpi/lerobot PI0.5 checkpoint into this package's key space."""
    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        for pattern, replacement in PI05_KEY_RULES:
            key, count = re.subn(pattern, replacement, key, count=1)
            if count:
                break
        remapped[key] = value
    # openpi ties the VLM input embedding to its lm_head and stores the head only.
    if PI05_VLM_EMBED_KEY not in remapped and PI05_VLM_HEAD_KEY in remapped:
        remapped[PI05_VLM_EMBED_KEY] = remapped[PI05_VLM_HEAD_KEY]
    return remapped


def load_pi05_checkpoint(model: nn.Module, model_file: str) -> None:
    """Fill a PI0.5 policy from ``model.safetensors``: rename, then one strict load.

    The file is read on the CPU (every rank of a distributed job shares one
    default GPU at this point); the caller moves the policy afterwards.
    """
    from safetensors.torch import load_file

    state_dict = remap_pi05_state_dict(load_file(model_file))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        if any(key.startswith(_PRE_ALIAS_FREE_MARKERS) for key in state_dict):
            raise RuntimeError(
                f"{model_file} was saved before the alias-free PI0.5 layout; convert it once with "
                "`python -m flashvla.policies.pi05.convert_legacy_checkpoint <src_dir> <dst_dir>`"
            ) from error
        raise


# Names only a checkpoint from before the alias-free layout can contain.
_PRE_ALIAS_FREE_MARKERS = (
    "model.prefix_embedder.lang_embedder.",
    "model.action_expert.model.embed_tokens.",
    "model.action_expert.model.layers.",
    "model.vlm.model.language_model.layers.",
)


__all__ = [
    "PI05_KEY_RULES",
    "PI05_VLM_EMBED_KEY",
    "PI05_VLM_HEAD_KEY",
    "remap_pi05_state_dict",
    "load_pi05_checkpoint",
    "assert_checkpoint_covers_parameters",
    "find_unfilled_parameters",
    "restore_untied_lm_head_embeddings",
]
