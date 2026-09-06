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

"""Rewrite a PI0.5 checkpoint saved before the alias-free layout.

Earlier versions of both PI0.5 policies registered several parameters under
more than one name: the joint layers aliased the backbone layer stacks, the
prefix embedder re-registered the VLM embedding as ``lang_embedder``, and
``PI05FlashVLAModel`` tied each lm_head to its embedding. Depending on the save
path a file could carry any subset of those names, and FSDP2 exports of the
tied model hold a stale ``vlm.lm_head``. The loader is now one strict
``load_state_dict``, so such files are rewritten once::

    python -m flashvla.policies.pi05.convert_legacy_checkpoint \\
        <run>/checkpoints/010000/pretrained_model <out_dir>

``out_dir`` receives the converted ``model.safetensors`` next to copies of every
other file in the source directory; passing the source directory itself
rewrites the file in place. Current checkpoints and raw openpi ones need no
conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from flashvla.policies.loading import PI05_VLM_EMBED_KEY, PI05_VLM_HEAD_KEY

_VLM_LAYER = r"^model\.vlm\.model\.language_model\.layers\.(\d+)\."
_EXPERT_LAYER = r"^model\.action_expert\.model\.layers\.(\d+)\."
LEGACY_KEY_RULES: tuple[tuple[str, str], ...] = (
    (_VLM_LAYER + r"self_attn\.", r"model.layers.\1.self_attn.vlm_attention."),
    (_VLM_LAYER + r"mlp\.", r"model.layers.\1.mlp.vlm_mlp."),
    (_VLM_LAYER + r"(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.0."),
    (_EXPERT_LAYER + r"self_attn\.", r"model.layers.\1.self_attn.action_expert_attention."),
    (_EXPERT_LAYER + r"mlp\.", r"model.layers.\1.mlp.action_expert_mlp."),
    (_EXPERT_LAYER + r"(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.1."),
    (r"^model\.prefix_embedder\.lang_embedder\.", "model.vlm.model.language_model.embed_tokens."),
)
EXPERT_EMBED_KEY = "model.action_expert.model.embed_tokens.weight"
EXPERT_HEAD_KEY = "model.action_expert.lm_head.weight"


def convert_legacy_state_dict(
    state_dict: dict[str, torch.Tensor], *, tied_heads: bool
) -> dict[str, torch.Tensor]:
    """Return the alias-free state dict for a legacy PI0.5 ``model.*`` file.

    Alias names of one parameter must agree and collapse to one entry. The
    action expert's embedding no longer exists and is dropped. With
    ``tied_heads`` (``PI05FlashVLAModel`` checkpoints) each lm_head is set to
    its embedding, the value the tied model actually held.
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for pattern, replacement in LEGACY_KEY_RULES:
            key, count = re.subn(pattern, replacement, key, count=1)
            if count:
                break
        if key in converted:
            if not torch.equal(converted[key], value):
                raise ValueError(f"alias names of {key!r} hold different values")
            continue
        converted[key] = value

    expert_embed = converted.pop(EXPERT_EMBED_KEY, None)
    if tied_heads:
        converted[PI05_VLM_HEAD_KEY] = converted[PI05_VLM_EMBED_KEY].clone()
        if expert_embed is not None:
            converted[EXPERT_HEAD_KEY] = expert_embed
    return converted


def expected_state_dict_shapes(pretrained_dir: Path) -> dict[str, torch.Size]:
    """Names and shapes the policy described by ``config.json`` loads strictly."""
    from lerobot.configs.policies import PreTrainedConfig

    from flashvla.policies.factory import get_policy_class

    policy_type = json.loads((pretrained_dir / "config.json").read_text())["type"]
    policy_cls = get_policy_class(policy_type)  # imports and registers the config class
    config = PreTrainedConfig.from_pretrained(str(pretrained_dir))
    with torch.device("meta"):
        policy = policy_cls(config)
    return {name: tensor.shape for name, tensor in policy.state_dict().items()}


def check_against_model(state_dict: dict[str, torch.Tensor], pretrained_dir: Path) -> None:
    expected = expected_state_dict_shapes(pretrained_dir)
    problems = [f"missing {name}" for name in sorted(expected.keys() - state_dict.keys())]
    problems += [f"unexpected {name}" for name in sorted(state_dict.keys() - expected.keys())]
    problems += [
        f"shape {name}: file {tuple(state_dict[name].shape)} vs model {tuple(expected[name])}"
        for name in sorted(expected.keys() & state_dict.keys())
        if state_dict[name].shape != expected[name]
    ]
    if problems:
        raise ValueError("converted checkpoint does not match the model:\n  " + "\n  ".join(problems))


def convert_pretrained_dir(src: Path, dst: Path, *, tied_heads: bool | None = None) -> None:
    config = json.loads((src / "config.json").read_text())
    if tied_heads is None:
        tied_heads = config["type"] == "pi05-flashvla"

    converted = convert_legacy_state_dict(load_file(str(src / "model.safetensors")), tied_heads=tied_heads)
    check_against_model(converted, src)

    dst.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        for item in src.iterdir():
            if item.name == "model.safetensors":
                continue
            if item.is_dir():
                shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst / item.name)
    tmp = dst / "model.safetensors.tmp"
    save_file(converted, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, dst / "model.safetensors")
    print(f"wrote {dst / 'model.safetensors'}: {len(converted)} tensors (tied_heads={tied_heads})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="legacy pretrained_model directory (config.json + model.safetensors)")
    parser.add_argument("dst", type=Path, help="output directory; may equal src to rewrite in place")
    parser.add_argument(
        "--tied-heads",
        choices=("auto", "yes", "no"),
        default="auto",
        help="whether the source model tied lm_head to embed_tokens "
        "(auto: yes for type pi05-flashvla, no for pi05)",
    )
    args = parser.parse_args()
    tied = {"auto": None, "yes": True, "no": False}[args.tied_heads]
    convert_pretrained_dir(args.src, args.dst, tied_heads=tied)


if __name__ == "__main__":
    main()
