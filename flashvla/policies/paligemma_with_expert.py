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

"""The parameter tree the PI0 / PI0.5 policies share with lerobot.

lerobot's ``PI0Policy`` / ``PI05Policy`` own their weights as
``model.paligemma_with_expert.{paligemma,gemma_expert}`` plus a few top-level
projections. FlashVLA keeps exactly that tree, so a checkpoint written by either
side loads in the other without renaming. FlashVLA's joint layers, which run the
VLM and the action expert side by side, only *reference* the backbone sublayers
they combine; they own nothing, and the backbone stacks stay the single
registered owner of every decoder parameter.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from lerobot.policies.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma as PaliGemmaForConditionalGeneration,
    PiGemmaForCausalLM as GemmaForCausalLM,
)
from torch import nn


class PaliGemmaWithExpert(nn.Module):
    """``paligemma`` (PaliGemma VLM) and ``gemma_expert`` (Gemma action expert)."""

    def __init__(self, vlm_config, action_expert_config):
        super().__init__()
        self.paligemma = PaliGemmaForConditionalGeneration(vlm_config)
        self.gemma_expert = GemmaForCausalLM(action_expert_config)
        # As in lerobot: the action expert never embeds tokens. Both lm_heads
        # stay untied, unused parameters that the openpi checkpoint provides.
        self.gemma_expert.model.embed_tokens = None


class ProjectionRefs(nn.Module):
    """A module that computes with projections owned by the model.

    The projections live on the model under lerobot's names (``action_in_proj``,
    ``time_mlp_in``, ...). Attribute access falls through to them, so
    ``self.action_in_proj(x)`` works here without registering a second owner.
    """

    def __init__(self, projections: Mapping[str, nn.Module]):
        super().__init__()
        self.__dict__["_projections"] = dict(projections)

    def __getattr__(self, name: str):
        projections = self.__dict__.get("_projections", {})
        if name in projections:
            return projections[name]
        return super().__getattr__(name)


def decoder_linear_groups(layers: Iterable[nn.Module]) -> list[list[nn.Module]]:
    """One FSDP2 communication group per depth.

    A joint layer's forward calls the q/k/v/o and gate/up/down projections of
    both backbones directly, so those Linear modules are what FSDP2 must hook.
    Grouping them per depth all-gathers each layer's weights in one collective,
    exactly as wrapping the joint layer itself used to.
    """
    groups: list[list[nn.Module]] = []
    for layer in layers:
        modules: list[nn.Module] = []
        for attention in (layer.self_attn.vlm_attention, layer.self_attn.action_expert_attention):
            modules += [attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj]
        for mlp in (layer.mlp.vlm_mlp, layer.mlp.action_expert_mlp):
            modules += [mlp.gate_proj, mlp.up_proj, mlp.down_proj]
        groups.append(modules)
    return groups


__all__ = ["PaliGemmaWithExpert", "ProjectionRefs", "decoder_linear_groups"]
