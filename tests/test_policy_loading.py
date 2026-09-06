"""PI0 / PI0.5 checkpoint loading: one state_dict name per parameter, and every
supported file layout renames exactly onto that name set.

Policies are built on the meta device (no weights), so these tests check names
and shapes only. Value-level checks against real checkpoints are done with the
GPU/CPU verification scripts.
"""
from __future__ import annotations

import re
import unittest

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from flashvla.policies.loading import (
    LEROBOT_KEY_RULES,
    PI0_KEY_RULES,
    PI05_KEY_RULES,
    VLM_EMBED_KEY,
    VLM_HEAD_KEY,
    remap_state_dict,
    to_lerobot_state_dict,
)
from flashvla.policies.pi0.configuration_pi0 import PI0Config, PI0FlashVLAConfig
from flashvla.policies.pi0.modeling_pi0 import PI0Policy
from flashvla.policies.pi0.modeling_pi0_flashvla import (
    _ADARMS_FRESH_KEYS,
    _ADARMS_INCOMPATIBLE_KEYS,
    PI0FlashVLAPolicy,
)
from flashvla.policies.pi05.configuration_pi05 import PI05Config, PI05FlashVLAConfig
from flashvla.policies.pi05.convert_legacy_checkpoint import EXPERT_EMBED_KEY, convert_legacy_state_dict
from flashvla.policies.pi05.modeling_pi05 import PI05Policy
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

# Distinct key patterns of lerobot/pi05_base (812 tensors) and lerobot/pi0_base
# (777 tensors); N is a layer index. Both store the VLM embedding as its lm_head only.
_PALIGEMMA_PATTERNS = """
paligemma_with_expert.paligemma.lm_head.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.input_layernorm.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.mlp.down_proj.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.mlp.gate_proj.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.mlp.up_proj.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.post_attention_layernorm.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.self_attn.k_proj.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.self_attn.o_proj.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.self_attn.q_proj.weight
paligemma_with_expert.paligemma.model.language_model.layers.N.self_attn.v_proj.weight
paligemma_with_expert.paligemma.model.language_model.norm.weight
paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias
paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.layer_norm1.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.layer_norm1.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.layer_norm2.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.layer_norm2.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.mlp.fc1.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.mlp.fc1.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.mlp.fc2.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.mlp.fc2.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.k_proj.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.k_proj.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.out_proj.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.out_proj.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.q_proj.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.q_proj.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.v_proj.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.N.self_attn.v_proj.weight
paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias
paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight
paligemma_with_expert.gemma_expert.lm_head.weight
paligemma_with_expert.gemma_expert.model.layers.N.mlp.down_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.mlp.gate_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.mlp.up_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.k_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.o_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.q_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.v_proj.weight
action_in_proj.bias
action_in_proj.weight
action_out_proj.bias
action_out_proj.weight
""".split()
RAW_PI05_PATTERNS = _PALIGEMMA_PATTERNS + """
paligemma_with_expert.gemma_expert.model.layers.N.input_layernorm.dense.bias
paligemma_with_expert.gemma_expert.model.layers.N.input_layernorm.dense.weight
paligemma_with_expert.gemma_expert.model.layers.N.post_attention_layernorm.dense.bias
paligemma_with_expert.gemma_expert.model.layers.N.post_attention_layernorm.dense.weight
paligemma_with_expert.gemma_expert.model.norm.dense.bias
paligemma_with_expert.gemma_expert.model.norm.dense.weight
time_mlp_in.bias
time_mlp_in.weight
time_mlp_out.bias
time_mlp_out.weight
""".split()
RAW_PI0_PATTERNS = _PALIGEMMA_PATTERNS + """
paligemma_with_expert.gemma_expert.model.layers.N.input_layernorm.weight
paligemma_with_expert.gemma_expert.model.layers.N.post_attention_layernorm.weight
paligemma_with_expert.gemma_expert.model.norm.weight
action_time_mlp_in.bias
action_time_mlp_in.weight
action_time_mlp_out.bias
action_time_mlp_out.weight
state_proj.bias
state_proj.weight
""".split()
LANG_EMBEDDER_KEY = "model.prefix_embedder.lang_embedder.weight"
EXPERT_HEAD_KEY = "model.action_expert.lm_head.weight"


def expand(patterns) -> list[str]:
    keys = []
    for pattern in patterns:
        if ".N." not in pattern:
            keys.append(pattern)
            continue
        depth = 27 if "vision_tower" in pattern else 18
        keys.extend(pattern.replace(".N.", f".{i}.") for i in range(depth))
    return keys


def _config(config_cls, **overrides):
    config = config_cls(**overrides)
    config.input_features = {
        f"observation.images.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for cam in ("a", "b", "c")
    }
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    config.device = "cpu"
    config.compile_model = False
    return config


def _fake(keys) -> dict[str, torch.Tensor]:
    return {key: torch.full((1,), float(i)) for i, key in enumerate(keys)}


def _matches(key, patterns) -> bool:
    return any(re.match(p, key) for p in patterns)


class PolicyLoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with torch.device("meta"):
            cls.policies = {
                "pi05": PI05Policy(_config(PI05Config)),
                "pi05-flashvla": PI05FlashVLAPolicy(_config(PI05FlashVLAConfig)),
                "pi0": PI0Policy(_config(PI0Config)),
                "pi0-flashvla": PI0FlashVLAPolicy(_config(PI0FlashVLAConfig)),
                "pi0-flashvla-adarms": PI0FlashVLAPolicy(_config(PI0FlashVLAConfig, use_adarms_time_cond=True)),
            }
        cls.names = {kind: set(policy.state_dict()) for kind, policy in cls.policies.items()}

    def test_every_parameter_has_exactly_one_name(self) -> None:
        for kind, policy in self.policies.items():
            with self.subTest(kind):
                all_names = [name for name, _ in policy.named_parameters(remove_duplicate=False)]
                unique = {id(p) for _, p in policy.named_parameters(remove_duplicate=False)}
                state = set(policy.state_dict())
                self.assertEqual(len(all_names), len(unique))
                self.assertEqual(set(all_names), state)  # no persistent buffers either
                self.assertNotIn(LANG_EMBEDDER_KEY, state)
                self.assertNotIn(EXPERT_EMBED_KEY, state)
                for key in (VLM_EMBED_KEY, VLM_HEAD_KEY, EXPERT_HEAD_KEY):
                    self.assertIn(key, state)
        self.assertEqual(self.names["pi05"], self.names["pi05-flashvla"])
        self.assertEqual(self.names["pi0"], self.names["pi0-flashvla"])
        self.assertEqual(len(self.names["pi05"]), 813)
        self.assertEqual(len(self.names["pi0"]), 778)

    def test_raw_openpi_layouts_map_exactly(self) -> None:
        for kind, rules, patterns in (
            ("pi05", PI05_KEY_RULES, RAW_PI05_PATTERNS),
            ("pi0", PI0_KEY_RULES, RAW_PI0_PATTERNS),
        ):
            with self.subTest(kind):
                raw = _fake(expand(patterns))
                remapped = remap_state_dict(raw, rules)
                self.assertEqual(set(remapped), self.names[kind])
                self.assertIs(remapped[VLM_EMBED_KEY], remapped[VLM_HEAD_KEY])
                # lerobot's save_pretrained nests the same names under `model.`
                prefixed = remap_state_dict(_fake("model." + key for key in raw), rules)
                self.assertEqual(set(prefixed), self.names[kind])

    def test_native_keys_pass_through_unchanged(self) -> None:
        for kind, rules in (("pi05", PI05_KEY_RULES), ("pi0", PI0_KEY_RULES), ("pi0-flashvla-adarms", PI0_KEY_RULES)):
            with self.subTest(kind):
                state = _fake(sorted(self.names[kind]))
                remapped = remap_state_dict(state, rules)
                self.assertEqual(remapped.keys(), state.keys())
                for key in state:
                    self.assertIs(remapped[key], state[key])

    def test_adarms_pi0_accounts_for_every_raw_key(self) -> None:
        # A plain pi0 base into the adaRMS expert: its norm scales are dropped, and
        # exactly the FiLM projections plus the conditioning MLPs stay fresh.
        remapped = set(remap_state_dict(_fake(expand(RAW_PI0_PATTERNS)), PI0_KEY_RULES))
        dropped = {k for k in remapped if _matches(k, _ADARMS_INCOMPATIBLE_KEYS)}
        model = self.names["pi0-flashvla-adarms"]
        fresh = {k for k in model if _matches(k, _ADARMS_FRESH_KEYS)}
        self.assertEqual(len(dropped), 18 * 2 + 1)
        self.assertEqual(remapped - dropped, model - fresh)
        self.assertEqual(len(fresh), (18 * 2 + 1) * 2 + 4 * 2)
        self.assertTrue(all(".dense." in k or "suffix_embedder" in k for k in fresh))

    def test_legacy_fsdp_export_converts(self) -> None:
        # Every alias as its own tensor: lang_embedder equal to embed_tokens, a
        # dead expert embedding, and a stale tied vlm.lm_head.
        state = _fake(sorted(self.names["pi05"]))
        state[LANG_EMBEDDER_KEY] = state[VLM_EMBED_KEY].clone()
        state[EXPERT_EMBED_KEY] = torch.full((1,), 5.0)
        state[VLM_HEAD_KEY] = torch.full((1,), -100.0)
        converted = convert_legacy_state_dict(state, tied_heads=True)
        self.assertEqual(set(converted), self.names["pi05"])
        self.assertTrue(torch.equal(converted[VLM_HEAD_KEY], state[VLM_EMBED_KEY]))
        self.assertTrue(torch.equal(converted[EXPERT_HEAD_KEY], state[EXPERT_EMBED_KEY]))

        state[LANG_EMBEDDER_KEY] = torch.full((1,), 42.0)
        with self.assertRaisesRegex(ValueError, "different values"):
            convert_legacy_state_dict(state, tied_heads=True)

    def test_legacy_save_pretrained_export_converts(self) -> None:
        # safetensors kept one alphabetical-first name per tied/aliased tensor:
        # lang_embedder instead of embed_tokens or vlm.lm_head, and the expert's
        # backbone layer names instead of the joint-layer ones.
        state = _fake(sorted(self.names["pi05"]))
        state[LANG_EMBEDDER_KEY] = state.pop(VLM_EMBED_KEY)
        del state[VLM_HEAD_KEY]
        for key in [k for k in state if k.startswith("model.layers.")]:
            depth, rest = key.split(".")[2], ".".join(key.split(".")[3:])
            legacy = None
            if rest.startswith("self_attn.action_expert_attention."):
                legacy = f"model.action_expert.model.layers.{depth}.self_attn." + rest.split(".", 2)[2]
            elif rest.startswith("mlp.action_expert_mlp."):
                legacy = f"model.action_expert.model.layers.{depth}.mlp." + rest.split(".", 2)[2]
            elif rest.split(".")[1] == "1":  # adaRMS norms of the expert
                norm, _, tail = rest.split(".", 2)
                legacy = f"model.action_expert.model.layers.{depth}.{norm}.{tail}"
            if legacy:
                state[legacy] = state.pop(key)
        converted = convert_legacy_state_dict(state, tied_heads=True)
        self.assertEqual(set(converted), self.names["pi05"])
        self.assertTrue(torch.equal(converted[VLM_HEAD_KEY], converted[VLM_EMBED_KEY]))

    def test_legacy_headless_fsdp_export_converts(self) -> None:
        # The z-lab/flashvla-pi05-robotwin layout: joint-layer names throughout, both
        # embeddings present, neither lm_head saved. The tied model held the embedding
        # value in each head, so conversion re-creates both heads from the embeddings.
        state = _fake(sorted(self.names["pi05"]))
        del state[VLM_HEAD_KEY]
        del state[EXPERT_HEAD_KEY]
        state[EXPERT_EMBED_KEY] = torch.full((1,), 5.0)
        converted = convert_legacy_state_dict(state, tied_heads=True)
        self.assertEqual(set(converted), self.names["pi05"])
        self.assertTrue(torch.equal(converted[VLM_HEAD_KEY], state[VLM_EMBED_KEY]))
        self.assertIs(converted[EXPERT_HEAD_KEY], state[EXPERT_EMBED_KEY])

    def test_untied_legacy_exports_keep_their_heads(self) -> None:
        # The PI0.5 baseline and both PI0 policies never tied their heads.
        for kind in ("pi05", "pi0", "pi0-flashvla-adarms"):
            with self.subTest(kind):
                state = _fake(sorted(self.names[kind]))
                state[LANG_EMBEDDER_KEY] = state[VLM_EMBED_KEY].clone()
                state[EXPERT_EMBED_KEY] = torch.full((1,), 5.0)
                converted = convert_legacy_state_dict(state, tied_heads=False)
                self.assertEqual(set(converted), self.names[kind])
                self.assertIs(converted[VLM_HEAD_KEY], state[VLM_HEAD_KEY])
                self.assertIs(converted[EXPERT_HEAD_KEY], state[EXPERT_HEAD_KEY])

    def test_lerobot_rename_round_trips(self) -> None:
        # flashvla -> lerobot -> flashvla is the identity, and the lerobot names are
        # exactly the raw openpi names under lerobot's `model.` prefix.
        for kind, rules, patterns in (
            ("pi05", PI05_KEY_RULES, RAW_PI05_PATTERNS),
            ("pi0", PI0_KEY_RULES, RAW_PI0_PATTERNS),
        ):
            with self.subTest(kind):
                native = _fake(sorted(self.names[kind]))
                exported = to_lerobot_state_dict(native)
                self.assertEqual(len(exported), len(native))
                self.assertTrue(all(k.startswith("model.") for k in exported))
                self.assertEqual(set(remap_state_dict(exported, rules)), self.names[kind])
                raw_with_embed = {"model." + k for k in expand(patterns)} | {
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                }
                self.assertEqual(set(exported), raw_with_embed)
        self.assertEqual(len(LEROBOT_KEY_RULES), 9)


if __name__ == "__main__":
    unittest.main()
