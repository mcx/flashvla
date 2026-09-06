"""PI0.5 checkpoint loading: one state_dict name per parameter, and every supported
file layout renames exactly onto that name set.

Policies are built on the meta device (no weights), so these tests check names
and shapes only. Value-level checks against real checkpoints are done with the
GPU/CPU verification scripts.
"""
from __future__ import annotations

import unittest

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from flashvla.policies.loading import PI05_VLM_EMBED_KEY, PI05_VLM_HEAD_KEY, remap_pi05_state_dict
from flashvla.policies.pi05.configuration_pi05 import PI05Config, PI05FlashVLAConfig
from flashvla.policies.pi05.convert_legacy_checkpoint import EXPERT_EMBED_KEY, convert_legacy_state_dict
from flashvla.policies.pi05.modeling_pi05 import PI05Policy
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy

# Distinct key patterns of lerobot/pi05_base (812 tensors); N is a layer index.
RAW_PATTERNS = """
action_in_proj.bias
action_in_proj.weight
action_out_proj.bias
action_out_proj.weight
paligemma_with_expert.gemma_expert.lm_head.weight
paligemma_with_expert.gemma_expert.model.layers.N.input_layernorm.dense.bias
paligemma_with_expert.gemma_expert.model.layers.N.input_layernorm.dense.weight
paligemma_with_expert.gemma_expert.model.layers.N.mlp.down_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.mlp.gate_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.mlp.up_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.post_attention_layernorm.dense.bias
paligemma_with_expert.gemma_expert.model.layers.N.post_attention_layernorm.dense.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.k_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.o_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.q_proj.weight
paligemma_with_expert.gemma_expert.model.layers.N.self_attn.v_proj.weight
paligemma_with_expert.gemma_expert.model.norm.dense.bias
paligemma_with_expert.gemma_expert.model.norm.dense.weight
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
time_mlp_in.bias
time_mlp_in.weight
time_mlp_out.bias
time_mlp_out.weight
""".split()
LANG_EMBEDDER_KEY = "model.prefix_embedder.lang_embedder.weight"
EXPERT_HEAD_KEY = "model.action_expert.lm_head.weight"


def raw_openpi_keys() -> list[str]:
    keys = []
    for pattern in RAW_PATTERNS:
        if ".N." not in pattern:
            keys.append(pattern)
            continue
        depth = 27 if "vision_tower" in pattern else 18
        keys.extend(pattern.replace(".N.", f".{i}.") for i in range(depth))
    return keys


def _config(config_cls):
    config = config_cls()
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


class PI05LoadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with torch.device("meta"):
            cls.policies = {
                "pi05": PI05Policy(_config(PI05Config)),
                "pi05-flashvla": PI05FlashVLAPolicy(_config(PI05FlashVLAConfig)),
            }
        cls.names = set(cls.policies["pi05"].state_dict())

    def test_every_parameter_has_exactly_one_name(self) -> None:
        for kind, policy in self.policies.items():
            with self.subTest(kind):
                all_names = [name for name, _ in policy.named_parameters(remove_duplicate=False)]
                unique = {id(p) for _, p in policy.named_parameters(remove_duplicate=False)}
                state = policy.state_dict()
                self.assertEqual(len(all_names), len(unique))
                self.assertEqual(set(all_names), set(state))  # no persistent buffers either
                self.assertNotIn(LANG_EMBEDDER_KEY, state)
                self.assertNotIn(EXPERT_EMBED_KEY, state)
                self.assertIn(PI05_VLM_EMBED_KEY, state)
                self.assertIn(PI05_VLM_HEAD_KEY, state)
                self.assertIn(EXPERT_HEAD_KEY, state)
                self.assertEqual(set(state), self.names)
                self.assertEqual(len(state), 813)

    def test_raw_openpi_layout_maps_exactly(self) -> None:
        remapped = remap_pi05_state_dict(_fake(raw_openpi_keys()))
        self.assertEqual(set(remapped), self.names)
        # The base stores the tied VLM embedding as its lm_head only.
        self.assertIs(remapped[PI05_VLM_EMBED_KEY], remapped[PI05_VLM_HEAD_KEY])

    def test_lerobot_save_pretrained_layout_maps_exactly(self) -> None:
        remapped = remap_pi05_state_dict(_fake("model." + key for key in raw_openpi_keys()))
        self.assertEqual(set(remapped), self.names)

    def test_native_keys_pass_through_unchanged(self) -> None:
        state = _fake(sorted(self.names))
        remapped = remap_pi05_state_dict(state)
        self.assertEqual(remapped.keys(), state.keys())
        for key in state:
            self.assertIs(remapped[key], state[key])

    def test_legacy_fsdp_export_converts(self) -> None:
        # Every alias as its own tensor: lang_embedder equal to embed_tokens, a
        # dead expert embedding, and a stale tied vlm.lm_head.
        state = _fake(sorted(self.names))
        state[LANG_EMBEDDER_KEY] = state[PI05_VLM_EMBED_KEY].clone()
        state[EXPERT_EMBED_KEY] = torch.full((1,), 5.0)
        state[PI05_VLM_HEAD_KEY] = torch.full((1,), -100.0)
        converted = convert_legacy_state_dict(state, tied_heads=True)
        self.assertEqual(set(converted), self.names)
        self.assertTrue(torch.equal(converted[PI05_VLM_HEAD_KEY], state[PI05_VLM_EMBED_KEY]))
        self.assertTrue(torch.equal(converted[EXPERT_HEAD_KEY], state[EXPERT_EMBED_KEY]))

        state[LANG_EMBEDDER_KEY] = torch.full((1,), 42.0)
        with self.assertRaisesRegex(ValueError, "different values"):
            convert_legacy_state_dict(state, tied_heads=True)

    def test_legacy_save_pretrained_export_converts(self) -> None:
        # safetensors kept one alphabetical-first name per tied/aliased tensor:
        # lang_embedder instead of embed_tokens or vlm.lm_head, and the expert's
        # backbone layer names instead of the joint-layer ones.
        state = _fake(sorted(self.names))
        state[LANG_EMBEDDER_KEY] = state.pop(PI05_VLM_EMBED_KEY)
        del state[PI05_VLM_HEAD_KEY]
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
        self.assertEqual(set(converted), self.names)
        self.assertTrue(torch.equal(converted[PI05_VLM_HEAD_KEY], converted[PI05_VLM_EMBED_KEY]))

    def test_legacy_headless_fsdp_export_converts(self) -> None:
        # The z-lab/flashvla-pi05-robotwin layout: joint-layer names throughout, both
        # embeddings present, neither lm_head saved. The tied model held the embedding
        # value in each head, so conversion re-creates both heads from the embeddings.
        state = _fake(sorted(self.names))
        del state[PI05_VLM_HEAD_KEY]
        del state[EXPERT_HEAD_KEY]
        state[EXPERT_EMBED_KEY] = torch.full((1,), 5.0)
        converted = convert_legacy_state_dict(state, tied_heads=True)
        self.assertEqual(set(converted), self.names)
        self.assertTrue(torch.equal(converted[PI05_VLM_HEAD_KEY], state[PI05_VLM_EMBED_KEY]))
        self.assertIs(converted[EXPERT_HEAD_KEY], state[EXPERT_EMBED_KEY])

    def test_untied_legacy_export_keeps_its_heads(self) -> None:
        state = _fake(sorted(self.names))
        state[LANG_EMBEDDER_KEY] = state[PI05_VLM_EMBED_KEY].clone()
        state[EXPERT_EMBED_KEY] = torch.full((1,), 5.0)
        converted = convert_legacy_state_dict(state, tied_heads=False)
        self.assertEqual(set(converted), self.names)
        self.assertIs(converted[PI05_VLM_HEAD_KEY], state[PI05_VLM_HEAD_KEY])
        self.assertIs(converted[EXPERT_HEAD_KEY], state[EXPERT_HEAD_KEY])


if __name__ == "__main__":
    unittest.main()
