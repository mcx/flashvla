"""The PI0 / PI0.5 policies share lerobot's parameter tree.

Every policy is built on the meta device (no weights) and compared name-for-name
with the stock lerobot policy of the same kind. The raw openpi bases go through
the same fixes lerobot applies, and FlashVLA's former layouts convert exactly.
Value-level checks against real checkpoints are done by the GPU/CPU scripts.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import draccus
import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from flashvla.distributed.fsdp import build_fsdp_module_plan
from flashvla.policies.export_lerobot_checkpoint import _lerobot_choice
from flashvla.policies.loading import EXPERT_HEAD_KEY, VLM_EMBED_KEY, VLM_HEAD_KEY, fix_openpi_state_dict, load_checkpoint
from flashvla.policies.pi0.configuration_pi0 import PI0Config, PI0FlashVLAConfig
from flashvla.policies.pi0.modeling_pi0 import PI0Policy
from flashvla.policies.pi0.modeling_pi0_flashvla import _ADARMS_FRESH_KEYS, PI0FlashVLAPolicy
from flashvla.policies.pi05.configuration_pi05 import PI05Config, PI05FlashVLAConfig
from flashvla.policies.pi05.convert_legacy_checkpoint import LEGACY_EXPERT_EMBED_KEY, convert_legacy_state_dict
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


def _lerobot_shapes(kind: str, features_from) -> dict[str, tuple[int, ...]]:
    """Stock lerobot's state_dict names and shapes for the same features."""
    if kind == "pi05":
        from lerobot.policies.pi05.configuration_pi05 import PI05Config as ConfigCls
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as PolicyCls
    else:
        from lerobot.policies.pi0.configuration_pi0 import PI0Config as ConfigCls
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as PolicyCls
    fields = {field.name for field in dataclasses.fields(ConfigCls)} - {"pretrained_path"}
    with tempfile.TemporaryDirectory() as tmp:
        features_from.save_pretrained(tmp)
        config = json.loads(Path(tmp, "config.json").read_text())
        config = {key: value for key, value in config.items() if key in fields}
        path = os.path.join(tmp, "lerobot.json")
        Path(path).write_text(json.dumps(config))
        with _lerobot_choice(kind, ConfigCls), draccus.config_type("json"):
            parsed = draccus.parse(ConfigCls, path, args=[])
    parsed.device = "meta"
    with torch.device("meta"):
        return {name: tuple(t.shape) for name, t in PolicyCls(parsed).state_dict().items()}


def _fake(keys) -> dict[str, torch.Tensor]:
    return {key: torch.full((1,), float(i)) for i, key in enumerate(keys)}


def _to_legacy_flashvla_names(keys) -> list[str]:
    """lerobot names -> FlashVLA's former single-owner names (the 813/778 layout)."""
    pali = "model.paligemma_with_expert.paligemma."
    expert = "model.paligemma_with_expert.gemma_expert."
    rules = (
        (rf"^{re.escape(pali)}model\.language_model\.layers\.(\d+)\.self_attn\.", r"model.layers.\1.self_attn.vlm_attention."),
        (rf"^{re.escape(pali)}model\.language_model\.layers\.(\d+)\.mlp\.", r"model.layers.\1.mlp.vlm_mlp."),
        (rf"^{re.escape(pali)}model\.language_model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.0."),
        (rf"^{re.escape(expert)}model\.layers\.(\d+)\.self_attn\.", r"model.layers.\1.self_attn.action_expert_attention."),
        (rf"^{re.escape(expert)}model\.layers\.(\d+)\.mlp\.", r"model.layers.\1.mlp.action_expert_mlp."),
        (rf"^{re.escape(expert)}model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.", r"model.layers.\1.\2.1."),
        (rf"^{re.escape(pali)}", "model.vlm."),
        (rf"^{re.escape(expert)}", "model.action_expert."),
        (r"^model\.(action_in_proj|time_mlp_in|time_mlp_out|state_proj|action_time_mlp_in|action_time_mlp_out)\.", r"model.suffix_embedder.\1."),
    )
    out = []
    for key in keys:
        for pattern, replacement in rules:
            key, count = re.subn(pattern, replacement, key, count=1)
            if count:
                break
        out.append(key)
    return out


class LerobotTreeTest(unittest.TestCase):
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
        cls.shapes = {kind: {k: tuple(v.shape) for k, v in p.state_dict().items()} for kind, p in cls.policies.items()}
        cls.lerobot = {"pi05": _lerobot_shapes("pi05", _config(PI05Config)), "pi0": _lerobot_shapes("pi0", _config(PI0Config))}

    def test_state_dicts_are_lerobots(self) -> None:
        for kind, reference in (("pi05", "pi05"), ("pi05-flashvla", "pi05"), ("pi0", "pi0"), ("pi0-flashvla", "pi0")):
            with self.subTest(kind):
                self.assertEqual(self.shapes[kind], self.lerobot[reference])
        self.assertEqual(len(self.lerobot["pi05"]), 813)
        self.assertEqual(len(self.lerobot["pi0"]), 778)

    def test_adarms_pi0_adds_only_the_conditioning_parameters(self) -> None:
        plain, adarms = self.shapes["pi0-flashvla"], self.shapes["pi0-flashvla-adarms"]
        dropped = set(plain) - set(adarms)
        added = set(adarms) - set(plain)
        self.assertEqual(len(dropped), 18 * 2 + 1)  # the expert's RMSNorm scales
        self.assertTrue(all(k.startswith("model.paligemma_with_expert.gemma_expert.model.") and k.endswith("norm.weight") for k in dropped))
        self.assertEqual(len(added), (18 * 2 + 1) * 2 + 4 * 2)  # FiLM dense w+b, four conditioning MLPs
        self.assertTrue(all(any(re.match(p, k) for p in _ADARMS_FRESH_KEYS) for k in added))
        self.assertEqual({k for k in adarms if any(re.match(p, k) for p in _ADARMS_FRESH_KEYS)}, added)

    def test_every_parameter_has_exactly_one_name(self) -> None:
        for kind, policy in self.policies.items():
            with self.subTest(kind):
                all_names = [name for name, _ in policy.named_parameters(remove_duplicate=False)]
                unique = {id(p) for _, p in policy.named_parameters(remove_duplicate=False)}
                self.assertEqual(len(all_names), len(unique))
                self.assertEqual(set(all_names), set(policy.state_dict()))  # no persistent buffers either
                # the joint layers own nothing; they only reference backbone sublayers
                self.assertEqual([n for n, _ in policy.model.layers.named_parameters()], [])

    def test_raw_openpi_bases_load_with_lerobots_fixes(self) -> None:
        for kind, patterns, reference in (("pi05", RAW_PI05_PATTERNS, "pi05"), ("pi0", RAW_PI0_PATTERNS, "pi0")):
            with self.subTest(kind):
                raw = _fake(expand(patterns))
                fixed = fix_openpi_state_dict(raw, kind)
                self.assertEqual(set(fixed), set(self.lerobot[reference]))
                self.assertIs(fixed[VLM_EMBED_KEY], fixed[VLM_HEAD_KEY])
                # lerobot's save_pretrained (prefixed) files pass through untouched
                prefixed = _fake("model." + key for key in expand(patterns))
                self.assertEqual(set(fix_openpi_state_dict(prefixed, kind)), set(self.lerobot[reference]))
        # adaRMS pi0: the raw expert norm scales are skipped, exactly the fresh set stays missing
        fixed = fix_openpi_state_dict(_fake(expand(RAW_PI0_PATTERNS)), "pi0", adarms_expert=True)
        adarms = set(self.shapes["pi0-flashvla-adarms"])
        self.assertTrue(set(fixed) <= adarms)
        self.assertEqual({k for k in adarms - set(fixed)}, {k for k in adarms if any(re.match(p, k) for p in _ADARMS_FRESH_KEYS)})

    def test_native_lerobot_keys_pass_through(self) -> None:
        for kind in ("pi05", "pi0"):
            state = _fake(sorted(self.lerobot[kind]))
            fixed = fix_openpi_state_dict(state, kind)
            self.assertEqual(fixed.keys(), state.keys())
            for key in state:
                self.assertIs(fixed[key], state[key])

    def test_fsdp_groups_cover_the_decoder_projections_exactly(self) -> None:
        for kind in ("pi05", "pi05-flashvla", "pi0-flashvla", "pi0-flashvla-adarms"):
            with self.subTest(kind):
                policy = self.policies[kind]
                groups = policy.fsdp_compute_groups()
                self.assertEqual(len(groups), 18)
                self.assertTrue(all(len(modules) == 14 for _, modules in groups))
                grouped = [id(p) for _, modules in groups for m in modules for p in m.parameters()]
                self.assertEqual(len(grouped), len(set(grouped)))
                projections = {
                    id(p) for name, p in policy.named_parameters()
                    if re.search(r"\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.", name)
                    and "vision_tower" not in name
                }
                self.assertEqual(set(grouped), projections)
                with torch.device("meta"):
                    policy.prepare_for_fsdp(compute_dtype=torch.bfloat16)
                plan = build_fsdp_module_plan(policy, "bf16")
                self.assertEqual(len(plan.compute_groups), 18)
                covered = {id(p) for item in [*plan.compute_modules, *plan.fp32_modules] for p in item.module.parameters()} | set(grouped)
                self.assertEqual([n for n, p in policy.named_parameters() if id(p) not in covered], [])
                fp32_names = {item.name for item in plan.fp32_modules}
                self.assertIn("model.action_out_proj", fp32_names)
                self.assertIn("model.action_in_proj", fp32_names)

    def test_legacy_flashvla_layouts_convert_to_lerobot_names(self) -> None:
        for kind in ("pi05", "pi0"):
            with self.subTest(kind):
                lerobot_names = sorted(self.lerobot[kind])
                legacy = _fake(_to_legacy_flashvla_names(lerobot_names))  # the 813/778 single-owner layout
                self.assertEqual(set(convert_legacy_state_dict(legacy, tied_heads=False)), set(lerobot_names))
                # the alias era: lang_embedder, a dead expert embedding, backbone names for the expert
                # layers and the VLM norms, and (tied model) no vlm.lm_head at all
                aliased = dict(legacy)
                aliased[LANG_EMBEDDER_KEY] = aliased.pop("model.vlm.model.language_model.embed_tokens.weight")
                del aliased["model.vlm.lm_head.weight"]
                aliased[LEGACY_EXPERT_EMBED_KEY] = torch.full((1,), 5.0)
                for key in [k for k in aliased if k.startswith("model.layers.")]:
                    depth, rest = key.split(".")[2], ".".join(key.split(".")[3:])
                    if rest.startswith("self_attn.action_expert_attention."):
                        aliased[f"model.action_expert.model.layers.{depth}.self_attn." + rest.split(".", 2)[2]] = aliased.pop(key)
                    elif rest.startswith("mlp.action_expert_mlp."):
                        aliased[f"model.action_expert.model.layers.{depth}.mlp." + rest.split(".", 2)[2]] = aliased.pop(key)
                    elif rest.split(".")[1] == "0":
                        norm, _, tail = rest.split(".", 2)
                        aliased[f"model.vlm.model.language_model.layers.{depth}.{norm}.{tail}"] = aliased.pop(key)
                converted = convert_legacy_state_dict(aliased, tied_heads=True)
                self.assertEqual(set(converted), set(lerobot_names))
                self.assertTrue(torch.equal(converted[VLM_HEAD_KEY], converted[VLM_EMBED_KEY]))
                self.assertIs(converted[EXPERT_HEAD_KEY], aliased[LEGACY_EXPERT_EMBED_KEY])
        state = _fake(_to_legacy_flashvla_names(sorted(self.lerobot["pi05"])))
        state[LANG_EMBEDDER_KEY] = torch.full((1,), 42.0)
        with self.assertRaisesRegex(ValueError, "different values"):
            convert_legacy_state_dict(state, tied_heads=True)


class LoadCheckpointTest(unittest.TestCase):
    def test_pre_lerobot_files_are_refused_with_the_conversion_hint(self) -> None:
        from safetensors.torch import save_file

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.paligemma_with_expert = torch.nn.Module()
        model.model.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            save_file({"model.vlm.weight": torch.zeros(2, 2), "model.vlm.bias": torch.zeros(2)}, path)
            with self.assertRaisesRegex(RuntimeError, "convert_legacy_checkpoint"):
                load_checkpoint(model, path, kind="pi05")
            # a lerobot-named file with a stray key is refused without the hint
            save_file({"model.paligemma_with_expert.paligemma.weight": torch.zeros(2, 2),
                       "model.paligemma_with_expert.paligemma.bias": torch.zeros(2), "model.extra": torch.zeros(1)}, path)
            with self.assertRaisesRegex(RuntimeError, "unexpected 1") as ctx:
                load_checkpoint(model, path, kind="pi05")
            self.assertNotIn("convert_legacy_checkpoint", str(ctx.exception))
            # exact match loads, and reports nothing left fresh
            save_file({"model.paligemma_with_expert.paligemma.weight": torch.ones(2, 2),
                       "model.paligemma_with_expert.paligemma.bias": torch.ones(2)}, path)
            self.assertEqual(load_checkpoint(model, path, kind="pi05"), [])
            self.assertTrue(torch.equal(model.model.paligemma_with_expert.paligemma.weight, torch.ones(2, 2)))


if __name__ == "__main__":
    unittest.main()
