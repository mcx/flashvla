"""The lerobot-config export: stock lerobot 0.5.1 must accept the config and the
(unchanged) tensors of a FlashVLA checkpoint."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType, PolicyFeature

from flashvla.policies.export_lerobot_checkpoint import (
    check_against_lerobot,
    expected_lerobot_shapes,
    lerobot_config_dict,
)
from flashvla.policies.pi0.configuration_pi0 import PI0Config, PI0FlashVLAConfig
from flashvla.policies.pi0.modeling_pi0 import PI0Policy
from flashvla.policies.pi0.modeling_pi0_flashvla import PI0FlashVLAPolicy
from flashvla.policies.pi05.configuration_pi05 import PI05Config, PI05FlashVLAConfig
from flashvla.policies.pi05.modeling_pi05 import PI05Policy
from flashvla.policies.pi05.modeling_pi05_flashvla import PI05FlashVLAPolicy


def _config(config_cls, **overrides):
    config = config_cls(**overrides)
    config.input_features = {
        f"observation.images.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for cam in ("a", "b")
    }
    config.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    config.device = "cpu"
    config.compile_model = False
    return config


class LerobotExportTest(unittest.TestCase):
    def _exported(self, policy_cls, config):
        with torch.device("meta"):
            policy = policy_cls(config)
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)  # flashvla's config.json, as a checkpoint carries it
            flashvla_config = json.loads((Path(tmp) / "config.json").read_text())
            lerobot_config = lerobot_config_dict(flashvla_config)
            (Path(tmp) / "lerobot.json").write_text(json.dumps(lerobot_config))
            expected = expected_lerobot_shapes(Path(tmp) / "lerobot.json", lerobot_config["type"])
        return lerobot_config, dict(policy.state_dict()), expected

    def test_pi05_configs_and_tensors_are_accepted(self) -> None:
        for policy_cls, config_cls in ((PI05Policy, PI05Config), (PI05FlashVLAPolicy, PI05FlashVLAConfig)):
            with self.subTest(policy_cls.__name__):
                lerobot_config, state, expected = self._exported(policy_cls, _config(config_cls))
                self.assertEqual(lerobot_config["type"], "pi05")
                self.assertNotIn("num_buffer_slots", lerobot_config)
                self.assertNotIn("pretrained_path", lerobot_config)
                check_against_lerobot(state, expected)  # raises on any name/shape difference
                self.assertEqual(len(state), 813)

    def test_pi0_configs_and_tensors_are_accepted(self) -> None:
        for policy_cls, config_cls in ((PI0Policy, PI0Config), (PI0FlashVLAPolicy, PI0FlashVLAConfig)):
            with self.subTest(policy_cls.__name__):
                lerobot_config, state, expected = self._exported(policy_cls, _config(config_cls))
                self.assertEqual(lerobot_config["type"], "pi0")
                check_against_lerobot(state, expected)
                self.assertEqual(len(state), 778)

    def test_adarms_pi0_is_refused(self) -> None:
        _, state, expected = self._exported(PI0FlashVLAPolicy, _config(PI0FlashVLAConfig, use_adarms_time_cond=True))
        with self.assertRaisesRegex(ValueError, "does not match lerobot"):
            check_against_lerobot(state, expected)


if __name__ == "__main__":
    unittest.main()
